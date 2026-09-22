import unittest

import numpy as np

from beam_modes import TubeBeam, calculate_modes
from sweep_math import make_filter_plan, settling_multiplier
from resonance_fit import (fit_complex_resonance, model_response,
                           detect_resonance_peaks, assess_fit_quality,
                           ResonanceFit, three_db_bounds)
from pll_design import design_pll


class ResonanceModelTests(unittest.TestCase):
    def test_clamped_clamped_roots(self):
        beam = TubeBeam(0.1, 0.006, 0.004, 200e9, 7850)
        roots = [mode.lambda_n for mode in calculate_modes(beam, 3)]
        np.testing.assert_allclose(roots, [4.73004074, 7.85320462, 10.99560784], rtol=0, atol=1e-6)

    def test_internal_fluid_lowers_frequency(self):
        dry = TubeBeam(0.055, 0.0008, 0.0006, 193e9, 7930, 0)
        water_filled = TubeBeam(0.055, 0.0008, 0.0006, 193e9, 7930, 997)
        self.assertLess(calculate_modes(water_filled, 1)[0].frequency_hz,
                        calculate_modes(dry, 1)[0].frequency_hz)

    def test_settling_and_auto_bandwidth_plan(self):
        self.assertGreater(settling_multiplier(4, 1e-3), settling_multiplier(1, 1e-3))
        plan = make_filter_plan(
            frequency_hz=15000, step_hz=10, order=4, mode="auto", manual_tc_s=0.002,
            fixed_bandwidth_hz=100, max_bandwidth_hz=1000, omega_suppression_db=60,
            bandwidth_overlap=False, min_settling_s=0.001, settling_residual=1e-3,
            sample_count=10, sample_rate_hz=1000, averaging_time_s=0.005, averaging_tc=5,
        )
        self.assertGreater(plan.time_constant_s, 0)
        self.assertGreaterEqual(plan.measurement_s, 0.01)
        self.assertGreaterEqual(plan.settling_s, 0.001)

    def test_complex_fit_recovers_resonance(self):
        f = np.linspace(9900, 10100, 401)
        true = np.array([10000.0, np.log(1200.0), 1.0, -0.3, 0.03, 0.01, 2e-5, -1e-5])
        z = model_response(f, true, np.mean(f))
        fit = fit_complex_resonance(f, z.real, z.imag)
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.f0_hz, 10000.0, places=3)
        self.assertAlmostEqual(fit.q, 1200.0, places=1)

    def test_complex_fit_avoids_low_q_background_cancellation(self):
        f = np.linspace(21112.449799196787, 21409.63855421687, 75)
        parameters = np.array([
            21259.0388, np.log(173.5078),
            6.85996101e-6, -1.05540837e-6,
            -1.94450147e-5, 9.75913344e-5,
            8.22593608e-8, 4.82407594e-8,
        ])
        rng = np.random.default_rng(0)
        z = model_response(f, parameters, np.mean(f))
        z += (rng.normal(0.0, 8.4e-6, len(f))
              + 1j * rng.normal(0.0, 8.4e-6, len(f)))
        missing_sem = np.full(len(f), np.nan)

        fit = fit_complex_resonance(
            f, z.real, z.imag, missing_sem, missing_sem,
            f0_guess=21261.0442, f0_bounds=(21148.5442, 21373.5442),
        )
        quality = assess_fit_quality(f, z.real, z.imag, fit)

        self.assertAlmostEqual(fit.f0_hz, 21259.0388, delta=3.0)
        self.assertGreater(fit.q, 50.0)
        self.assertTrue(quality.accepted, quality.reasons)

    def test_complex_fit_preserves_both_phase_directions_and_count_units(self):
        f = np.linspace(21050.0, 21550.0, 61)
        center = float(np.mean(f))
        f0, q = 21380.0, 99.0
        # Independently construct a response, then conjugate the entire trace
        # to simulate the opposite quadrature convention (including background).
        z = (2.8e5 + 1.1e5j) / (1 - (f / f0) ** 2 + 1j * f / (q * f0))
        z += 1.2e6 - 2.0e6j + (1400.0 + 900.0j) * (f - center)
        sem = np.linspace(700.0, 1000.0, len(f))
        for sign, observed in ((1, z), (-1, z.conjugate())):
            with self.subTest(resonance_sign=sign):
                fit = fit_complex_resonance(
                    f, observed.real, observed.imag, sem, 1.2 * sem,
                    f0_guess=21360.0, f0_bounds=(21200.0, 21500.0),
                )
                self.assertTrue(fit.success)
                self.assertEqual(fit.resonance_sign, sign)
                self.assertAlmostEqual(fit.f0_hz, f0, places=3)
                self.assertAlmostEqual(fit.q, q, places=2)
                predicted = model_response(f, fit.parameters, center, fit.resonance_sign)
                np.testing.assert_allclose(predicted, observed, rtol=1e-5)
                quality = assess_fit_quality(f, observed.real, observed.imag, fit)
                self.assertTrue(quality.accepted, quality.reasons)
                self.assertGreater(quality.complex_r_squared, 0.99999)
                self.assertEqual(np.sign(fit.phase_slope_deg_per_hz), -sign)
                expected_at_f0 = (2.8e5 + 1.1e5j) / (1j / q)
                expected_at_f0 += 1.2e6 - 2.0e6j + (1400.0 + 900.0j) * (f0 - center)
                expected_phase = sign * np.angle(expected_at_f0, deg=True)
                self.assertAlmostEqual(fit.phase_deg, expected_phase, places=3)

    def test_constrained_fit_stays_on_selected_mode(self):
        f = np.linspace(9500, 10500, 1001)
        selected = np.array([9800.0, np.log(800.0), 0.35, 0.1, 0.0, 0.0, 0.0, 0.0])
        dominant_neighbor = np.array([10200.0, np.log(1000.0), 1.2, -0.2, 0.0, 0.0, 0.0, 0.0])
        z = model_response(f, selected, np.mean(f)) + model_response(f, dominant_neighbor, np.mean(f))
        fit = fit_complex_resonance(
            f, z.real, z.imag, f0_guess=9800.0, f0_bounds=(9700.0, 9900.0),
        )
        self.assertTrue(fit.success)
        self.assertGreaterEqual(fit.f0_hz, 9700.0)
        self.assertLessEqual(fit.f0_hz, 9900.0)

    def test_noisy_broad_peak_is_not_split_into_many_candidates(self):
        rng = np.random.default_rng(1234)
        f = np.linspace(28500.0, 30500.0, 500)
        broad_peak = 0.15 / (1.0 + ((f - 29400.0) / 230.0) ** 2)
        amplitude = 0.01 + broad_peak + rng.normal(0.0, 0.004, len(f))
        indices, _, _, noise = detect_resonance_peaks(f, amplitude)
        self.assertGreater(noise, 0)
        self.assertEqual(len(indices), 1)
        self.assertAlmostEqual(f[indices[0]], 29400.0, delta=35.0)

    def test_three_db_bounds_use_absolute_amplitude_and_require_both_sides(self):
        f = np.linspace(-10.0, 10.0, 2001)
        r = 2.0 + 8.0 / np.sqrt(1.0 + (f / 2.0) ** 2)
        bounds = three_db_bounds(f, r, 1000)
        self.assertIsNotNone(bounds)
        expected = 2.0 * np.sqrt((8.0 / (10.0 / np.sqrt(2.0) - 2.0)) ** 2 - 1.0)
        self.assertAlmostEqual(bounds[0], -expected, places=2)
        self.assertAlmostEqual(bounds[1], expected, places=2)
        self.assertIsNone(three_db_bounds(f[1000:], r[1000:], 0))

    def test_fit_quality_rejects_sub_bin_linewidth(self):
        f = np.arange(29380.0, 29424.0, 4.0)
        parameters = np.array([29401.0, np.log(58000.0), 1e-6, 0.0,
                               2e-5, 0.0, 0.0, 0.0])
        z = model_response(f, parameters, np.mean(f))
        fit = ResonanceFit(
            f0_hz=29401.0, q=58000.0, linewidth_hz=29401.0 / 58000.0,
            phase_deg=0.0, phase_slope_deg_per_hz=-0.37,
            f0_std_hz=0.01, q_std=100.0, reduced_chi2=0.0,
            success=True, message="synthetic", parameters=parameters,
        )
        quality = assess_fit_quality(f, z.real, z.imag, fit)
        self.assertFalse(quality.accepted)
        self.assertLess(quality.linewidth_bins, 1.0)

    def test_fit_quality_rejects_linewidth_larger_than_measured_span(self):
        f = np.linspace(38350.0, 39250.0, 151)
        parameters = np.array([39030.0, np.log(0.5), 1.0, 0.0,
                               0.02, 0.01, 1e-5, 0.0])
        z = model_response(f, parameters, np.mean(f))
        fit = ResonanceFit(
            f0_hz=39030.0, q=0.5, linewidth_hz=78060.0,
            phase_deg=0.0, phase_slope_deg_per_hz=0.0,
            f0_std_hz=0.1, q_std=1e-6, reduced_chi2=0.0,
            success=True, message="synthetic", parameters=parameters,
        )
        quality = assess_fit_quality(f, z.real, z.imag, fit)
        self.assertFalse(quality.accepted)
        self.assertGreater(quality.complex_r_squared, 0.999)
        self.assertTrue(any("측정 구간" in reason for reason in quality.reasons))

    def test_fit_quality_accepts_resolved_clean_resonance(self):
        f = np.linspace(9900.0, 10100.0, 401)
        parameters = np.array([10000.0, np.log(500.0), 1.0, -0.2,
                               0.03, 0.01, 1e-5, -1e-5])
        z = model_response(f, parameters, np.mean(f))
        fit = fit_complex_resonance(f, z.real, z.imag)
        quality = assess_fit_quality(f, z.real, z.imag, fit)
        self.assertTrue(quality.accepted, quality.reasons)
        self.assertGreaterEqual(quality.linewidth_bins, 3.0)

    def test_pi_design_is_stable_for_synthetic_plant(self):
        result = design_pll(
            phase_slope_deg_per_hz=-12.0, f0_hz=15813.0, q=1000.0,
            demod_tc_s=0.002, demod_order=2, loop_rate_hz=10000.0,
            target_bandwidth_hz=50.0,
        )
        self.assertTrue(result.stable)
        self.assertGreaterEqual(result.phase_margin_deg, 60.0)
        self.assertEqual(result.i_hz2_per_deg, 0.0)

    def test_screenshot_pll_settings_are_flagged_unstable(self):
        result = design_pll(
            phase_slope_deg_per_hz=-0.3556, f0_hz=29360.672, q=84.51,
            demod_tc_s=0.002, demod_order=2, loop_rate_hz=1.84e6,
            target_bandwidth_hz=3.0, minimum_phase_margin_deg=60.0,
        )
        self.assertFalse(result.stable)
        self.assertLess(result.phase_margin_deg, 0.0)
        self.assertGreater(result.achieved_bandwidth_hz, 100.0)


if __name__ == "__main__":
    unittest.main()
