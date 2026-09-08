import unittest

import numpy as np

from beam_modes import TubeBeam, calculate_modes
from sweep_math import make_filter_plan, settling_multiplier
from resonance_fit import (fit_complex_resonance, model_response,
                           detect_resonance_peaks, assess_fit_quality,
                           ResonanceFit)
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
