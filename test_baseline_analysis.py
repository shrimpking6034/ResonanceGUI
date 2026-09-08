import unittest
import numpy as np

from baseline_analysis import (timing_quality, linear_baseline, rolling_metrics,
                               local_event_metrics, false_event_distribution, correlations,
                               detect_particle_candidates, calibrate_waveform_thresholds,
                               detect_waveform_events, evaluate_blank, poisson_rate_summary)


class BaselineAnalysisTests(unittest.TestCase):
    def test_timestamp_rate_and_jitter(self):
        t = np.array([0., .10, .21, .31, .41, .91])
        result = timing_quality(t, .1)
        self.assertAlmostEqual(result["effective_rate_hz"], 10.0)
        self.assertEqual(result["long_gap_count"], 1)

    def test_linear_detrending_uses_time(self):
        t = np.array([0., .1, .25, .6, 1.])
        f = 100. + .2*t
        result = linear_baseline(t, f)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["drift_hz_per_min"], 12.0, places=8)
        self.assertLess(result["detrended_std_hz"], 1e-10)

    def test_event_depth_and_false_distribution(self):
        t = np.arange(0, 3, .05); f = np.full_like(t, 100., dtype=float)
        f[(t >= 1.) & (t <= 1.15)] -= 3.
        event = local_event_metrics(t, f, [(1., 1.15)], baseline_s=.4, guard_s=.05)[0]
        self.assertAlmostEqual(event["DipDepth_Hz"], 3.)
        false = false_event_distribution(t, np.full_like(t, 100.), .15, baseline_s=.4)
        self.assertEqual(false["summary"]["max_hz"], 0.)

    def test_correlation_and_rolling(self):
        t=np.arange(0,20,.1); f=100+t*.01; a=2*f; p=-f
        roll=rolling_metrics(t,f,2.)
        self.assertTrue(np.isfinite(roll["metrics"]["detrended_std_hz"][-1]))
        self.assertAlmostEqual(correlations(f,a,p)["frequency_amplitude"]["r"],1.)

    def test_automatic_candidate_requires_independent_calibration(self):
        t = np.arange(0., 8., .02)
        f = np.full_like(t, 100.)
        f[(t >= 4.) & (t < 4.3)] -= 5.
        result = detect_particle_candidates(t, f, .3, .4, .05)
        self.assertTrue(result["calibration_required"])
        self.assertEqual(result["candidates"], [])

    def test_independent_waveform_calibration_and_metrics(self):
        rng = np.random.default_rng(123)
        t = np.arange(0., 20., .02)
        blank = 100. + rng.normal(0., .01, len(t))
        calibration = calibrate_waveform_thresholds(t, blank)
        self.assertEqual([row["Duration_s"] for row in calibration["thresholds"]], [.3, .5, 1.])
        self.assertTrue(all(row["Percentile"] > 99. for row in calibration["thresholds"]))

        particle = blank.copy()
        mask = (t >= 10.) & (t <= 10.5)
        particle[mask] -= 2. * np.sin(np.pi * (t[mask] - 10.) / .5)
        result = detect_waveform_events(t, particle, calibration["thresholds"], distance_mm=49.)
        self.assertTrue(result["candidates"])
        event = max(result["candidates"], key=lambda row: row["PeakDepth_Hz"])
        self.assertGreater(event["IntegratedDip_Hz_s"], 0.)
        self.assertGreater(event["ActualTransitTime_s"], 0.)
        self.assertAlmostEqual(event["Velocity_mm_s"], 49. / event["ActualTransitTime_s"])

    def test_transient_overlap_is_separated(self):
        rng = np.random.default_rng(5)
        t = np.arange(0., 12., .02)
        blank = 100. + rng.normal(0., .01, len(t))
        calibration = calibrate_waveform_thresholds(t, blank)
        particle = blank.copy(); particle[(t >= 6.) & (t <= 6.4)] -= 2.
        flags = (t >= 6.) & (t <= 6.4)
        result = detect_waveform_events(t, particle, calibration["thresholds"], transient=flags)
        strong_accepted = [row for row in result["candidates"] if row["PeakDepth_Hz"] > 1.]
        strong_excluded = [row for row in result["excluded_transient"] if row["PeakDepth_Hz"] > 1.]
        self.assertFalse(strong_accepted)
        self.assertTrue(strong_excluded)

    def test_evaluation_blank_rate_and_poisson_upper(self):
        rng = np.random.default_rng(7)
        t = np.arange(0., 30., .02)
        calibration = calibrate_waveform_thresholds(t, 100. + rng.normal(0., .02, len(t)))
        evaluation = evaluate_blank(t, np.full_like(t, 100.), calibration)
        self.assertTrue(evaluation["thresholds_frozen"])
        rate = poisson_rate_summary(0, 3 * 3600.)
        self.assertAlmostEqual(rate["poisson_upper_events_per_h"], 0.9986, places=3)


if __name__ == "__main__":
    unittest.main()
