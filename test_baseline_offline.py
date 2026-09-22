import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from baseline_analysis import allan_deviation, linear_baseline, rolling_metrics
from baseline_offline import analyze_noise, load_noise_csv, save_noise_result, offline_rolling, offline_allan
from baseline_offline_gui import BaselineOfflineWindow


class BaselineOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "noise.csv"
        self.t = np.arange(0, 4, .02)
        self.f = 1000 + .2 * self.t + np.random.default_rng(23).normal(0, .01, len(self.t))
        self.write_csv(self.t, self.f)

    def write_csv(self, t, f):
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["# Metadata", "Plotter_requested_interval_ms", "20"])
            writer.writerow(["Time_s", "Frequency_Hz"])
            writer.writerows(zip(t, f))

    def test_range_and_existing_analysis_agree_without_changing_source(self):
        source = load_noise_csv(self.path)
        result = analyze_noise(source, 1, 3, .5)
        mask = (self.t >= 1) & (self.t <= 3)
        expected = linear_baseline(self.t[mask], self.f[mask])
        self.assertAlmostEqual(result["global"]["detrended_std_hz"], expected["detrended_std_hz"])
        self.assertAlmostEqual(result["global"]["drift_hz_per_min"], expected["drift_hz_per_min"])
        self.assertEqual(result["timing"]["nominal_interval_s"], .02)
        self.assertEqual(result["correlation"]["frequency_amplitude"]["n"], 0)
        expected_allan = allan_deviation(self.t[mask] - 1, self.f[mask])
        np.testing.assert_allclose([r["allan_deviation_hz"] for r in result["allan"]],
                                   [r["allan_deviation_hz"] for r in expected_allan], atol=1e-10)
        np.testing.assert_array_equal(source["columns"]["Frequency_Hz"], self.f)

    def test_fast_estimators_match_legacy_for_irregular_samples_and_gaps(self):
        rng = np.random.default_rng(92)
        t = np.cumsum(rng.uniform(.01, .04, 800))
        t[400:] += 2
        f = 1e6 + .3 * t + rng.normal(0, .02, len(t))
        for window in (.03, .3, 10):
            expected = rolling_metrics(t, f, window)
            actual = offline_rolling(t, f, window)
            for key in expected["metrics"]:
                np.testing.assert_allclose(actual["metrics"][key], expected["metrics"][key],
                                           atol=1e-5, rtol=1e-5, equal_nan=True, err_msg=key)
        expected, actual = allan_deviation(t, f), offline_allan(t, f)
        self.assertEqual([r["cluster_count"] for r in actual], [r["cluster_count"] for r in expected])
        np.testing.assert_allclose([r["allan_deviation_hz"] for r in actual],
                                   [r["allan_deviation_hz"] for r in expected], atol=1e-9)

    def test_reject_bad_timestamps_and_nonfinite_frequency(self):
        for t, f in (([0, 0, 1], [1, 2, 3]), ([0, 2, 1], [1, 2, 3]),
                     ([0, 1, 2], [1, np.nan, 3])):
            with self.subTest(t=t, f=f):
                self.write_csv(t, f)
                with self.assertRaises(ValueError):
                    load_noise_csv(self.path)

    def test_optional_auxiliary_channels_and_phase_error_preserved(self):
        self.path.write_text(
            "Time_s,Frequency_Hz,Amplitude_R_V,Phase_deg,PLL_PhaseError_deg\n"
            "0,100,1,2,3\n1,101,2,4,6\n2,102,3,6,9\n", encoding="utf-8")
        result = analyze_noise(load_noise_csv(self.path))
        self.assertAlmostEqual(result["correlation"]["frequency_amplitude"]["r"], 1)
        np.testing.assert_array_equal(result["columns"]["PLL_PhaseError_deg"], [3, 6, 9])

    def test_invalid_range_or_window(self):
        source = load_noise_csv(self.path)
        for args in ((3, 1, 10), (20, 30, 10), (0, 1, 0)):
            with self.assertRaises(ValueError):
                analyze_noise(source, *args)

    def test_exports_are_reproducible_and_do_not_overwrite_source(self):
        original = self.path.read_bytes()
        result = analyze_noise(load_noise_csv(self.path), rolling_s=.5)
        first = save_noise_result(result, self.temp.name)
        second = save_noise_result(result, self.temp.name)
        self.assertNotEqual(first, second)
        summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["source_path"], str(self.path.resolve()))
        self.assertIsNone(summary["correlation"]["frequency_amplitude"]["r"])
        self.assertTrue((first / "allan.png").is_file())
        with (first / "derived.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(self.t))
        np.testing.assert_allclose([float(row["Frequency_Hz"]) for row in rows], self.f)
        self.assertEqual(self.path.read_bytes(), original)

    def wait_for_job(self, window):
        deadline = time.monotonic() + 15
        while window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(window.worker, "worker did not finish")

    def test_gui_load_analyze_and_invalidate(self):
        window = BaselineOfflineWindow()
        self.addCleanup(window.close)
        window.load_path(str(self.path))
        self.wait_for_job(window)
        self.assertTrue(window.analyze_button.isEnabled())
        self.assertFalse(window.save_button.isEnabled())
        window.analyze()
        self.assertFalse(window.controls.isEnabled())
        self.wait_for_job(window)
        self.assertIsNotNone(window.result)
        self.assertTrue(window.save_button.isEnabled())
        # View clipping/downsampling must not reduce the underlying analysis series.
        self.assertEqual(len(window.residual_curve.xData), len(self.t))
        self.assertEqual(len(window.result["columns"]["Time_s"]), len(self.t))
        window.rolling_spin.setValue(2)
        self.assertIsNone(window.result)
        self.assertFalse(window.save_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
