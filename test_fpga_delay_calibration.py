import unittest

import numpy as np

from fpga_delay_calibration import positive_error_zero, refine_positive_error_zero


class DelayCalibrationTests(unittest.TestCase):
    def test_selects_positive_slope_branch(self):
        delays = np.arange(16) / 16
        for expected in (0.053, 0.8485):
            errors = np.sin(2 * np.pi * (delays - expected)) * 30_000_000
            actual, span = positive_error_zero(delays, errors, noise_counts=100_000)
            self.assertLess(abs((actual - expected + 0.5) % 1 - 0.5), 0.002)
            self.assertGreater(span, 50_000_000)

    def test_rejects_noisy_or_absent_error_response(self):
        delays = np.arange(16) / 16
        with self.assertRaises(ValueError):
            positive_error_zero(delays, np.zeros(16))
        with self.assertRaises(ValueError):
            positive_error_zero(delays, np.linspace(-1, 1, 16), noise_counts=1)

    def test_fine_scan_refines_crossing_including_wraparound(self):
        coarse = np.arange(16) / 16
        for expected in (0.053, 0.997):
            coarse_errors = np.sin(2 * np.pi * (coarse - expected)) * 30_000_000
            candidate, _ = positive_error_zero(coarse, coarse_errors)
            start = np.floor(candidate * 16) / 16
            fine = start + np.linspace(0.0, 1.0 / 16.0, 17)
            fine_errors = np.sin(2 * np.pi * (fine - expected)) * 30_000_000
            refined, _ = refine_positive_error_zero(candidate, fine, fine_errors)
            self.assertLess(abs((refined - expected + 0.5) % 1 - 0.5), 1e-5)

    def test_fine_scan_rejects_missing_positive_crossing(self):
        fine = np.linspace(0.0, 1.0 / 16.0, 17)
        with self.assertRaises(ValueError):
            refine_positive_error_zero(0.03, fine, np.full(17, 2_000_000))
        with self.assertRaises(ValueError):
            refine_positive_error_zero(0.03, fine, np.linspace(2_000_000, -2_000_000, 17))


if __name__ == "__main__":
    unittest.main()
