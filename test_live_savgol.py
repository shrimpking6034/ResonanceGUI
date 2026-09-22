import unittest

import numpy as np

from live_savgol import filter_frequency


class LiveSavgolTests(unittest.TestCase):
    def test_disabled_returns_exact_raw_samples(self):
        times = np.arange(100) * .001
        raw = np.sin(times * 100)
        result, window = filter_frequency(times, raw, enabled=False)
        np.testing.assert_array_equal(result, raw)
        self.assertEqual(window, 0)

    def test_enabled_suppresses_high_frequency_noise_and_preserves_short_dip(self):
        times = np.arange(0, 2, .001)
        dip = -2 * np.exp(-.5 * ((times - 1) / .02) ** 2)
        clean = 100 + dip
        noisy = clean + .4 * np.sin(2 * np.pi * 300 * times)
        filtered, window = filter_frequency(times, noisy, enabled=True,
                                            window_length=27, polyorder=4)
        self.assertEqual(window, 27)
        self.assertLess(np.std(filtered[100:500] - clean[100:500]),
                        np.std(noisy[100:500] - clean[100:500]) / 3)
        self.assertLess(abs(np.min(filtered[950:1050]) - np.min(clean[950:1050])), .2)

    def test_even_window_is_raised_to_next_odd_like_offline_panel(self):
        times = np.arange(100) * .001
        raw = np.linspace(1, 2, 100)
        filtered, window = filter_frequency(times, raw, enabled=True,
                                            window_length=26, polyorder=4)
        self.assertEqual(window, 27)
        np.testing.assert_allclose(filtered, raw, atol=1e-10)

    def test_invalid_window_is_reported(self):
        times = np.arange(10) * .001
        with self.assertRaisesRegex(ValueError, "데이터 길이"):
            filter_frequency(times, np.ones(10), enabled=True,
                             window_length=27, polyorder=4)


if __name__ == "__main__":
    unittest.main()
