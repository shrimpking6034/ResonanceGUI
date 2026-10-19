import unittest

import numpy as np

from live_savgol import filter_frequency, filter_frequency_ms, window_samples_for_ms


class LiveSavgolTests(unittest.TestCase):
    def test_fourth_order_large_frequency_preserves_dc_and_dip(self):
        t = np.arange(5000) / 12500
        change = -.05 * np.exp(-.5 * ((t - .2) / .025) ** 2)
        raw = 39061. + change
        original = raw.copy()
        filtered, _ = filter_frequency(t, raw, True, 1251, 4)
        from scipy.signal import savgol_filter
        expected = 39061. + savgol_filter(change, 1251, 4)
        np.testing.assert_allclose(filtered, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(raw, original)
        constant, _ = filter_frequency(t, np.full(len(t), 39061.), True, 1251, 4)
        np.testing.assert_array_equal(constant, np.full(len(t), 39061.))

    def test_other_orders_keep_existing_calculation(self):
        from scipy.signal import savgol_filter
        t = np.arange(2000) / 12500
        raw = 39061. + .01 * np.sin(t * 100)
        for order in (1, 2, 3, 5, 6, 7, 8):
            actual, _ = filter_frequency(t, raw, True, 1251, order)
            np.testing.assert_array_equal(actual, savgol_filter(raw, 1251, order))

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

    def test_ms_window_tracks_the_actual_sample_interval(self):
        fast = np.arange(1000) * .001
        slow = np.arange(1000) * .002
        self.assertEqual(window_samples_for_ms(fast, 20, 4), 21)
        self.assertEqual(window_samples_for_ms(slow, 20, 4), 11)
        raw = np.sin(fast * 100)
        filtered, count = filter_frequency_ms(fast, raw, True, 20, 4)
        expected, _ = filter_frequency(fast, raw, True, 21, 4)
        self.assertEqual(count, 21)
        np.testing.assert_allclose(filtered, expected)


if __name__ == "__main__":
    unittest.main()
