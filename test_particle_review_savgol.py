import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from particle_review_savgol_gui import SavgolPanel


class SavgolPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def data():
        t = np.arange(0, 2, .001)
        f = 100 + np.sin(2 * np.pi * 50 * t) + .2 * np.sin(2 * np.pi * 250 * t)
        return pd.DataFrame(dict(Time_s=t, Frequency_Hz=f))

    def test_panel_converts_ms_to_odd_samples(self):
        panel = SavgolPanel()
        events = [dict(time_s=1., event_start_s=.94, event_end_s=1.06)]
        panel.set_source(self.data(), events, 0)
        panel.window.setValue(26)
        self.assertEqual(panel.window.value(), 26)
        self.assertEqual(panel.get_settings()['window_length'], 27)
        self.assertEqual(panel.order.value(), 4)
        self.assertIsNotNone(panel.filtered)
        self.assertEqual(len(panel.filtered), len(self.data()))
        self.assertIn('27.000 ms', panel.info.text())
        panel.close()

    def test_ms_conversion_tracks_sampling_rate_and_minimum_order(self):
        panel=SavgolPanel()
        data=self.data(); data['Time_s']/=3.051850948
        panel.set_source(data,[])
        panel.window.setValue(50)
        self.assertEqual(panel.get_settings()['window_length'],153)
        self.assertAlmostEqual(panel.get_settings()['applied_window_ms'],50.1335,places=3)
        panel.window.setValue(.001)
        self.assertEqual(panel.get_settings()['window_length'],5)
        panel.window.setValue(1000)
        self.assertIsNone(panel.filtered)
        with self.assertRaises(ValueError):panel.get_settings()
        panel.close()

    def test_legacy_sample_settings_migrate_to_ms(self):
        panel=SavgolPanel(); panel.set_source(self.data(),[])
        panel.set_settings({'order':4,'window_length':27})
        self.assertEqual(panel.window.value(),27)
        self.assertEqual(panel.get_settings()['window_length'],27)
        panel.close()

    def test_review_window_adds_tab_and_persists_settings(self):
        from unittest.mock import patch
        from particle_review_gui import ReviewWindow
        from particle_review_core import manual_event

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'raw.csv'; self.data().to_csv(source, index=False)
            window = ReviewWindow(); window.error = lambda message: self.fail(message); window.load_file(source)
            window.events = [manual_event(window.data, 1.)]; window.refresh_table(); window.table.selectRow(0)
            labels = [window.right_tabs.tabText(i) for i in range(window.right_tabs.count())]
            self.assertIn('Savitzky–Golay 비교', labels)
            window.savgol_panel.order.setValue(4); window.savgol_panel.window.setValue(27)
            target = Path(folder) / 'review.csv'
            with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(str(target), 'CSV')):
                window.export_results()
            window.savgol_panel.order.setValue(2); window.savgol_panel.window.setValue(9)
            window.events = []; window.refresh_table()
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(target), 'CSV')):
                window.import_review()
            self.assertEqual(window.savgol_panel.get_settings()['order'], 4)
            self.assertEqual(window.savgol_panel.get_settings()['window_length'], 27)
            self.assertEqual(window.savgol_panel.get_settings()['window_ms'],27)
            window.close(); self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
