import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(1,r'C:\Users\user\Desktop\김세호\ResonanceGUI')
from datetime import datetime
import tempfile
import unittest
import numpy as np
from PyQt5 import QtWidgets
from particle_review_gui import OverviewTimeAxis,ReviewWindow,local_measurement_start,configure_application


class OverviewTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_application(cls.app)

    def test_local_ticks_fractional_seconds_and_midnight(self):
        axis=OverviewTimeAxis(orientation='bottom')
        start=datetime.fromisoformat('2026-10-15T19:22:52+09:00')
        axis.set_time_display(start,True)
        self.assertEqual(axis.tickStrings([0,50,100],1,50),['19:22:52','19:23:42','19:24:32'])
        self.assertEqual(axis.tickStrings([.00008],1,.00008),['19:22:52.00008'])
        axis.set_time_display(datetime.fromisoformat('2026-10-15T23:59:59+09:00'),True)
        self.assertEqual(axis.tickStrings([2],1,1),['10-16\n00:00:01'])

    def test_no_fabricated_start_time(self):
        self.assertIsNone(local_measurement_start({}))
        self.assertIsNone(local_measurement_start({'Measurement_start_timestamp_local':'invalid'}))

    def test_toggle_only_changes_overview_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            window=ReviewWindow(Path(folder)/'profiles.json')
            window.metadata={'Measurement_start_timestamp_local':'2026-10-15T19:22:52+09:00'}
            window.configure_overview_time_display()
            x=np.array([10.,11.,12.]); curve=window.overview.plot(x,[1,2,1])
            window.overview.setXRange(10,12,padding=0)
            window.region.setBounds([10,12]);window.region.setRegion([10.25,11.25])
            before=window.overview.viewRange()[0][:]
            region=window.region.getRegion()
            window.overview_local_timestamp.setChecked(True)
            self.assertTrue(window.overview_time_axis.show_local)
            self.assertEqual(window.overview_time_axis.tickStrings([10],1,1),['19:23:02'])
            np.testing.assert_array_equal(curve.getData()[0],x)
            np.testing.assert_allclose(window.overview.viewRange()[0],before)
            self.assertEqual(window.region.getRegion(),region)
            self.assertEqual(window.plots[-1].getAxis('bottom').labelText,'Time')
            window.overview_local_timestamp.setChecked(False)
            self.assertFalse(window.overview_time_axis.show_local)
            self.assertEqual(window.overview_time_axis.tickStrings([10],1,1),['10'])
            window.overview_local_timestamp.setChecked(True)
            window.metadata={};window.configure_overview_time_display()
            self.assertFalse(window.overview_local_timestamp.isEnabled())
            self.assertFalse(window.overview_local_timestamp.isChecked())
            window.close()


if __name__=='__main__':unittest.main()
