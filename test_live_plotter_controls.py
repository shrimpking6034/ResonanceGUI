import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
import resonance_gui as gui


class LivePlotterControlTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.folder = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(gui, "STATE_DIR", Path(self.folder.name)),
            patch.object(gui.MainWindow, "_connect_hardware"),
            patch.object(gui, "nidaqmx", None),
        ]
        for item in self.patches:
            item.start()
        self.window = gui.MainWindow()
        self.window.plotter_actual_rate_hz = 12500.0
        self.window.plotter_worker = MagicMock()
        self.window.plotter_worker.isRunning.return_value = True
        self.window.measurement_metadata = {"SessionID": "test"}
        self.window._schedule_live_particle_analysis = MagicMock()
        self.window._refresh_plotter_ui = MagicMock()

    def tearDown(self):
        self.window.plotter_worker = None
        self.window.close()
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def sample(self, time_s):
        return (time_s, float("nan"), float("nan"), 39000.0,
                float("nan"), float("nan"), float("nan"))

    def test_no_disk_recording_still_updates_live_data(self):
        self.window.plotter_record_check.setChecked(False)
        self.window._reset_plotter_logs()
        self.assertIsNone(self.window.plotter_spool)
        self.window.on_plotter_batch([self.sample(0.1)])
        self.assertEqual(self.window.received_sample_count, 1)
        self.assertEqual(self.window.analysis_t, [0.1])
        self.assertEqual(self.window.plot_t, [0.1])

    def test_pause_skips_samples_and_preserves_detector_history(self):
        self.window._reset_plotter_logs()
        self.window.on_plotter_batch([self.sample(0.1)])
        self.window.live_particle_thresholds = [{"threshold_hz": .08}]
        self.window.live_particle_events = [{"time_s": .1, "family": "short"}]
        self.window.toggle_plotter_pause()
        self.assertTrue(self.window.plotter_paused)
        self.window.on_plotter_batch([self.sample(1.0)])
        self.assertEqual(self.window.plotter_spool.rows, 1)
        self.window.toggle_plotter_pause()
        self.assertFalse(self.window.plotter_paused)
        self.assertEqual(self.window.live_particle_events[0]["time_s"], .1)
        self.assertEqual(self.window.live_particle_thresholds[0]["threshold_hz"], .08)
        self.assertEqual(self.window.analysis_t, [])
        self.window.on_plotter_batch([self.sample(1.1 + i * .00008) for i in range(4)])
        self.assertEqual(self.window.plotter_spool.rows, 5)
        self.assertTrue(self.window.analysis_t)
        self.assertGreaterEqual(self.window.analysis_t[0], 1.1)


if __name__ == "__main__":
    unittest.main()
