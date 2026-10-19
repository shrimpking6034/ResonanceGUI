import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication
import resonance_gui as gui
import camera_link_gui


class CameraGUIRegressionTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(gui, 'STATE_DIR', Path(self.tmp.name)),
                        patch.object(gui.MainWindow, '_connect_hardware'),
                        patch.object(gui, 'nidaqmx', None)]
        for p in self.patches:
            p.start()
        self.w = gui.MainWindow()
        self.w._reset_plotter_logs()

    def tearDown(self):
        self.w.close()
        self.app.processEvents()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def enable_fake(self):
        fake = MagicMock()
        fake.snapshot.return_value = dict(state='DISCONNECTED', ready=False, connected=False,
                                         sent=0, accepted=0, saved=0, rejected=0, missed=0,
                                         unknown=0, recent=[], log_error='')
        self.w.camera_mapping.setCurrentIndex(1)
        with patch.object(camera_link_gui, 'CameraLink', return_value=fake):
            self.w.camera_enabled.setChecked(True)
        return fake

    def test_default_off_and_mapping_required(self):
        self.assertFalse(self.w.camera_enabled.isChecked())
        self.assertIsNone(self.w._camera_link)
        with patch.object(camera_link_gui, 'CameraLink') as ctor:
            self.w.camera_enabled.setChecked(True)
            ctor.assert_not_called()
        self.assertFalse(self.w.camera_enabled.isChecked())

    def test_recording_and_pingpong_path_unchanged_when_camera_disconnected(self):
        self.enable_fake()
        self.w.plotter_actual_rate_hz = 100
        self.w._schedule_live_particle_analysis = MagicMock()
        self.w._refresh_plotter_ui = MagicMock()
        self.w.pingpong_active = True
        self.w._pingpong_step = MagicMock()
        self.w.on_plotter_batch([(1., 0., 0., 39000., 0., 0., 0.)])
        self.w._pingpong_step.assert_called_once_with(1., 39000.)
        self.assertEqual(self.w.received_sample_count, 1)
        self.assertEqual(self.w.analysis_f, [39000.])
        self.w.pingpong_active = False

    def test_live_event_once_and_uses_historical_direction(self):
        fake = self.enable_fake()
        self.w._camera_history.observe(0, 2, 1)
        self.w._camera_history.observe(4, 1, 2)
        self.w._camera_history.observe(10, 1, 2)
        self.w.analysis_t = [0, 2, 10]
        self.w.analysis_f = [39000, 38999, 39000]
        payload = dict(mode='detection', analyze_end=9,
                       events=[dict(time_s=2, family='short', depth_hz=1, duration_s=.1, reference_ratio=8)])
        sender = SimpleNamespace(generation=self.w.live_particle_generation)
        with patch.object(self.w, 'sender', return_value=sender):
            self.w.on_live_particle_analysis(payload)
            self.w.on_live_particle_analysis(payload)
        self.assertEqual(len(self.w.live_particle_events), 1)
        fake.submit.assert_called_once()
        msg = fake.submit.call_args[0][0]
        self.assertEqual(msg['direction'], 'LEFT_TO_RIGHT')
        self.assertEqual(msg['left_voltage'], 2)

    def test_switch_guard_only_skips_camera(self):
        fake = self.enable_fake()
        self.w._camera_history.observe(0, 2, 1)
        self.w._camera_history.observe(4, 1, 2)
        self.w._camera_history.observe(10, 1, 2)
        self.w._send_camera_events([dict(time_s=4, duration_s=.1)])
        fake.submit.assert_not_called()
        fake.miss.assert_called_once()

    def test_optional_send_failure_cannot_interrupt_detector(self):
        fake = self.enable_fake()
        fake.submit.side_effect = RuntimeError('camera stopped')
        self.w._camera_history.observe(0, 2, 1)
        self.w._camera_history.observe(10, 2, 1)
        self.w._send_camera_events([dict(time_s=2, duration_s=.1)])
        fake.miss.assert_called_once()
        self.w.camera_enabled.setChecked(False)
        fake.close.assert_called_once()
        self.assertIsNone(self.w._camera_link)


if __name__ == '__main__':
    unittest.main()
