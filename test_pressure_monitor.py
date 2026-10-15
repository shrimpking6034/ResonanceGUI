import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
import resonance_gui as gui
from pressure_monitor import PressureMonitor, pressure_kpa


class PressureTests(unittest.TestCase):
    def test_nominal_endpoints_and_invalid_signal(self):
        self.assertEqual(pressure_kpa(1), 1)
        self.assertEqual(pressure_kpa(5), 900)
        self.assertEqual(pressure_kpa(3), 450.5)
        for value in (0, 5.1, float("nan"), float("inf")):
            self.assertIsNone(pressure_kpa(value))

    def test_worker_only_configures_ai_and_closes_task(self):
        task = MagicMock()
        task.__enter__.return_value = task
        task.read.return_value = [2, 3]
        worker = PressureMonitor("Dev1")
        with patch("nidaqmx.Task", return_value=task), \
                patch.object(worker, "isInterruptionRequested", side_effect=[False, True]), \
                patch.object(worker, "msleep"):
            worker.run()
        self.assertEqual(task.ai_channels.add_ai_voltage_chan.call_args.args, ("Dev1/ai0:1",))
        task.write.assert_not_called()
        task.ao_channels.add_ao_voltage_chan.assert_not_called()
        task.__exit__.assert_called_once()

    def test_paired_pressure_controls_and_failure(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, "STATE_DIR", Path(folder)), \
                patch.object(gui.MainWindow, "_connect_hardware"), \
                patch.object(gui, "nidaqmx", None):
            window = gui.MainWindow()
            try:
                window.pp_base_v_spin.setValue(.300)
                window.pp_delta_v_spin.setValue(.020)
                self.assertAlmostEqual(window._pressure_pair()[0], .310)
                self.assertAlmostEqual(window._pressure_pair()[1], .290)
                window.pressure_direction_combo.setCurrentIndex(1)
                self.assertAlmostEqual(window._pressure_pair()[0], .290)
                task = MagicMock()
                task.__enter__.return_value = task
                driver = MagicMock()
                driver.Task.return_value = task
                with patch.object(gui, "nidaqmx", driver):
                    self.assertTrue(window._pingpong_apply_direction(False))
                    driver.Task.assert_called_once()
                    self.assertEqual(task.ao_channels.add_ao_voltage_chan.call_count, 2)
                    values = task.write.call_args.args[0]
                    self.assertAlmostEqual(values[0], .290)
                    self.assertAlmostEqual(values[1], .310)
                    before = dict(window.pressure_commands)
                    task.write.side_effect = RuntimeError("write failed")
                    with patch.object(gui.QMessageBox, "critical"):
                        self.assertFalse(window._write_pressure_pair(.4, .2))
                    self.assertEqual(window.pressure_commands, before)
                window.pp_base_v_spin.setValue(0)
                with self.assertRaises(ValueError):
                    window._pressure_pair()
                window.zero_daq_channels()
                self.assertTrue(all(v == 0 for v in window.pressure_commands.values()))
                sections = window.findChildren(gui.CollapsibleSection)
                self.assertGreaterEqual(len(sections), 11)
                for section in sections:
                    section.toggle.setChecked(False)
                    self.assertTrue(section.content.isHidden())
                    section.toggle.setChecked(True)
                    self.assertFalse(section.content.isHidden())
            finally:
                window.close()
                app.processEvents()

    def test_live_depth_setting_invalidates_frozen_calibration(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, "STATE_DIR", Path(folder)), \
                patch.object(gui.MainWindow, "_connect_hardware"), \
                patch.object(gui, "nidaqmx", None):
            window = gui.MainWindow()
            try:
                window.live_particle_thresholds = [{"threshold_hz": .08}]
                generation = window.live_particle_generation
                window.live_floor_spin.setValue(.05)
                self.assertIsNone(window.live_particle_thresholds)
                self.assertGreater(window.live_particle_generation, generation)
                self.assertEqual(window.live_state_value.text(), "학습 대기")
            finally:
                window.close()
                app.processEvents()

    def test_gui_live_metadata_stale_and_error(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, "STATE_DIR", Path(folder)), \
                patch.object(gui.MainWindow, "_connect_hardware"), \
                patch.object(gui, "nidaqmx", None):
            window = gui.MainWindow()
            try:
                window.pressure_monitor_btn.blockSignals(True)
                window.pressure_monitor_btn.setChecked(True)
                window.metadata_actual_pressure_edit.setText("999, 888")
                window._pressure_sample(2, 3, time.monotonic())
                metadata = window._capture_measurement_metadata(False)
                self.assertEqual(metadata["Measured_pressure_A_kPa"], 225.75)
                self.assertEqual(metadata["Measured_pressure_B_kPa"], 450.5)
                self.assertIn("NI AI", metadata["Pressure_source"])
                window._pressure_sample(0, 3, time.monotonic())
                self.assertIsNone(window.measured_pressure_a)
                self.assertEqual(window.measured_pressure_b, 450.5)
                window._pressure_sample(2, 3, time.monotonic() - 2)
                self.assertIsNone(window.measured_pressure_b)
                window._pressure_sample(2, 3, time.monotonic())
                window._pressure_failed("disconnected")
                self.assertIsNone(window.measured_pressure_a)
                self.assertIsNone(window.measured_pressure_b)
            finally:
                window.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
