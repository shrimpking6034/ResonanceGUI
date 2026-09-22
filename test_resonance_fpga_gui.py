import os
import time
import unittest
import csv
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from fpga_backend import FPGAController, StreamBatch
from fpga_transport import NetworkProfile
import resonance_gui as hf_module
from resonance_fpga_gui import FPGAMainWindow


class FakeFPGA(FPGAController):
    def __init__(self):
        super().__init__(NetworkProfile())
        self.running = False
        self.tracking = False
        self.signal = "frequency"
        self.error_mode = False
        self.delay_turns = 0.0
        self.flat_error = False
        self.frequency = 118000.0
        self.next_batch = 0.0
        self.packet = 0
        self.stream_starts = 0
        self.stream_stops = 0

    def connect(self):
        return "SMR Controller Host Ready\nRunning version test"

    def disconnect(self):
        self.running = False

    def write_named(self, name, value):
        if name == "delay":
            self.delay_turns = value / (2**32 - 1)
        return "wrote"

    def start_stream(self, *, signal="frequency"):
        self.signal = signal
        self.stream_starts += 1

    def stop_stream(self):
        self.stream_stops += 1

    def apply_verified_sweep_preset(self, minimum_hz, maximum_hz, drive_fraction):
        self.running = False
        self.frequency = minimum_hz
        self.error_mode = False

    def apply_parameters(self, parameters, **kwargs):
        self.running = False
        self.frequency = parameters.frequency_hz
        self.last_parameters = parameters
        self.error_mode = parameters.mode.signal_of_interest == 1
        self.delay_turns = parameters.pll_delay_turns

    def set_fixed_frequency(self, frequency_hz):
        self.frequency = frequency_hz

    def start_fixed(self, **kwargs):
        self.running = True
        self.tracking = False
        if self.last_parameters is not None:
            self.last_parameters = replace(
                self.last_parameters, mode=replace(self.last_parameters.mode, run=True,
                                                   run_nco_at_fixed_frequency=True))

    def start_tracking(self, **kwargs):
        self.running = True
        self.tracking = True
        self.last_parameters = replace(
            self.last_parameters, mode=replace(self.last_parameters.mode, run=True,
                                               run_nco_at_fixed_frequency=False))

    def restart_all_plls(self):
        pass

    def stop_output(self):
        self.running = False
        self.tracking = False
        if self.last_parameters is not None:
            self.last_parameters = replace(
                self.last_parameters, mode=replace(self.last_parameters.mode, run=False))

    def drain_stream(self, max_batches=100):
        if not self.running or time.monotonic() < self.next_batch:
            return []
        self.next_batch = time.monotonic() + 0.01
        self.packet += 1
        if self.signal == "signed" and self.error_mode:
            raw_error = (0.0 if self.flat_error else
                         np.sin(2*np.pi*(self.delay_turns - 0.053))
                         * 30_000_000 + 5 * (self.frequency - 118000))
            words = np.full(128, raw_error, float)
        elif self.signal == "signed":
            q = 150.0
            z = 1000 + 40000 / (1 + 1j*2*q*(self.frequency-118000)/118000)
            words = np.tile([z.real, z.imag], 64).astype(float)
        else:
            words = np.full(128, 118000.0, float)
        return [StreamBatch(0, 0, self.packet, "header_unverified", 0,
                            time.monotonic(), words)]


class SameLayoutFPGATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.state_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.state_temp.cleanup)
        self.state_patch = patch.object(hf_module, "STATE_DIR",
                                        Path(self.state_temp.name))
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_sweep_and_plotter_use_original_layout(self):
        fake = FakeFPGA()
        dialogs = []
        def connect(window):
            window.fpga = fake
            window.status_label.setText("FPGA test connected")
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            self.assertEqual(window.tabs.count(), 5)
            self.assertEqual(window.pll_order_spin.value(), 1)
            self.assertFalse(window.input_range_spins[0].isEnabled())
            window.f_start_spin.setValue(117000)
            window.f_stop_spin.setValue(119000)
            window.n_points_spin.setValue(10)
            window.sweep_samples_spin.setValue(64)
            window.run_sweep()
            deadline = time.monotonic() + 10
            while window.sweep_worker.isRunning() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.app.processEvents()
            self.assertEqual(len(window.sweep_metadata["freqs"]), 10)
            with tempfile.TemporaryDirectory() as directory:
                sweep_path = Path(directory) / "sweep.csv"
                with patch.object(QFileDialog, "getSaveFileName",
                                  return_value=(str(sweep_path), "CSV Files (*.csv)")):
                    window.export_sweep_csv()
                with sweep_path.open(encoding="utf-8-sig", newline="") as handle:
                    sweep_rows = list(csv.reader(handle))
                self.assertIn("Amplitude_R_counts", sweep_rows[2])
            window.pll_center_spin.setValue(118000)
            window.pll_range_spin.setValue(1000)
            window.pll_q_spin.setValue(150)
            window.fpga_pll_delay_spin.setValue(0.125)
            window.fpga_cic_rate_spin.setValue(16384)
            window.fpga_cic_shift_spin.setValue(20)
            window.fpga_agc_check.setChecked(True)
            window.fpga_auto_delay_check.setChecked(False)
            with patch.object(QMessageBox, "critical",
                              lambda _parent, title, message: dialogs.append((title, message))):
                window.pll_enable_btn.setChecked(True)
            self.assertEqual(dialogs, [])
            self.assertFalse(fake.last_parameters.mode.run_nco_at_fixed_frequency)
            self.assertTrue(fake.last_parameters.mode.enable_agc)
            self.assertEqual(fake.last_parameters.pll_delay_turns, 0.125)
            self.assertEqual(fake.last_parameters.cic_rate, 16384)
            self.assertEqual(fake.last_parameters.cic_bit_shift, 20)
            deadline = time.monotonic() + 0.9
            while time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertGreater(window.received_sample_count, 128)
            self.assertEqual(window.measurement_metadata["Backend"],
                             "SMR FPGA channel A")
            self.assertEqual(window.measurement_metadata["FPGA_PLL_gain_formula"],
                             "LabVIEW VI frequency-and-CIC-dependent")
            self.assertGreater(window.measurement_metadata["FPGA_PLL_gain_integral_word"], 0)
            self.assertIn("적용 BW", window.pll_locked_label.text())
            applied_bw = fake.last_parameters.loop_bandwidth_hz
            window.pll_target_bw_spin.setValue(applied_bw + 5)
            self.assertIn("PLL 재시작 필요", window.pll_locked_label.text())
            self.assertEqual(window.measurement_metadata["PLL_target_bandwidth_Hz"], applied_bw)
            self.assertTrue(np.isnan(window.log_pll_locked[-1]))
            self.assertTrue(np.isnan(window.log_pll_target_phase[-1]))
            self.assertTrue(np.isnan(window.measurement_metadata["Demod_stream_rate_Hz_actual"]))
            with patch.object(QMessageBox, "critical",
                              lambda _parent, title, message: dialogs.append((title, message))):
                window.start_baseline_test()
                deadline = time.monotonic() + 1.2
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
            self.assertEqual(dialogs, [])
            self.assertGreater(window.received_sample_count, 128)
            self.assertTrue(window.baseline_active)
            window.stop_baseline_test()
            directions = []
            with patch.object(window, "_pingpong_apply_direction",
                              lambda forward: directions.append(forward)), patch.object(
                                  window, "zero_daq_channels", lambda: None):
                window.start_pingpong()
                self.assertTrue(window.pingpong_active)
                self.assertEqual(directions, [True])
                window.stop_pingpong()
        finally:
            window.close()

    def test_plotter_restarts_while_pll_keeps_tracking(self):
        fake = FakeFPGA()
        def connect(window):
            window.fpga = fake
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            window.fpga_auto_delay_check.setChecked(False)
            window.pll_center_spin.setValue(118000)
            window.pll_enable_btn.setChecked(True)
            deadline = time.monotonic() + 1.5
            while window.received_sample_count == 0 and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertGreater(window.received_sample_count, 0)
            window.stop_plotter()
            self.assertTrue(fake.running)
            self.assertTrue(fake.tracking)
            stops = fake.stream_stops
            starts = fake.stream_starts
            window.live_frequency_curve.setVisible(False)
            window.live_particle_plot.setXRange(262, 271, padding=0)
            window.live_savgol_enabled.setChecked(True)
            window.start_plotter()
            deadline = time.monotonic() + 1.5
            while window.received_sample_count == 0 and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertGreater(fake.stream_stops, stops)
            self.assertGreater(fake.stream_starts, starts)
            self.assertGreater(window.received_sample_count, 0)
            self.assertTrue(fake.tracking)
            self.assertTrue(window.live_frequency_curve.isVisible())
            self.assertTrue(window.live_filtered_frequency_curve.isVisible())
            self.assertLess(window.live_particle_plot.viewRange()[0][0], 1)
            self.assertGreater(len(window.live_frequency_curve.getData()[0]), 0)
            self.assertGreater(len(window.live_filtered_frequency_curve.getData()[0]), 0)
        finally:
            window.close()

    def test_raw_error_scan_uses_vi_signal_and_stops_output(self):
        fake = FakeFPGA()
        def connect(window):
            window.fpga = fake
            window.status_label.setText("FPGA test connected")
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            window.last_resonance_fit = SimpleNamespace(
                f0_hz=118000.0, linewidth_hz=40.0)
            window.pll_center_spin.setValue(118000)
            window.pll_range_spin.setValue(200)
            window.calibrate_pll_phase_detector()
            deadline = time.monotonic() + 8
            while window.fpga_error_worker.isRunning() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.app.processEvents()
            self.assertFalse(window.fpga_error_worker.isRunning())
            self.assertEqual(fake.signal, "signed")
            self.assertEqual(fake.last_parameters.mode.signal_of_interest, 1)
            self.assertFalse(fake.running)
            self.assertIn("counts/Hz", window.pll_design_label.text())
        finally:
            window.close()

    def test_pll_start_calibrates_positive_error_zero_then_tracks(self):
        fake = FakeFPGA()
        dialogs = []
        def connect(window):
            window.fpga = fake
            window.status_label.setText("FPGA test connected")
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            window.pll_center_spin.setValue(118000)
            window.pll_range_spin.setValue(500)
            window.pll_q_spin.setValue(140)
            window.drive_v_spin.setValue(0.005)
            with patch.object(QMessageBox, "critical",
                              lambda _parent, title, message: dialogs.append((title, message))), patch.object(
                                  QMessageBox, "warning",
                                  lambda _parent, title, message: dialogs.append((title, message))):
                window.pll_enable_btn.setChecked(True)
                deadline = time.monotonic() + 12
                while not fake.tracking and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()
            self.assertEqual(dialogs, [])
            self.assertTrue(fake.tracking)
            self.assertTrue(window.pll_enable_btn.isChecked())
            self.assertAlmostEqual(window.fpga_pll_delay_spin.value(), 0.053, delta=0.003)
            self.assertAlmostEqual(window._last_delay_calibration["fine_step_turns"], 1 / 256)
            self.assertAlmostEqual(window._last_delay_calibration["coarse_delay_turns"], 0.053,
                                   delta=0.003)
            self.assertTrue(window.measurement_metadata["FPGA_PLL_delay_auto_calibrated"])
        finally:
            window.close()

    def test_missing_error_response_keeps_pll_off(self):
        fake = FakeFPGA()
        fake.flat_error = True
        dialogs = []
        def connect(window):
            window.fpga = fake
            window.status_label.setText("FPGA test connected")
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            window.pll_center_spin.setValue(118000)
            window.pll_q_spin.setValue(140)
            with patch.object(QMessageBox, "warning",
                              lambda _parent, title, message: dialogs.append((title, message))):
                window.pll_enable_btn.setChecked(True)
                deadline = time.monotonic() + 12
                while window.fpga_delay_worker is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()
            self.assertFalse(window.pll_enable_btn.isChecked())
            self.assertFalse(fake.tracking)
            self.assertFalse(fake.running)
            self.assertTrue(dialogs)
        finally:
            window.close()

    def test_cancel_delay_calibration_does_not_start_tracking(self):
        fake = FakeFPGA()
        def connect(window):
            window.fpga = fake
            window.status_label.setText("FPGA test connected")
        with patch.object(FPGAMainWindow, "_connect_hardware", connect):
            window = FPGAMainWindow()
        try:
            window.pll_center_spin.setValue(118000)
            window.pll_q_spin.setValue(140)
            window.pll_enable_btn.setChecked(True)
            self.app.processEvents()
            self.assertTrue(window.fpga_delay_worker.isRunning())
            window.pll_enable_btn.setChecked(False)
            self.app.processEvents()
            self.assertFalse(fake.tracking)
            self.assertFalse(fake.running)
            self.assertFalse(window.pll_enable_btn.isChecked())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
