import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PyQt5.QtWidgets import QApplication
from hf2_features import HF2StepWorker, HF2AutoController, HF2AutoParameters, HF2AutoSweepWorker
from fpga_auto_sweep import AutoSweepSettings
from beam_modes import TubeBeam, calculate_modes
import resonance_gui as gui


class FakeHF:
    def __init__(self):
        self.zi = self
        self.phase = 12.0
        self.rate = 1800.0
        self.enabled = False
        self.closed = False
        self.start = time.monotonic()
        self.last = 0
        self.writes = []

    def get_int(self, path):
        return int(self.enabled) if "DEMODS" in path.upper() else 1

    def get_double(self, path):
        return self.phase

    def set_double(self, path, value):
        self.phase = value
        self.writes.append(value)

    def get_demod_rate(self): return self.rate
    def set_demod_rate(self, value): self.rate = value
    def set_demod_enabled(self, value): self.enabled = value
    def get_clockbase(self): return 1000000
    def allocate_event(self): return object()
    def subscribe(self, path): pass
    def unsubscribe(self, path): pass
    def deallocate_event(self, event): pass
    def close(self): self.closed = True

    def read_demod_timestamped(self):
        return (int((time.monotonic()-self.start)*1e6), 0, 0, 0, 0, 0)

    def poll_event(self, event, timeout):
        time.sleep(.005)
        end = int((time.monotonic()-self.start)*1e6)
        stamps = range(self.last+1000, end+1, 1000)
        samples = [(t, 1., 0., 30000+(self.phase-12)*5, 0, 0, 0, 0, 0) for t in stamps]
        if samples: self.last = samples[-1][0]
        return 0, samples


class HF2FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_step_preserves_device_timestamps_and_restores_settings(self):
        hf = FakeHF()
        worker = HF2StepWorker(.18, 7200)
        result, failures = [], []
        worker.completed.connect(result.append)
        worker.failed.connect(failures.append)
        with patch.object(gui, "HFController", return_value=hf):
            worker.run()
        self.assertEqual(failures, [])
        self.assertEqual(hf.phase, 12)
        self.assertEqual(hf.rate, 1800)
        self.assertFalse(hf.enabled)
        self.assertTrue(hf.closed)
        data = result[0]
        self.assertTrue(np.all(np.diff(data["time_s"]) > 0))
        self.assertAlmostEqual(np.median(np.diff(data["time_s"])), .001)
        self.assertEqual(set(data["segment"]), {"baseline", "step", "recovery"})
        self.assertEqual(data["stream_rate_hz"], 7200)

    def test_cancel_restores_setpoint_and_stream_settings(self):
        hf = FakeHF()
        worker = HF2StepWorker(.18, 7200)
        worker.stop()
        failures = []
        worker.failed.connect(failures.append)
        with patch.object(gui, "HFController", return_value=hf):
            worker.run()
        self.assertTrue(failures)
        self.assertEqual(hf.phase, 12)
        self.assertEqual(hf.rate, 1800)
        self.assertTrue(hf.closed)

    def test_hf2_ui_has_auto_sweep_and_step_controls(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(gui, "STATE_DIR", Path(folder)), \
                patch.object(gui.MainWindow, "_connect_hardware", lambda self: None):
            window = gui.MainWindow()
            try:
                self.assertEqual(window.tabs.count(), 6)
                self.assertIn("hf2_step_spin", window._settings_widgets())
                self.assertIn("auto_first_mode", window._settings_widgets())
                self.assertEqual(window._hf2_sweep_settings()["filter_mode"], window.sweep_filter_mode_combo.currentText())
            finally:
                window.close()

    def test_auto_segment_uses_hf2_filters_and_voltage_export(self):
        from PyQt5.QtCore import QThread, pyqtSignal
        class Segment(QThread):
            point_ready = pyqtSignal(int, float, float, float, float, float)
            finished_ok = pyqtSignal(object)
            failed = pyqtSignal(str)
            def __init__(self, low, high, count, settings):
                super().__init__()
                self.grid = np.linspace(low, high, count)
                self.settings = settings
                self._stop = False
            def stop(self): self._stop = True
            def run(self):
                f = self.grid
                z = .01/(1+1j*(f-f.mean())/20)
                for i in range(len(f)):
                    self.point_ready.emit(i, f[i], abs(z[i]), np.angle(z[i],deg=True), z[i].real, z[i].imag)
                data = dict(freqs=f, R=abs(z), theta=np.angle(z,deg=True), X=z.real, Y=z.imag,
                            X_sem=np.ones(len(f))*1e-6, Y_sem=np.ones(len(f))*1e-6,
                            XY_cov=np.zeros(len(f)), n_samples=np.ones(len(f))*64)
                self.finished_ok.emit(data)
        controller = HF2AutoController(.1)
        modes = calculate_modes(TubeBeam(.049,.001,.0008,200e9,7800,1000), 1)
        with tempfile.TemporaryDirectory() as folder:
            worker = HF2AutoSweepWorker(controller, HF2AutoParameters(1000,900,1100,.1), modes,
                                        AutoSweepSettings(first_mode=1,last_mode=1), folder,
                                        sweep_settings={"filter_mode":"Auto"})
            record = dict(mode=1,theory_hz=1000,status="running",notes=[],stages=[])
            worker.records = [record]
            with patch.object(gui,"SweepWorker",Segment), patch.object(controller,"enable"):
                data = worker._measure(record,"coarse",900,1100,10)
            self.assertEqual(len(data["freqs"]),21)
            text = (Path(folder)/"mode_01_coarse.csv").read_text(encoding="utf-8-sig")
            self.assertIn("Amplitude_R_V",text)
            self.assertIn("Drive_Vpk",text)
            self.assertNotIn("Amplitude_R_counts",text)
