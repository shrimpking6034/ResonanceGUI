import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
from unittest.mock import patch
import tempfile
import time
import unittest
import numpy as np
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5.QtWidgets import QApplication
from fpga_registers import PLLUserParameters
from fpga_initial_resonance import InitialResonanceWorker,isolated_peak,verification_bounds
from test_fpga_pll_refinement import FakeController


class SyntheticScan:
    centers=[38450.]
    def __init__(self,controller,p,grid,settle,samples):
        self.grid=grid; self.stop_requested=False; self.last_result={}
    def stop(self): self.stop_requested=True
    def measure(self):
        f=self.grid
        z=sum(1e6/(1-1j*2*100*(f-c)/c) for c in self.centers)
        self.last_result=dict(freqs=f,R=abs(z),theta=np.angle(z,deg=True),X=z.real,Y=z.imag,
            X_sem=np.ones(len(f))*10,Y_sem=np.ones(len(f))*10)


class InitialResonanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_wrong_initial_frequency_is_corrected_before_pll(self):
        c=FakeController()
        p=PLLUserParameters(38000,37000,39500,100,55)
        with tempfile.TemporaryDirectory() as folder, patch("fpga_initial_resonance.SweepWorker",SyntheticScan):
            worker=InitialResonanceWorker(c,p,low=37000,high=39500,reference=38000,directory=folder)
            results=[]; failures=[]
            worker.completed.connect(results.append); worker.failed.connect(failures.append)
            worker.run()
            self.assertEqual(failures,[])
            self.assertAlmostEqual(results[0]["frequency_hz"],38450,delta=10)
            self.assertAlmostEqual(results[0]["q"],100,delta=3)
            self.assertTrue(c.stopped)
            self.assertTrue(list(Path(folder).glob("*_bounded_retry.npz")))

    def test_broad_resonance_uses_sweeper_coverage_not_capture_range(self):
        class Broad(SyntheticScan):
            def measure(self):
                f=self.grid
                z=1e8/(1-1j*2*24*(f-38900)/38900)
                self.last_result=dict(freqs=f,R=abs(z),theta=np.angle(z,deg=True),X=z.real,Y=z.imag,
                    X_sem=np.ones(len(f))*10,Y_sem=np.ones(len(f))*10)
        c=FakeController()
        p=PLLUserParameters(39000,38500,39500,24,80)
        low,high=verification_bounds(39000,36000,41000,[38900])
        self.assertEqual((low,high),(36000,41000))
        narrow=np.linspace(38500,39500,161)
        with self.assertRaisesRegex(ValueError,"−3 dB"):
            isolated_peak(dict(freqs=narrow,R=abs(1e8/(1-1j*48*(narrow-38900)/38900))),39000,38500,39500)
        with tempfile.TemporaryDirectory() as folder,patch("fpga_initial_resonance.SweepWorker",Broad):
            worker=InitialResonanceWorker(c,p,low=low,high=high,reference=39000,directory=folder)
            results=[]; failures=[]
            worker.completed.connect(results.append);worker.failed.connect(failures.append)
            worker.run()
            self.assertEqual(failures,[])
            self.assertAlmostEqual(results[0]["frequency_hz"],38900,delta=20)
            self.assertEqual(p.maximum_frequency_hz-p.minimum_frequency_hz,1000)

    def test_sweeper_coverage_respects_known_neighbor_midpoints(self):
        self.assertEqual(verification_bounds(39000,30000,50000,[35000,39000,43000]),(37000,41000))
        with self.assertRaises(ValueError):verification_bounds(39000,1000,2000,[])

    def test_no_peak_or_cancellation_never_enables_tracking(self):
        class Flat(SyntheticScan):
            def measure(self):
                super().measure(); self.last_result["R"]=np.ones(len(self.grid))
        for cancelled in (False,True):
            c=FakeController()
            with tempfile.TemporaryDirectory() as folder, patch("fpga_initial_resonance.SweepWorker",Flat):
                worker=InitialResonanceWorker(c,PLLUserParameters(38000,37000,39500,100,55),
                    low=37000,high=39500,reference=38000,directory=folder)
                if cancelled: worker.stop()
                errors=[]; worker.failed.connect(errors.append); worker.run()
                self.assertTrue(errors)
                self.assertTrue(c.stopped)

    def test_multiple_significant_peaks_are_rejected(self):
        f=np.linspace(37000,39500,501)
        r=1+10*np.exp(-((f-37800)/100)**2)+9*np.exp(-((f-38800)/100)**2)
        with self.assertRaisesRegex(ValueError,"여러"):
            isolated_peak(dict(freqs=f,R=r),38000,37000,39500)

    def test_failed_closed_loop_validation_stops_output_and_blocks_recording(self):
        import resonance_gui as gui
        from resonance_fpga_gui import FPGAMainWindow
        from test_resonance_fpga_gui import FakeFPGA
        c=FakeFPGA()
        with tempfile.TemporaryDirectory() as folder, patch.object(gui,"STATE_DIR",Path(folder)), \
             patch.object(FPGAMainWindow,"_connect_hardware",lambda window:setattr(window,"fpga",c)):
            window=FPGAMainWindow()
            try:
                window.pll_enable_btn.blockSignals(True); window.pll_enable_btn.setChecked(True)
                window.pll_enable_btn.blockSignals(False)
                window._peak_outcome=(False,"평탄한 진폭")
                window._peak_finished()
                self.assertFalse(window.pll_enable_btn.isChecked())
                self.assertFalse(c.running)
                window.start_plotter(); window.start_plotter(baseline=True)
                self.assertIsNone(window.plotter_worker)
                self.assertTrue(window._pll_validation_failed)
            finally: window.close()

    def test_preflight_failure_stops_before_error_calibration(self):
        import resonance_gui as gui
        from resonance_fpga_gui import FPGAMainWindow
        from test_resonance_fpga_gui import FakeFPGA
        class Failure(QThread):
            completed=pyqtSignal(object); failed=pyqtSignal(str); progress=pyqtSignal(str)
            def __init__(self,*args,**kwargs): super().__init__()
            def stop(self): pass
            def run(self): self.failed.emit("공진 피크 없음")
        c=FakeFPGA()
        with tempfile.TemporaryDirectory() as folder, patch.object(gui,"STATE_DIR",Path(folder)), \
             patch.object(FPGAMainWindow,"_connect_hardware",lambda window:setattr(window,"fpga",c)), \
             patch("fpga_refinement_gui.InitialResonanceWorker",Failure), \
             patch.object(FPGAMainWindow,"_start_fpga_delay_calibration") as calibrate:
            window=FPGAMainWindow()
            try:
                window.pll_enable_btn.setChecked(True)
                deadline=time.monotonic()+2
                while window.fpga_initial_worker is not None and time.monotonic()<deadline:
                    self.app.processEvents(); time.sleep(.01)
                self.app.processEvents()
                calibrate.assert_not_called()
                self.assertFalse(window.pll_enable_btn.isChecked())
                self.assertIsNone(window.plotter_worker)
            finally: window.close()

    def test_successful_preflight_updates_frequency_before_error_calibration(self):
        import resonance_gui as gui
        from resonance_fpga_gui import FPGAMainWindow
        from test_resonance_fpga_gui import FakeFPGA
        c=FakeFPGA()
        with tempfile.TemporaryDirectory() as folder, patch.object(gui,"STATE_DIR",Path(folder)), \
             patch.object(FPGAMainWindow,"_connect_hardware",lambda window:setattr(window,"fpga",c)), \
             patch.object(FPGAMainWindow,"_start_fpga_delay_calibration",return_value=True) as calibrate:
            window=FPGAMainWindow()
            try:
                window.pll_enable_btn.blockSignals(True); window.pll_enable_btn.setChecked(True)
                window.pll_enable_btn.blockSignals(False)
                window._pll_validation_failed=True
                window._initial_outcome=(True,dict(frequency_hz=38450,q=100,low_hz=37000,high_hz=39500,
                    original_frequency_hz=38000,path="test"))
                window._initial_finished()
                self.assertEqual(window.pll_center_spin.value(),38450)
                self.assertEqual(window.pll_q_spin.value(),100)
                self.assertLessEqual(window.pll_range_spin.value(),500)
                calibrate.assert_called_once_with(start_tracking=True)
                self.assertIsNone(window.plotter_worker)
                window.start_plotter()
                self.assertIsNone(window.plotter_worker)
                window.drive_v_spin.setValue(window.drive_v_spin.value()+.001)
                window._start_fpga_closed_loop()
                self.assertFalse(window.pll_enable_btn.isChecked())
                self.assertFalse(c.running)
            finally: window.close()
