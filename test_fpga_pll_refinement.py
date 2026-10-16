import tempfile
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import time
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PyQt5.QtWidgets import QApplication
from fpga_registers import PLLUserParameters, ModeFields
from fpga_pll_refinement import ClosedLoopDelayWorker, select_amplitude_peak
from fpga_stream_health import DeliveryMonitor


class FakeController:
    channel=0
    receiver=None
    dropped_local_batches=0
    def __init__(self):
        self.last_parameters=PLLUserParameters(39000,38500,39500,50,55,
            cic_rate=32767,cic_bit_shift=21,pll_delay_turns=.25,
            mode=ModeFields(run=True,enable_agc=True,send_data_to_pc=True,run_nco_at_fixed_frequency=False))
        self.delay=.25
        self.mode=self.last_parameters.mode.encode()
        self.stopped=False
        self.signal="frequency"
    def stop_stream(self): pass
    def start_stream(self,signal="frequency"): self.signal=signal
    def drain_stream(self,*args): return []
    def write_named(self,name,value):
        if name=="delay": self.delay=value/(2**32-1)
        if name=="mode": self.mode=value
    def stop_output(self): self.stopped=True


class FastWorker(ClosedLoopDelayWorker):
    flat=False
    fail_after=None
    count=0
    def _wait(self,seconds): self._check()
    def _samples(self,count=512):
        self.count+=1
        if self.fail_after is not None and self.count>=self.fail_after:
            self.cancelled=True
            self._check()
        offset=self.controller.delay-.25
        if self.controller.signal=="frequency": return 39000+offset*3000, .01
        return (1e6 if self.flat else 1e6*(1-800*(offset-.002)**2)),1.


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_peak_selected_and_tracking_agc_restored_without_restart(self):
        c=FakeController(); original=c.last_parameters
        with tempfile.TemporaryDirectory() as folder:
            worker=FastWorker(c,directory=folder)
            results=[]; failures=[]
            worker.completed.connect(results.append); worker.failed.connect(failures.append)
            worker.run()
            self.assertEqual(failures,[])
            self.assertAlmostEqual(c.delay,.252,delta=.0006)
            self.assertEqual(c.mode,original.mode.encode())
            self.assertTrue(c.last_parameters.mode.enable_agc)
            self.assertFalse(c.stopped)
            self.assertEqual(c.signal,"frequency")
            self.assertTrue(Path(results[0]["path"]).exists())

    def test_flat_and_cancel_restore_original_delay_and_agc(self):
        for flat,fail_after in ((True,None),(False,7)):
            c=FakeController()
            with tempfile.TemporaryDirectory() as folder:
                worker=FastWorker(c,directory=folder)
                worker.flat=flat; worker.fail_after=fail_after
                failures=[]; worker.failed.connect(failures.append)
                worker.run()
                self.assertTrue(failures)
                self.assertAlmostEqual(c.delay,.25)
                self.assertTrue(c.last_parameters.mode.enable_agc)
                self.assertEqual(c.mode,c.last_parameters.mode.encode())
                if fail_after is not None:
                    self.assertTrue(c.stopped)
                    self.assertFalse(c.last_parameters.mode.run)

    def test_edge_peak_is_rejected(self):
        rows=[dict(offset_turns=x,amplitude_counts=1000+i*100,amplitude_uncertainty_counts=1)
              for i,x in enumerate(np.linspace(-.01,.01,5))]
        with self.assertRaises(ValueError): select_amplitude_peak(rows,.25,.01)

    def test_quantized_interior_plateau_keeps_original_delay(self):
        # Regression: measured 2026-10-15 scan; 318/319-count magnitude steps.
        amplitudes=[404496,404814,405450,405450,405769,405769,406087,406087,406087,
                    406087,406087,406087,406087,406087,406087,405769,405450]
        rows=[dict(offset_turns=float(x),amplitude_counts=a,amplitude_uncertainty_counts=159.5 if i==16 else 0)
              for i,(x,a) in enumerate(zip(np.linspace(-.007813,.007813,17),amplitudes))]
        result=select_amplitude_peak(rows,.861731,.007813)
        self.assertAlmostEqual(result["delay_turns"],.861731)
        self.assertGreater(result["contrast_counts"],result["tolerance_counts"])
        self.assertLess(result["plateau_low_turns"],0)
        self.assertGreater(result["plateau_high_turns"],0)

    def test_boundary_plateau_expands_once_then_finds_peak(self):
        class Shifted(FastWorker):
            def _samples(self,count=512):
                offset=self.controller.delay-.25
                if self.controller.signal=="frequency":return 39000+offset*3000,.01
                return round(1e6*(1-800*(offset-.012)**2)/319)*319,1.
        c=FakeController()
        with tempfile.TemporaryDirectory() as folder:
            worker=Shifted(c,directory=folder)
            results=[];errors=[]
            worker.completed.connect(results.append);worker.failed.connect(errors.append)
            worker.run()
            self.assertEqual(errors,[])
            self.assertGreater(results[0]["search_half_width_turns"],1/128)
            self.assertAlmostEqual(results[0]["delay_turns"],.262,delta=.002)
            self.assertTrue(c.last_parameters.mode.enable_agc)

    def test_delivery_monitor_counts_verified_gaps(self):
        monitor=DeliveryMonitor()
        for i in range(10):
            monitor.observe(SimpleNamespace(host_monotonic_s=i*.1,values=np.zeros(128),missing_packets=1 if i==5 else 0))
        stats=monitor.summary(1280,2)
        self.assertAlmostEqual(stats["host_delivery_rate_hz"],1280)
        self.assertEqual(stats["missing_packets_verified"],1)
        self.assertTrue(stats["warnings"])

    def test_gui_starts_refinement_before_plotter_and_updates_delay(self):
        import resonance_gui as gui
        from resonance_fpga_gui import FPGAMainWindow
        from test_resonance_fpga_gui import FakeFPGA
        class Worker(FastWorker):
            def _samples(self,count=512):
                p=self.controller.last_parameters
                offset=(self.controller.delay_turns-p.pll_delay_turns+.5)%1-.5
                if self.controller.signal=="frequency": return p.frequency_hz+offset*3000,.01
                return 1e6*(1-800*(offset-.002)**2),1.
        c=FakeFPGA()
        with tempfile.TemporaryDirectory() as folder, patch.object(gui,"STATE_DIR",Path(folder)), \
             patch.object(FPGAMainWindow,"_connect_hardware",lambda window:setattr(window,"fpga",c)), \
             patch.object(FPGAMainWindow,"start_pll_preflight",lambda window:window._start_fpga_closed_loop() or True), \
             patch("fpga_refinement_gui.ClosedLoopDelayWorker",
                   lambda controller,**kwargs:Worker(controller,directory=Path(folder)/"diagnostics",**kwargs)):
            window=FPGAMainWindow()
            try:
                window.fpga_auto_delay_check.setChecked(False)
                window.fpga_peak_auto_check.setChecked(True)
                window.pll_center_spin.setValue(39000)
                window.pll_q_spin.setValue(50)
                window.fpga_pll_delay_spin.setValue(.25)
                window.pll_enable_btn.setChecked(True)
                self.assertIsNotNone(window.fpga_peak_worker)
                self.assertIsNone(window.plotter_worker)
                deadline=time.monotonic()+3
                while time.monotonic()<deadline and window.fpga_peak_worker is not None:
                    self.app.processEvents(); time.sleep(.01)
                self.app.processEvents()
                self.assertIsNotNone(window._last_peak_refinement)
                self.assertAlmostEqual(window.fpga_pll_delay_spin.value(),.252,delta=.0006)
                self.assertIsNotNone(window.plotter_worker)
                self.assertTrue(window.plotter_worker.isRunning())
            finally:
                window.close()
