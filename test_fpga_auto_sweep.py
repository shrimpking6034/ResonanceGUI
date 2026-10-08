import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from beam_modes import BeamMode
from fpga_auto_sweep import (AutoSweepSettings, AutoModeSweepWorker, build_mode_plans,
                             fine_window, frequency_grid, load_auto_session, peak_candidates,
                             reconcile_coarse_candidates)
from fpga_registers import PLLUserParameters, frequency_to_word, word_to_frequency
from fpga_resonance_gui import SweepWorker
from resonance_fit import three_db_bounds


class SyntheticController:
    def __init__(self, centers=(1000, 2000), flat=False, fail_after=None):
        self.centers = centers
        self.flat = flat
        self.fail_after = fail_after
        self.calls = 0
        self.running = False
        self.stops = 0

    def apply_parameters(self, parameters):
        self.parameters = parameters

    def start_fixed(self, **kwargs):
        self.running = True

    def set_fixed_frequency(self, f):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("injected TCP failure")
        return word_to_frequency(frequency_to_word(f))

    def complex_response(self, f, count):
        z = complex(1.0) if self.flat else sum(1000/(1+1j*2*(f-center)/10) for center in self.centers)
        # Nonzero SEMs, exactly balanced noise.
        noise = np.tile([0.01+0.01j, -0.01-0.01j], (count+1)//2)[:count]
        return z + noise

    def stop_output(self):
        self.running = False
        self.stops += 1

    def stop_stream(self):
        pass


class AutoSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)/"session"
        self.modes = [BeamMode(i, 1, i*1000) for i in range(1, 4)]
        self.settings = AutoSweepSettings(last_mode=2, coarse_step_hz=2, coarse_samples=16, fine_samples=16)

    def worker(self, controller=None, settings=None):
        return AutoModeSweepWorker(controller or SyntheticController(), PLLUserParameters(1000, 900, 1100, 100, 30), self.modes,
                                   settings or self.settings, self.directory)

    def test_grid_preserves_one_hz_and_limits(self):
        grid = frequency_grid(100.3, 110.8, 1)
        np.testing.assert_allclose(np.diff(grid), 1)
        self.assertLessEqual(grid[-1], 110.8)
        for args in [(0, 10, 1), (1, 10, 0), (1, 6250000, 1), (1, 1000, .01, 10)]:
            with self.assertRaises(ValueError):
                frequency_grid(*args)

    def test_plans_and_neighbor_clipping(self):
        settings = replace(self.settings, coarse_percent=90)
        plans = build_mode_plans(self.modes, settings)
        self.assertEqual(plans[0]["high"], plans[1]["low"])
        best = dict(frequency_hz=1000, linewidth_hz=20)
        low, high, _, _, clipped = fine_window(best, [best, dict(frequency_hz=1080)], plans[0], settings, 10)
        self.assertEqual(low, 940)
        self.assertEqual(high, 1040)
        self.assertTrue(clipped)

    def test_three_db_interpolates_both_sides(self):
        low, high = three_db_bounds(np.arange(5), np.array([0, .5, 1, .5, 0]), 2)
        self.assertAlmostEqual(low, np.sqrt(2))
        self.assertAlmostEqual(high, 4-np.sqrt(2))

    def test_single_sweep_completion_occurs_after_output_cleanup(self):
        controller = SyntheticController()
        worker = SweepWorker(controller, PLLUserParameters(1000, 900, 1100, 100, 30),
                             np.arange(990, 1011, 2), .05, 16)
        running_at_completion = []
        worker.completed.connect(lambda _: running_at_completion.append(controller.running), Qt.DirectConnection)
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual(running_at_completion, [False])

    def test_fitting_can_be_cancelled(self):
        from resonance_fit import fit_complex_resonance
        f = np.arange(990, 1011)
        with self.assertRaises(InterruptedError):
            fit_complex_resonance(f, np.ones(len(f)), np.zeros(len(f)), cancel_check=lambda: True)

    def test_narrow_peak_has_position_uncertainty_floor(self):
        plan = build_mode_plans(self.modes, self.settings)[0]
        best = dict(frequency_hz=1000, linewidth_hz=2)
        low, high, _, _, _ = fine_window(best, [best], plan, self.settings, 10)
        self.assertEqual((low, high), (960, 1040))

    def test_saved_settings_include_theory_and_drive(self):
        import resonance_gui as base
        from resonance_fpga_gui import FPGAMainWindow
        state = Path(self.temp.name)/"state"
        with patch.object(base, "STATE_DIR", state), patch.object(FPGAMainWindow, "_connect_hardware", lambda w: None):
            window = FPGAMainWindow()
            window.theory_length_spin.setValue(60)
            window.theory_fluid_density_spin.setValue(1.2)
            window.drive_v_spin.setValue(.3)
            window.auto_fields["last_mode"].setValue(8)
            window._save_last_settings()
            window.close()
            restored = FPGAMainWindow()
            try:
                self.assertEqual(restored.theory_length_spin.value(), 60)
                self.assertEqual(restored.theory_fluid_density_spin.value(), 1.2)
                self.assertEqual(restored.drive_v_spin.value(), .3)
                self.assertEqual(restored.auto_fields["last_mode"].value(), 8)
            finally:
                restored.close()

    def test_two_modes_fine_files_fit_and_reload(self):
        worker = self.worker()
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual(worker.outcome["status"], "complete", worker.outcome)
        self.assertEqual(len(worker.records), 2)
        for record in worker.records:
            self.assertEqual(record["status"], "complete", record)
            self.assertLess(abs(record["f0_hz"]-record["theory_hz"]), 1)
            self.assertTrue((self.directory/f"mode_{record['mode']:02d}_coarse.csv").exists())
            with np.load(self.directory/f"mode_{record['mode']:02d}_fine.npz") as data:
                np.testing.assert_allclose(np.diff(data["requested_freqs"]), 1)
                np.testing.assert_allclose(np.diff(data["freqs"]), 1, atol=.003)
        session, stages = load_auto_session(self.directory)
        self.assertEqual(len(stages), 4)
        self.assertEqual(session["status"], "complete")
        self.assertFalse(worker.controller.running)

    def test_missing_peak_retries_then_continues(self):
        worker = self.worker(SyntheticController(flat=True))
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual([r["status"] for r in worker.records], ["not_found", "not_found"])
        self.assertTrue(all(len(r["stages"]) == 2 for r in worker.records))

    def test_mid_segment_stop_preserves_partial_and_pending(self):
        worker = self.worker()
        worker.point.connect(lambda point: worker.stop() if point["count"] == 3 else None, Qt.DirectConnection)
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual(worker.outcome["status"], "stopped")
        self.assertEqual([r["status"] for r in worker.records], ["stopped", "pending"])
        _, stages = load_auto_session(self.directory)
        self.assertEqual(len(stages[0]["data"]["freqs"]), 3)
        self.assertFalse(worker.controller.running)

    def test_error_preserves_partial_and_stops(self):
        worker = self.worker(SyntheticController(fail_after=4))
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual(worker.outcome["status"], "failed")
        self.assertIn("TCP failure", worker.outcome["error"])
        _, stages = load_auto_session(self.directory)
        self.assertEqual(len(stages[0]["data"]["freqs"]), 4)
        self.assertFalse(worker.controller.running)

    def test_similar_candidates_are_ambiguous(self):
        worker = self.worker(SyntheticController(centers=(970, 1030)), replace(self.settings, last_mode=1))
        with patch.object(SweepWorker, "_pause", lambda *args: None):
            worker.run()
        self.assertEqual(worker.records[0]["status"], "ambiguous")
        self.assertFalse(any(s["stage"].startswith("fine") for s in worker.records[0]["stages"]))

    def test_dominant_peak_near_edge_beats_smaller_theory_neighbor(self):
        f = np.arange(9800, 12001, 10, dtype=float)
        small = 5*np.exp(-((f-10800)/180)**2)
        large = 20/(1+np.exp(-(f-11270)/50))*np.exp(-np.maximum(f-11550, 0)/2000)
        r = 1 + small + large
        candidates, _ = peak_candidates(dict(freqs=f, R=r), 10896, f[0], f[-1])
        self.assertGreater(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["frequency_hz"], 11550, delta=100)
        self.assertGreater(candidates[0]["peak_height"], 3*candidates[1]["peak_height"])

    def test_local_retry_cannot_reweight_other_broad_candidates(self):
        original = [dict(frequency_hz=40679., peak_height=108., prominence=50.,
                         linewidth_hz=None, bounds=None, edge=False),
                    dict(frequency_hz=38269., peak_height=24., prominence=5.,
                         linewidth_hz=None, bounds=None, edge=False)]
        retry = [dict(frequency_hz=40665., peak_height=108., prominence=1.,
                      linewidth_hz=None, bounds=None, edge=False)]
        result = reconcile_coarse_candidates(original, retry, 38877., 34989., 42759.,
                                             40579., 40779., 10., False)
        self.assertEqual(result[0]["frequency_hz"], 40665.)
        self.assertGreater(result[0]["score"], 2*result[1]["score"])

    def test_background_peak_gets_broad_fine_window(self):
        settings = replace(self.settings, coarse_step_hz=10)
        plan = build_mode_plans(self.modes, settings)[0]
        best = dict(frequency_hz=1000., linewidth_hz=None, prominence_width_hz=70.)
        low, high, _, _, _ = fine_window(best, [best], plan, settings, 10)
        self.assertEqual((low, high), (790., 1210.))

    def test_gui_runs_and_browses_saved_session(self):
        import resonance_gui as base
        from resonance_fpga_gui import FPGAMainWindow
        with patch.object(base, "STATE_DIR", Path(self.temp.name)/"state"), \
             patch.object(FPGAMainWindow, "_connect_hardware", lambda w: None):
            window = FPGAMainWindow()
            try:
                window.fpga = SyntheticController()
                with patch.object(window, "_auto_configuration", return_value=(self.settings, __import__('beam_modes').TubeBeam(.06,.0008,.0006,193e9,7930,1.2), self.modes, build_mode_plans(self.modes, self.settings))), \
                     patch.object(SweepWorker, "_pause", lambda *args: None):
                    window.start_auto_sweep(directory=self.directory)
                    self.assertTrue(window.auto_stop_btn.isEnabled())
                    self.assertFalse(window.run_sweep_btn.isEnabled())
                    self.assertTrue(window.auto_worker.wait(15000))
                    self.app.processEvents()
                self.assertEqual(window.auto_table.rowCount(), 2)
                self.assertEqual(window.auto_view.currentIndex(), 0)
                self.assertGreaterEqual(len(window.auto_amp_plot.listDataItems()), 4)
                window.auto_view.setCurrentIndex(1)
                self.assertEqual(window.auto_view.currentData(), 1)
                window.open_auto_session(directory=self.directory)
                self.assertEqual(len(window.auto_stages), 4)
                self.assertTrue(window.run_sweep_btn.isEnabled())
                self.assertTrue(window.n_points_spin.lineEdit().isEnabled())
                self.assertTrue(window.auto_fields["fine_step_hz"].lineEdit().isEnabled())
                self.assertFalse(window.auto_stop_btn.isEnabled())
                settings_file = self.directory/"settings.json"
                settings = json.loads(settings_file.read_text(encoding="utf-8"))
                settings.pop("peak_selection_version")
                settings_file.write_text(json.dumps(settings), encoding="utf-8")
                (self.directory/"selection_replay.json").write_text(json.dumps([
                    {"mode": 1, "revised_coarse_peak_hz": 1100.0}]), encoding="utf-8")
                original_session = (self.directory/"session.json").read_text(encoding="utf-8")
                window.open_auto_session(directory=self.directory)
                self.assertIn("이전 피크 판정", window.auto_status.text())
                self.assertIn("새 coarse 후보", window.auto_records[0]["notes"][-1])
                self.assertEqual((self.directory/"session.json").read_text(encoding="utf-8"), original_session)
            finally:
                window.fpga = None
                window.close()


if __name__ == "__main__":
    unittest.main()

