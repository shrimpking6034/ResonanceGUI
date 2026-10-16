"""HF2LI adapters for the shared automatic sweep and PLL response tools."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QLabel, QMessageBox, QPushButton

from fpga_auto_sweep import AutoModeSweepWorker
from fpga_auto_sweep_gui import AutoSweepMixin
from fpga_i18n import source_combo_text
from fpga_pll_response import analyze_step_response, save_step_response


@dataclass(frozen=True)
class HF2AutoParameters:
    frequency_hz: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    drive_amplitude_fraction: float  # shared field; actual unit recorded as Vpk


class HF2AutoController:
    """Open the controller on the owning worker thread only."""
    def __init__(self, drive_vpk):
        self.drive_vpk = drive_vpk
        self.hf = None

    def enable(self):
        from resonance_gui import HFController
        if self.hf is None:
            self.hf = HFController()
        self.hf.set_pll_enabled(False)
        self.hf.set_drive_amplitude(self.drive_vpk)
        self.hf.set_amp_enabled(True)
        self.hf.set_output_on(True)

    def stop_output(self):
        if self.hf:
            try:
                self.hf.set_output_on(False)
                self.hf.set_amp_enabled(False)
            finally:
                self.hf.close()
                self.hf = None


class HF2SweepSegment(QThread):
    point = pyqtSignal(float, float, float, float, float)

    def __init__(self, controller, grid, settings):
        super().__init__()
        from resonance_gui import SweepWorker
        self.controller = controller
        self.last_result = None
        self.error = None
        self.worker = SweepWorker(float(grid[0]), float(grid[-1]), len(grid), settings)
        self.worker.point_ready.connect(
            lambda i, f, r, phase, x, y: self.point.emit(f, r, phase, x, y), Qt.DirectConnection)
        self.worker.finished_ok.connect(self._result, Qt.DirectConnection)
        self.worker.failed.connect(self._failed, Qt.DirectConnection)

    def _result(self, data):
        self.last_result = data
        data["requested_freqs"] = data["freqs"].copy()

    def _failed(self, message):
        self.error = message

    def stop(self):
        self.worker.stop()

    def measure(self):
        if self.worker._stop:
            return
        self.controller.enable()
        self.worker.run()
        if self.error:
            raise RuntimeError(self.error)


class HF2AutoSweepWorker(AutoModeSweepWorker):
    amplitude_unit = "V"

    def __init__(self, *args, sweep_settings, **kwargs):
        super().__init__(*args, **kwargs)
        self.sweep_settings = sweep_settings

    def _make_segment(self, params, grid, settle, samples):
        settings = dict(self.sweep_settings, min_settling_s=settle, sample_count=samples)
        return HF2SweepSegment(self.controller, grid, settings)


class HF2StepWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, step_deg, rate_hz):
        super().__init__()
        self.step_deg, self.rate_hz = step_deg, rate_hz
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from resonance_gui import HFController, HF2_PLL, HF2_DEMOD, np_path
        hf = event = None
        original = None
        old_rate = old_enabled = None
        subscribed = False
        rows = []
        payload = error = None
        path = np_path(f"DEMODS/{HF2_DEMOD}/SAMPLE")
        setpoint_path = np_path(f"PLLS/{HF2_PLL}/SETPOINT")
        try:
            hf = HFController()
            if not hf.zi.get_int(np_path(f"PLLS/{HF2_PLL}/ENABLE")):
                raise RuntimeError("HF2LI PLL tracking을 먼저 활성화하세요.")
            original = hf.zi.get_double(setpoint_path)
            old_rate, old_enabled = hf.get_demod_rate(), hf.zi.get_int(np_path(f"DEMODS/{HF2_DEMOD}/ENABLE"))
            hf.set_demod_rate(self.rate_hz)
            hf.set_demod_enabled(True)
            rate, clock = hf.get_demod_rate(), hf.get_clockbase()
            event = hf.zi.allocate_event()
            hf.zi.subscribe(path)
            subscribed = True
            first = last = None
            marks = []
            command_intervals = []
            for segment, duration in (("baseline", .75), ("step", 1.5), ("recovery", .75)):
                if segment != "baseline":
                    stamp = hf.read_demod_timestamped()[0]
                    if first is None:
                        raise RuntimeError("HF2LI 주파수 스트림 수신 부족")
                    marks.append((stamp-first)/clock)
                    before = time.monotonic()
                    target = ((original+self.step_deg+180) % 360-180) if segment == "step" else original
                    hf.zi.set_double(setpoint_path, target)
                    command_intervals.append(time.monotonic()-before)
                deadline = time.monotonic()+duration
                while time.monotonic() < deadline:
                    if self._stop:
                        raise RuntimeError("폐루프 step 측정 중지")
                    _, samples = hf.zi.poll_event(event, 10)
                    for sample in samples:
                        stamp, _, _, frequency, *_ = sample
                        if last is not None and stamp <= last:
                            continue
                        if first is None:
                            first = stamp
                        last = stamp
                        rows.append(((stamp-first)/clock, float(frequency), segment))
                    if not samples:
                        self.msleep(1)
            payload = dict(time_s=np.array([r[0] for r in rows]), frequency_hz=np.array([r[1] for r in rows]),
                           segment=np.array([r[2] for r in rows]), step_time_s=marks[0], restore_time_s=marks[1],
                           delay_step_turns=self.step_deg/360, phase_step_deg=self.step_deg,
                           base_setpoint_deg=original, stream_rate_hz=rate,
                           command_roundtrip_s=command_intervals,
                           command_time_source="device sample read immediately before host setpoint write")
        except Exception as exc:
            error = str(exc)
        finally:
            if hf:
                if original is not None:
                    try:
                        hf.zi.set_double(setpoint_path, original)
                    except Exception as exc:
                        error = f"원래 위상 setpoint 복구 실패: {exc}"
                if subscribed:
                    try:
                        hf.zi.unsubscribe(path)
                    except Exception as exc:
                        error = error or str(exc)
                for cleanup in (
                    lambda: hf.zi.deallocate_event(event) if event is not None else None,
                    lambda: hf.set_demod_rate(old_rate) if old_rate is not None else None,
                    lambda: hf.set_demod_enabled(bool(old_enabled)) if old_enabled is not None else None,
                    hf.zi.close,
                ):
                    try:
                        cleanup()
                    except Exception as exc:
                        error = error or f"HF2LI 측정 종료 복구 실패: {exc}"
        if error:
            self.failed.emit(error)
        else:
            self.completed.emit(payload)


class HF2FeaturesMixin(AutoSweepMixin):
    def start_auto_sweep(self, *args, **kwargs):
        response = getattr(self, "hf2_response_worker", None)
        if response and response.isRunning():
            QMessageBox.warning(self, "자동 스윕", "폐루프 step 측정이 끝난 뒤 실행하세요.")
            return
        return super().start_auto_sweep(*args, **kwargs)
    def _auto_connected(self):
        if hasattr(self, "fpga"):
            return super()._auto_connected()
        return self.hf is not None

    def _auto_backend(self, plans, modes, settings, directory, beam):
        if hasattr(self, "fpga"):
            return super()._auto_backend(plans, modes, settings, directory, beam)
        from resonance_gui import MAX_DRIVE_V
        drive = self.drive_v_spin.value()
        if not 0 <= drive <= MAX_DRIVE_V:
            raise ValueError("HF2LI Drive 범위를 확인하세요.")
        parameters = HF2AutoParameters(plans[0]["low"], plans[0]["low"], plans[0]["high"], drive)
        return HF2AutoSweepWorker(HF2AutoController(drive), parameters, modes, settings, directory,
                                 {"beam": asdict(beam), "drive_unit": "Vpk"},
                                 sweep_settings=self._hf2_sweep_settings())

    def _build_hf2_step_controls(self, group):
        self.hf2_response_worker = None
        self.hf2_last_response = None
        self.hf2_step_restart_plotter = False
        layout = group.layout()
        row = layout.rowCount()
        layout.addWidget(QLabel("폐루프 시험 phase step (°):"), row, 0)
        self.hf2_step_spin = QDoubleSpinBox()
        self.hf2_step_spin.setRange(.01, .7)
        self.hf2_step_spin.setDecimals(3)
        self.hf2_step_spin.setValue(360/2048)
        layout.addWidget(self.hf2_step_spin, row, 1)
        self.hf2_step_btn = QPushButton("실제 폐루프 step 응답 · BW 측정")
        self.hf2_step_btn.clicked.connect(self.measure_hf2_step_response)
        layout.addWidget(self.hf2_step_btn, row+1, 0, 1, 2)
        self.hf2_step_label = QLabel("실측 폐루프 BW: 아직 측정하지 않았습니다.")
        self.hf2_step_label.setWordWrap(True)
        layout.addWidget(self.hf2_step_label, row+2, 0, 1, 2)

    def measure_hf2_step_response(self):
        if not self.hf or not self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "폐루프 응답", "HF2LI PLL tracking을 먼저 활성화하세요.")
            return
        if self.baseline_active or self.pingpong_active or self._auto_busy() or (self.sweep_worker and self.sweep_worker.isRunning()):
            QMessageBox.warning(self, "폐루프 응답", "현재 측정을 중지한 뒤 실행하세요.")
            return
        if self.hf2_response_worker and self.hf2_response_worker.isRunning():
            return
        self.hf2_step_restart_plotter = bool(self.plotter_worker and self.plotter_worker.isRunning())
        if self.hf2_step_restart_plotter:
            self.stop_plotter()
        worker = HF2StepWorker(self.hf2_step_spin.value(), self.plotter_rate_spin.value())
        self.hf2_response_worker = worker
        worker.completed.connect(self._hf2_step_complete)
        worker.failed.connect(self._hf2_step_failed)
        worker.finished.connect(self._hf2_step_finished)
        self._hf2_step_locked = []
        for widget in list(self._settings_widgets().values()) + [self.run_sweep_btn, self.auto_run_btn,
                self.plotter_start_btn, self.baseline_start_btn, self.pp_start_btn]:
            self._hf2_step_locked.append((widget, widget.isEnabled()))
            widget.setEnabled(False)
        self.hf2_step_btn.setEnabled(False)
        self.hf2_step_label.setText("위상 setpoint step으로 폐루프 응답 측정 중…")
        worker.start()

    def _hf2_step_complete(self, payload):
        try:
            result = analyze_step_response(payload["time_s"], payload["frequency_hz"],
                                           step_time_s=payload["step_time_s"], restore_time_s=payload["restore_time_s"],
                                           delay_step_turns=payload["delay_step_turns"])
            metadata = {k: v for k, v in payload.items() if k not in ("time_s", "frequency_hz", "segment")}
            metadata.update(Backend="HF2LI", Time_source="device timestamp", target_bandwidth_hz=self.pll_target_bw_spin.value())
            paths = save_step_response(Path(__file__).parent/"diagnostics"/"hf2li", payload["time_s"],
                                       payload["frequency_hz"], payload["segment"], result, metadata)
            self.hf2_last_response = result
            self.hf2_step_label.setText(f"BW={result.bandwidth_hz:.2f} Hz · τ={result.tau_s*1000:.2f} ms · "
                                       f"R²={result.r_squared:.3f} · {'통과' if result.accepted else '검토 필요'}")
            self.hf2_step_label.setToolTip("\n".join(map(str, paths))+"\n"+"\n".join(result.reasons))
        except Exception as exc:
            self._hf2_step_failed(str(exc))

    def _hf2_step_failed(self, message):
        self.hf2_step_label.setText(f"폐루프 응답 측정 실패: {message}")

    def _hf2_step_finished(self):
        for widget, enabled in getattr(self, "_hf2_step_locked", []):
            widget.setEnabled(enabled)
        self._hf2_step_locked = []
        self.hf2_step_btn.setEnabled(True)
        if self.hf2_step_restart_plotter and self.pll_enable_btn.isChecked():
            self.start_plotter()
        self.hf2_step_restart_plotter = False

    def _hf2_measurement_busy(self):
        response = getattr(self, "hf2_response_worker", None)
        return self._auto_busy() or bool(response and response.isRunning())

    def _hf2_sweep_settings(self):
        return dict(filter_mode=source_combo_text(self.sweep_filter_mode_combo),
                    order=self.sweep_filter_order_spin.value(),
                    fixed_bandwidth_hz=self.sweep_fixed_bw_spin.value(),
                    max_bandwidth_hz=self.sweep_max_bw_spin.value(),
                    omega_suppression_db=self.sweep_suppression_spin.value(),
                    bandwidth_overlap=self.sweep_overlap_check.isChecked(),
                    min_settling_s=self.sweep_min_settle_spin.value(),
                    settling_residual=self.sweep_residual_spin.value(),
                    sample_count=self.sweep_samples_spin.value(),
                    sample_rate_hz=self.sweep_rate_spin.value(),
                    averaging_time_s=self.sweep_avg_time_spin.value(),
                    averaging_tc=self.sweep_avg_tc_spin.value())
