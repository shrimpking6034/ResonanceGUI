"""The original ResonanceGUI interface backed by the channel-A SMR FPGA.

The FPGA has one TCP client and one selected UDP signal. All device writes
share one FPGAController; HF2LI-only controls are disabled in this view.
"""
from __future__ import annotations

import sys
import time
import csv
from contextlib import contextmanager
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox,
                             QLabel, QMessageBox, QPushButton,
                             QSpinBox)

from fpga_backend import FPGAController
from fpga_delay_calibration import positive_error_zero, refine_positive_error_zero
from fpga_pll_response import analyze_step_response, save_step_response
from fpga_registers import (ModeFields, PLLUserParameters, pll_update_rate_hz,
                            user_parameters_to_registers)
from fpga_resonance_gui import SweepWorker as FPGASweepWorker
from fpga_transport import NetworkProfile
import resonance_gui as hf_module
from resonance_gui import MainWindow
from fpga_auto_sweep_gui import AutoSweepMixin
from fpga_refinement_gui import RefinementMixin
from fpga_stream_health import DeliveryMonitor
from fpga_i18n import install_language_selector,FileDialog as QFileDialog,source_widget_text,source_combo_text


class FPGAPlotterWorker(QThread):
    """Adapt FPGA frequency batches to the original Plotter tuple format."""

    batch_ready = pyqtSignal(object)
    stream_started = pyqtSignal(float, float)
    failed = pyqtSignal(str)
    health_ready = pyqtSignal(object)

    def __init__(self, controller: FPGAController, rate_hz: float = 3051.0):
        super().__init__()
        self.controller = controller
        self.rate_hz = float(rate_hz)
        self._stop = False
        self._sample_index = 0

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.stream_started.emit(self.rate_hz, self.rate_hz)
            ignore_before = time.monotonic() + 0.5
            last_data = last_report = time.monotonic()
            health = DeliveryMonitor()
            while not self._stop:
                if self.controller.dropped_local_batches:
                    raise RuntimeError("FPGA 로컬 수신 큐 누락: 시간축 압축을 방지하기 위해 Plotter 기록을 중단합니다.")
                receiver = self.controller.receiver
                if receiver is not None and receiver.error is not None:
                    raise RuntimeError(f"FPGA UDP receiver: {receiver.error}")
                output = []
                for batch in self.controller.drain_stream(100):
                    if batch.host_monotonic_s < ignore_before:
                        continue
                    if batch.channel != self.controller.channel:
                        continue
                    last_data = time.monotonic()
                    health.observe(batch)
                    if batch.missing_packets:
                        self._sample_index += batch.missing_packets * len(batch.values)
                    for frequency in batch.values:
                        t = self._sample_index / self.rate_hz
                        output.append((t, np.nan, np.nan, float(frequency),
                                       np.nan, np.nan, np.nan))
                        self._sample_index += 1
                now = time.monotonic()
                if now-last_data > 3:
                    raise TimeoutError("FPGA 주파수 스트림이 3초 이상 중단되었습니다.")
                if now-last_report >= 1:
                    self.health_ready.emit(health.summary(self.rate_hz,self.controller.dropped_local_batches))
                    last_report=now
                if output:
                    self.batch_ready.emit(output)
                else:
                    self.msleep(10)
        except Exception as exc:
            self.failed.emit(str(exc))


class FPGAErrorScanWorker(QThread):
    """Sample the VI's selected raw error stream with a fixed NCO."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, controller: FPGAController, parameters: PLLUserParameters,
                 frequencies: np.ndarray):
        super().__init__()
        self.controller = controller
        self.parameters = parameters
        self.frequencies = np.asarray(frequencies, dtype=float)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            mode = replace(self.parameters.mode, signal_of_interest=1,
                           run_nco_at_fixed_frequency=True)
            params = replace(self.parameters,
                             frequency_hz=float(self.frequencies[0]), mode=mode)
            self.controller.start_stream(signal="signed")
            self.controller.apply_parameters(params)
            self.controller.start_fixed(stream=True)
            measurements = []
            for frequency in self.frequencies:
                if self._stop:
                    raise RuntimeError("FPGA error 진단 중지")
                self.controller.set_fixed_frequency(float(frequency))
                self.msleep(150)
                self.controller.drain_stream(200)  # discard transient packets
                samples = []
                deadline = time.monotonic() + 3.0
                while len(samples) < 256 and time.monotonic() < deadline:
                    if self._stop:
                        raise RuntimeError("FPGA error 진단 중지")
                    receiver = self.controller.receiver
                    if receiver is not None and receiver.error is not None:
                        raise RuntimeError(f"FPGA UDP receiver: {receiver.error}")
                    for batch in self.controller.drain_stream(30):
                        samples.extend(batch.values.tolist())
                    if len(samples) < 256:
                        self.msleep(10)
                if len(samples) < 256:
                    raise TimeoutError("FPGA raw error UDP 수신 부족")
                measurements.append(float(np.median(samples[:256])))
            self.completed.emit((self.frequencies.copy(), np.asarray(measurements)))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                self.controller.stop_output()
            except Exception as exc:
                self.failed.emit(f"FPGA 진단 정지 명령 실패: {exc}")
            self.controller.stop_stream()


class FPGADelayCalibrationWorker(QThread):
    """Measure raw error versus delay at the requested drive and fixed f0."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, controller: FPGAController, parameters: PLLUserParameters):
        super().__init__()
        self.controller = controller
        self.parameters = parameters
        self._stop = False

    def stop(self):
        self._stop = True

    def _median_error(self):
        values = []
        deadline = time.monotonic() + 1.5
        while len(values) < 256 and time.monotonic() < deadline:
            if self._stop:
                raise RuntimeError("PLL delay 보정 중지")
            receiver = self.controller.receiver
            if receiver is not None and receiver.error is not None:
                raise RuntimeError(f"FPGA UDP receiver: {receiver.error}")
            for batch in self.controller.drain_stream(30):
                if batch.channel == self.controller.channel:
                    values.extend(batch.values.tolist())
            if len(values) < 256:
                self.msleep(10)
        if len(values) < 256:
            raise TimeoutError("FPGA 원시 error 수신 부족")
        samples = np.asarray(values[:256], dtype=float)
        return float(np.median(samples)), float(np.median(np.abs(samples - np.median(samples))))

    def _measure_delay(self, delay):
        self.controller.write_named("delay", round(float(delay) * (2**32 - 1)))
        self.msleep(180)
        self.controller.drain_stream(200)  # prior-delay transient
        return self._median_error()

    def run(self):
        result, failure = None, None
        try:
            mode = replace(self.parameters.mode, signal_of_interest=1,
                           run_nco_at_fixed_frequency=True)
            parameters = replace(self.parameters, mode=mode,
                                 pll_delay_turns=0.0)
            self.controller.start_stream(signal="signed")
            self.controller.apply_parameters(parameters)
            self.controller.start_fixed(stream=True)
            delays = np.arange(16, dtype=float) / 16.0
            errors, noises = [], []
            for delay in delays:
                if self._stop:
                    raise RuntimeError("PLL delay 보정 중지")
                error, noise = self._measure_delay(delay)
                errors.append(error)
                noises.append(noise)
            coarse_delay, span = positive_error_zero(
                delays, errors, noise_counts=float(np.median(noises)))
            coarse_step = 1.0 / len(delays)
            fine_start = np.floor(coarse_delay / coarse_step) * coarse_step
            fine_subdivisions = 16
            fine_delays = fine_start + np.linspace(
                0.0, coarse_step, fine_subdivisions + 1)
            fine_errors, fine_noises = [], []
            for fine_delay in fine_delays:
                if self._stop:
                    raise RuntimeError("PLL delay 보정 중지")
                error, noise = self._measure_delay(fine_delay % 1.0)
                fine_errors.append(error)
                fine_noises.append(noise)
            delay, fine_span = refine_positive_error_zero(
                coarse_delay, fine_delays, fine_errors,
                noise_counts=float(np.median(fine_noises)))
            residual, noise = self._measure_delay(delay)
            if abs(residual) > max(0.05 * span, 8.0 * noise):
                raise RuntimeError(
                    f"PLL delay 영점 검증 실패: 잔차 {residual:,.0f} counts")
            result = {"delay_turns": delay, "error_counts": residual,
                      "error_span_counts": span,
                      "coarse_delay_turns": coarse_delay,
                      "fine_step_turns": coarse_step / fine_subdivisions,
                      "fine_error_span_counts": fine_span,
                      "drive_fraction": self.parameters.drive_amplitude_fraction,
                      "center_hz": self.parameters.frequency_hz}
        except Exception as exc:
            failure = str(exc)
        finally:
            try:
                self.controller.stop_output()
            except Exception as exc:
                failure = f"FPGA 보정 정지 명령 실패: {exc}"
            self.controller.stop_stream()
        if failure:
            self.failed.emit(failure)
        else:
            self.completed.emit(result)


class FPGAClosedLoopStepWorker(QThread):
    """Apply a small reversible delay step and record the PLL frequency."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, controller: FPGAController, base_delay_turns: float,
                 step_turns: float, sample_rate_hz: float,
                 baseline_s: float = 0.75, step_s: float = 1.50,
                 recovery_s: float = 0.75):
        super().__init__()
        self.controller = controller
        self.base_delay_turns = float(base_delay_turns) % 1.0
        self.step_turns = float(step_turns)
        self.sample_rate_hz = float(sample_rate_hz)
        self.baseline_s = float(baseline_s)
        self.step_s = float(step_s)
        self.recovery_s = float(recovery_s)
        self._stop = False

    def stop(self):
        self._stop = True

    def _write_delay(self, turns):
        turns = float(turns) % 1.0
        self.controller.write_named("delay", round(turns * (2**32 - 1)))

    def _collect(self, duration_s):
        values = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if self._stop:
                raise RuntimeError("폐루프 step 응답 측정 중지")
            receiver = self.controller.receiver
            if receiver is not None and receiver.error is not None:
                raise RuntimeError(f"FPGA UDP receiver: {receiver.error}")
            for batch in self.controller.drain_stream(100):
                if batch.channel == self.controller.channel:
                    self.delivery.observe(batch)
                    values.extend(np.asarray(batch.values, float).tolist())
            self.msleep(5)
        if not values:
            raise TimeoutError("FPGA 주파수 응답 데이터가 수신되지 않았습니다.")
        return np.asarray(values, dtype=float)

    def run(self):
        payload, failure = None, None
        stepped_delay = (self.base_delay_turns + self.step_turns) % 1.0
        try:
            parameters = self.controller.last_parameters
            self.delivery = DeliveryMonitor()
            if (parameters is None or not parameters.mode.run or
                    parameters.mode.run_nco_at_fixed_frequency):
                raise RuntimeError("FPGA PLL tracking이 실행 중이어야 합니다.")
            self.controller.stop_stream()
            self.controller.start_stream(signal="frequency")
            self.controller.drain_stream(500)
            self._write_delay(self.base_delay_turns)
            baseline = self._collect(self.baseline_s)
            self._write_delay(stepped_delay)
            stepped = self._collect(self.step_s)
            self._write_delay(self.base_delay_turns)
            recovery = self._collect(self.recovery_s)
            frequency = np.concatenate((baseline, stepped, recovery))
            time_s = np.arange(len(frequency), dtype=float) / self.sample_rate_hz
            step_time = len(baseline) / self.sample_rate_hz
            restore_time = (len(baseline) + len(stepped)) / self.sample_rate_hz
            segment = np.concatenate((
                np.full(len(baseline), "baseline", dtype="U8"),
                np.full(len(stepped), "step", dtype="U8"),
                np.full(len(recovery), "recovery", dtype="U8"),
            ))
            payload = {
                "time_s": time_s, "frequency_hz": frequency,
                "segment": segment, "step_time_s": step_time,
                "restore_time_s": restore_time,
                "delay_step_turns": self.step_turns,
                "base_delay_turns": self.base_delay_turns,
                "stream_health": self.delivery.summary(self.sample_rate_hz,self.controller.dropped_local_batches),
                "sample_rate_hz": self.sample_rate_hz,
            }
        except Exception as exc:
            failure = str(exc)
        finally:
            try:
                self._write_delay(self.base_delay_turns)
            except Exception as exc:
                failure = f"원래 PLL delay 복구 실패: {exc}"
            self.controller.stop_stream()
        if failure:
            self.failed.emit(failure)
        else:
            self.completed.emit(payload)


class FPGAMainWindow(RefinementMixin, MainWindow):
    """Reuse the HF2LI layout, fit, baseline, particle and DAQ workflows."""

    def __init__(self):
        self.fpga: FPGAController | None = None
        self._fpga_sweep_index = 0
        self._fpga_fixed_plotter = False
        self.fpga_error_worker = None
        self.fpga_delay_worker = None
        self.fpga_response_worker = None
        self._start_pll_after_delay = False
        self._last_delay_calibration = None
        self._last_step_response = None
        self._last_step_response_context = None
        self._restart_plotter_after_response = False
        super().__init__()
        self._build_auto_results()
        self.sweep_samples_spin.setValue(max(256, self.sweep_samples_spin.value()))
        self.sweep_min_settle_spin.setValue(max(0.05, self.sweep_min_settle_spin.value()))
        self.setWindowTitle("Steel-Tube Resonator Control (FPGA + NI-DAQ)")
        self.tabs.setTabText(2, "③ Plotter (FPGA Frequency)")
        self.plotter_plot.getAxis("right").setLabel("Amplitude unavailable in frequency stream")
        self.plotter_r_curve.setData([], [])
        self.hist_channel_combo.clear()
        self.hist_channel_combo.addItem("Frequency (Hz)")
        self.pll_target_bw_spin.valueChanged.connect(self._update_fpga_pll_status)
        self.fpga_agc_check.toggled.connect(self._update_fpga_pll_status)
        self._add_reconnect_button()
        install_language_selector(self,hf_module.STATE_DIR/"fpga"/"language.json")

    def _update_fpga_pll_status(self, *_args):
        if not self.pll_enable_btn.isChecked() or self.fpga is None:
            return
        parameters = self.fpga.last_parameters
        if parameters is None or not parameters.mode.run:
            return
        agc = "켬" if parameters.mode.enable_agc else "끔"
        pending = (abs(self.pll_target_bw_spin.value() - parameters.loop_bandwidth_hz) > 1e-6
                   or self.fpga_agc_check.isChecked() != parameters.mode.enable_agc)
        suffix = " · 변경 적용은 PLL 재시작 필요" if pending else ""
        self.pll_locked_label.setText(
            f"FPGA PLL 실행 · 적용 BW {parameters.loop_bandwidth_hz:g} Hz / AGC {agc}"
            f" · lock bit 미제공{suffix}")

    @contextmanager
    def _fpga_state_dir(self):
        original = hf_module.STATE_DIR
        hf_module.STATE_DIR = original / "fpga"
        try:
            yield
        finally:
            hf_module.STATE_DIR = original

    def _settings_widgets(self):
        widgets = super()._settings_widgets()
        widgets.update(self._auto_settings_widgets())
        widgets.update({name: getattr(self, name) for name in (
            "fpga_pll_delay_spin", "fpga_cic_rate_spin",
            "fpga_cic_shift_spin", "fpga_agc_check",
            "fpga_auto_delay_check", "fpga_step_size_spin",
            "fpga_peak_auto_check", "fpga_peak_width_spin")})
        return widgets

    def _save_last_settings(self):
        with self._fpga_state_dir():
            return super()._save_last_settings()

    def _restore_last_settings(self):
        with self._fpga_state_dir():
            return super()._restore_last_settings()

    def _save_last_sweep(self):
        with self._fpga_state_dir():
            return super()._save_last_sweep()

    def _restore_last_sweep(self):
        with self._fpga_state_dir():
            return super()._restore_last_sweep()

    def _add_reconnect_button(self):
        left_layout = self.status_label.parentWidget().layout()
        button = QPushButton("FPGA 재연결")
        button.clicked.connect(self._connect_hardware)
        left_layout.insertWidget(1, button)

    def _build_drive_group(self):
        group = super()._build_drive_group()
        group.setTitle("1. 구동 피에조 (FPGA NCO)")
        layout = group.layout()
        layout.itemAtPosition(1, 0).widget().setText("Drive (0..1 full scale):")
        self.range_combo.setEnabled(False)
        self.drive_v_spin.setRange(0, 1)
        self.drive_v_spin.setDecimals(4)
        self.drive_v_spin.setSingleStep(0.001)
        self.drive_v_spin.setSuffix(" FS")
        self.drive_v_spin.setValue(0.005)
        return group

    def _build_signal_input_group(self):
        group = super()._build_signal_input_group()
        group.setTitle("FPGA ADC 입력 (현재 펌웨어에서 설정 불가)")
        group.setEnabled(False)
        return group

    def _build_sweep_group(self):
        group = super()._build_sweep_group()
        self.f_start_spin.setMaximum(6_249_999)
        self.f_stop_spin.setMaximum(6_249_999)
        self.sweep_samples_spin.setValue(256)
        self.sweep_min_settle_spin.setValue(0.05)
        for widget in (self.sweep_filter_mode_combo, self.sweep_filter_order_spin,
                       self.sweep_fixed_bw_spin, self.sweep_max_bw_spin,
                       self.sweep_suppression_spin, self.sweep_overlap_check,
                       self.sweep_residual_spin, self.sweep_avg_time_spin,
                       self.sweep_avg_tc_spin, self.sweep_rate_spin):
            widget.setEnabled(False)
            widget.setToolTip("FPGA sweep는 고정된 CIC/lock-in 설정을 사용합니다.")
        self._build_auto_controls(group)
        return group

    def _build_pll_group(self):
        group = super()._build_pll_group()
        group.setTitle("4. FPGA PLL — Type-2 PI · 목표 폐루프 응답 1차")
        self.pll_center_spin.setMaximum(6_249_999)
        self.pll_order_spin.setRange(1, 1)
        for widget in (self.pll_input_combo, self.pll_setpoint_spin,
                       self.pll_tc_spin, self.pll_p_spin, self.pll_i_spin,
                       self.pll_d_spin, self.pll_derivative_check,
                       self.pll_integral_check, self.pll_invert_sign_check,
                       self.pll_pm_spin):
            widget.setEnabled(False)
            widget.setToolTip("이 HF2LI 노드에 대응하는 FPGA 제어 항목은 확인되지 않았습니다.")
        for button in group.findChildren(QPushButton):
            if button.objectName() in ("pll_pid_advisor","pll_phase_calibration","pll_sign_test"):
                button.setEnabled(False)
            if button.objectName()=="pll_phase_calibration":
                button.setText("FPGA 원시 error 진단 (f0 주변)")
                button.setEnabled(True)
        layout = group.layout()
        row = layout.rowCount()
        layout.addWidget(QLabel("FPGA PLL delay (turn):"), row, 0)
        self.fpga_pll_delay_spin = QDoubleSpinBox()
        self.fpga_pll_delay_spin.setRange(0, 0.999999)
        self.fpga_pll_delay_spin.setDecimals(6)
        self.fpga_pll_delay_spin.setSingleStep(0.001)
        self.fpga_pll_delay_spin.setValue(0.079)
        self.fpga_pll_delay_spin.setToolTip(
            "VI의 PLL_delay 레지스터 설정입니다. 단위는 0..1 주기 비율입니다.")
        layout.addWidget(self.fpga_pll_delay_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("FPGA CIC rate:"), row, 0)
        self.fpga_cic_rate_spin = QSpinBox()
        self.fpga_cic_rate_spin.setRange(16, 32767)
        self.fpga_cic_rate_spin.setValue(32767)
        self.fpga_cic_rate_spin.setToolTip(
            "낮출수록 PLL 갱신/UDP 샘플 속도가 증가합니다. "
            "32767≈3.05 kSa/s, 16384≈6.10 kSa/s, 8192≈12.2 kSa/s. "
            "변경 후 PLL을 다시 시작하고 안정성·노이즈를 확인하세요.")
        layout.addWidget(self.fpga_cic_rate_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("예상 PLL/stream rate:"), row, 0)
        self.fpga_rate_estimate_label = QLabel()
        layout.addWidget(self.fpga_rate_estimate_label, row, 1)
        self.fpga_cic_rate_spin.valueChanged.connect(self._update_fpga_rate_estimate)
        self._update_fpga_rate_estimate()
        row += 1
        layout.addWidget(QLabel("FPGA CIC bit shift:"), row, 0)
        self.fpga_cic_shift_spin = QSpinBox()
        self.fpga_cic_shift_spin.setRange(0, 31)
        self.fpga_cic_shift_spin.setValue(21)
        layout.addWidget(self.fpga_cic_shift_spin, row, 1)
        row += 1
        self.fpga_agc_check = QCheckBox("FPGA AGC 사용")
        self.fpga_agc_check.setChecked(False)
        self.fpga_agc_check.setToolTip("VI의 Enable AGC mode bit. 출력 및 검출 신호 변화를 확인하며 사용하세요.")
        layout.addWidget(self.fpga_agc_check, row, 0, 1, 2)
        row += 1
        self.fpga_auto_delay_check = QCheckBox("PLL 시작 전 delay 자동 보정")
        self.fpga_auto_delay_check.setChecked(True)
        self.fpga_auto_delay_check.setToolTip(
            "현재 중심 주파수와 Drive에서 원시 error를 스캔해 안정적인 영점을 선택합니다.")
        layout.addWidget(self.fpga_auto_delay_check, row, 0, 1, 2)
        row += 1
        self.fpga_delay_calibrate_btn = QPushButton("FPGA PLL delay 지금 보정")
        self.fpga_delay_calibrate_btn.clicked.connect(self.calibrate_fpga_pll_delay)
        layout.addWidget(self.fpga_delay_calibrate_btn, row, 0, 1, 2)
        row += 1
        layout.addWidget(QLabel("폐루프 시험 delay step (turn):"), row, 0)
        self.fpga_step_size_spin = QDoubleSpinBox()
        self.fpga_step_size_spin.setRange(1 / 65536, 1 / 512)
        self.fpga_step_size_spin.setDecimals(7)
        self.fpga_step_size_spin.setSingleStep(1 / 4096)
        self.fpga_step_size_spin.setValue(1 / 2048)
        self.fpga_step_size_spin.setToolTip(
            "PLL tracking 중 delay에 가하는 작은 양의 step입니다. 최대 1/512 turn으로 제한됩니다.")
        layout.addWidget(self.fpga_step_size_spin, row, 1)
        row += 1
        self.fpga_step_response_btn = QPushButton("실제 폐루프 step 응답 · BW 측정")
        self.fpga_step_response_btn.clicked.connect(self.measure_fpga_step_response)
        layout.addWidget(self.fpga_step_response_btn, row, 0, 1, 2)
        row += 1
        self.fpga_step_response_label = QLabel("실측 폐루프 BW: 아직 측정하지 않았습니다.")
        self.fpga_step_response_label.setWordWrap(True)
        self.fpga_step_response_label.setStyleSheet("color: #555;")
        layout.addWidget(self.fpga_step_response_label, row, 0, 1, 2)
        row += 1
        note = QLabel("선택 피크의 복소 fit f0·Q와 위 설정으로 FPGA Type-2 PI PLL을 구성하고, "
                      "공진기를 포함한 목표 응답을 1차 저역통과로 설정합니다. "
                      "원시 error stream은 선택 수신할 수 있지만 HF2LI의 ° 단위 ERROR와 같지 않습니다. "
                      "lock bit는 확인되지 않았습니다.")
        note.setWordWrap(True)
        layout.addWidget(note, row, 0, 1, 2)
        self._build_refinement_controls(group)
        return group

    def _build_plotter_group(self):
        group = super()._build_plotter_group()
        # The HF2LI widget is capped at 7.2 kSa/s.  The FPGA CIC can update
        # much faster, so widen the display-only range before assigning it.
        self.plotter_rate_spin.setRange(1.0, pll_update_rate_hz(16))
        self.plotter_rate_spin.setDecimals(2)
        self.plotter_rate_spin.setValue(pll_update_rate_hz(32767))
        self.plotter_rate_spin.setEnabled(False)
        self.plotter_rate_spin.setToolTip(
            "100 MHz / CIC rate로 계산한 FPGA PLL 갱신 속도입니다. "
            "현재 mode의 추가 datarate decimation은 0입니다.")
        return group

    def _expected_fpga_stream_rate_hz(self):
        if self.fpga and self.fpga.last_parameters and self.fpga.last_parameters.mode.run:
            return pll_update_rate_hz(self.fpga.last_parameters.cic_rate)
        return pll_update_rate_hz(self.fpga_cic_rate_spin.value())

    def _update_fpga_rate_estimate(self, *_args):
        rate = self._expected_fpga_stream_rate_hz()
        self.fpga_rate_estimate_label.setText(f"{rate:,.2f} Sa/s")
        if hasattr(self, "plotter_rate_spin"):
            self.plotter_rate_spin.setValue(rate)

    def _connect_hardware(self):
        if self._auto_busy():
            self.status_label.setText("자동 스윕 종료 후 재연결할 수 있습니다.")
            return
        if self.fpga is not None:
            try:
                self.emergency_stop()
                self.fpga.disconnect()
            except Exception:
                pass
        self.fpga = FPGAController(NetworkProfile())
        try:
            banner = self.fpga.connect()
            self.status_label.setText(
                f"FPGA 192.168.100.2:30 연결됨 ({banner.strip().splitlines()[-1]}) · "
                "Run=0은 사용자 명령 시 설정")
        except Exception as exc:
            self.fpga = None
            self.status_label.setText(f"FPGA 연결 실패: {exc}")

    def set_drive_amplitude_value(self, fraction):
        if self._auto_busy():
            return
        if self.fpga and self.fpga.last_parameters and self.output_on_btn.isChecked() and self.amp_enable_btn.isChecked():
            self.fpga.write_named("nco_gain", round(float(fraction) * 0xFFFFFFFF))

    def on_range_changed(self, index):
        del index  # HF2LI output range has no FPGA equivalent.

    def _write_drive_switches(self):
        if self._auto_busy():
            return
        if self.fpga and self.fpga.last_parameters:
            fraction = (self.drive_v_spin.value()
                        if self.output_on_btn.isChecked() and self.amp_enable_btn.isChecked()
                        else 0.0)
            self.fpga.write_named("nco_gain", round(fraction * 0xFFFFFFFF))

    def toggle_output(self, checked):
        self.output_on_btn.setText("On ✓" if checked else "On")
        self._write_drive_switches()

    def toggle_amp_enabled(self, checked):
        self.amp_enable_btn.setText("Amp ✓" if checked else "Amp")
        self._write_drive_switches()

    def _ensure_drive_active(self):
        self.output_on_btn.setChecked(True)
        self.amp_enable_btn.setChecked(True)

    def _fpga_parameters(self, *, fixed=False, minimum=None, maximum=None):
        center = self.pll_center_spin.value()
        span = self.pll_range_spin.value()
        low = max(1.0, center - span) if minimum is None else minimum
        high = min(6_249_999.0, center + span) if maximum is None else maximum
        return PLLUserParameters(
            frequency_hz=center if not fixed else low,
            minimum_frequency_hz=low, maximum_frequency_hz=high,
            resonator_q=self.pll_q_spin.value(),
            loop_bandwidth_hz=self.pll_target_bw_spin.value(),
            loop_order=1, cic_rate=self.fpga_cic_rate_spin.value(),
            cic_bit_shift=self.fpga_cic_shift_spin.value(),
            pll_delay_turns=self.fpga_pll_delay_spin.value(),
            drive_amplitude_fraction=self.drive_v_spin.value(),
            feedback_gain=0.1,
            mode=ModeFields(enable_agc=self.fpga_agc_check.isChecked(),
                            send_data_to_pc=True,
                            run_nco_at_fixed_frequency=fixed, dac_a_output=1))

    def apply_peak(self, row, show_msg=True):
        """Transfer a quality-accepted sweep fit to the FPGA PLL advisor."""
        if row < 0 or row >= len(self._peak_candidates):
            if show_msg:
                QMessageBox.warning(self, "FPGA PLL", "먼저 sweep 피크를 선택하세요.")
            return
        self.last_resonance_fit = None
        super().apply_peak(row, show_msg=False)
        fit = self.last_resonance_fit
        if fit is None:
            if show_msg:
                QMessageBox.warning(self, "FPGA PLL", "복소 fit 품질을 통과한 피크를 선택하세요.")
            return
        self.pll_design_label.setText(
            f"FPGA PLL 준비: f0={fit.f0_hz:,.4f} Hz, Q={fit.q:,.2f}. "
            "Target BW·capture range·delay·CIC를 확인한 뒤 PLL을 활성화하세요.")
        if show_msg:
            QMessageBox.information(self, "FPGA PLL 설정",
                                    source_widget_text(self.pll_design_label))

    def calibrate_pll_phase_detector(self):
        if self._auto_busy():
            return
        """Inspect raw FPGA error versus frequency, without degree calibration."""
        if self.fpga is None:
            return
        if ((self.sweep_worker and self.sweep_worker.isRunning()) or
                (self.plotter_worker and self.plotter_worker.isRunning()) or
                self.pll_enable_btn.isChecked() or
                (self.fpga_error_worker and self.fpga_error_worker.isRunning())):
            QMessageBox.warning(self, "FPGA error 진단",
                                "sweep·PLL·Plotter를 정지한 뒤 실행하세요.")
            return
        fit = self.last_resonance_fit
        if fit is None:
            QMessageBox.warning(self, "FPGA error 진단",
                                "먼저 복소 fit 품질을 통과한 피크를 선택하세요.")
            return
        half_span = min(self.pll_range_spin.value(),
                        max(3.0 * fit.linewidth_hz, 0.25))
        frequencies = np.linspace(fit.f0_hz - half_span,
                                  fit.f0_hz + half_span, 9)
        if frequencies[0] <= 0 or frequencies[-1] >= 6_250_000:
            QMessageBox.warning(self, "FPGA error 진단", "주파수 범위가 FPGA NCO 한계를 벗어납니다.")
            return
        self._ensure_drive_active()
        params = self._fpga_parameters(fixed=True,
                                       minimum=float(frequencies[0]),
                                       maximum=float(frequencies[-1]))
        self.fpga_error_worker = FPGAErrorScanWorker(self.fpga, params, frequencies)
        self.fpga_error_worker.completed.connect(self._on_fpga_error_scan_completed)
        self.fpga_error_worker.failed.connect(self._on_fpga_error_scan_failed)
        self.pll_design_label.setText("FPGA 원시 error 진단 중…")
        self.fpga_error_worker.start()

    def _on_fpga_error_scan_completed(self, result):
        frequencies, errors = result
        slope = float(np.polyfit(frequencies - frequencies.mean(), errors, 1)[0])
        self.pll_design_label.setText(
            f"FPGA 원시 error: f0={frequencies[4]:,.4f} Hz에서 "
            f"{errors[4]:,.0f} counts, 기울기={slope:,.4g} counts/Hz. "
            "° 단위 보정값이나 LOCK 판정은 아닙니다.")

    def _on_fpga_error_scan_failed(self, message):
        self.pll_design_label.setText(f"FPGA 원시 error 진단 실패: {message}")

    def calibrate_fpga_pll_delay(self):
        self._start_fpga_delay_calibration(start_tracking=False)

    def _start_fpga_delay_calibration(self, *, start_tracking):
        if self._auto_busy():
            return False
        if self.fpga is None:
            return False
        if ((self.sweep_worker and self.sweep_worker.isRunning()) or
                (self.fpga_error_worker and self.fpga_error_worker.isRunning()) or
                (self.fpga_delay_worker and self.fpga_delay_worker.isRunning())):
            QMessageBox.warning(self, "FPGA PLL delay", "다른 FPGA 측정이 끝난 뒤 보정하세요.")
            return False
        if not start_tracking and self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "FPGA PLL delay", "PLL을 먼저 정지하세요.")
            return False
        if self.plotter_worker and self.plotter_worker.isRunning():
            self.stop_plotter()
        self._ensure_drive_active()
        self._start_pll_after_delay = bool(start_tracking)
        self.fpga_delay_worker = FPGADelayCalibrationWorker(
            self.fpga, self._fpga_parameters())
        self.fpga_delay_worker.completed.connect(self._on_fpga_delay_calibrated)
        self.fpga_delay_worker.failed.connect(self._on_fpga_delay_calibration_failed)
        self.pll_design_label.setText(
            "FPGA PLL delay 보정 중: 현재 Drive에서 원시 error 영점·기울기를 측정합니다…")
        if start_tracking:
            self.pll_enable_btn.setText("PLL delay 보정 중…")
        self.fpga_delay_worker.start()
        return True

    def _on_fpga_delay_calibrated(self, result):
        if self.fpga_delay_worker:
            self.fpga_delay_worker.wait(1000)
        self.fpga_delay_worker = None
        start_tracking = self._start_pll_after_delay and self.pll_enable_btn.isChecked()
        self._start_pll_after_delay = False
        self.fpga_pll_delay_spin.setValue(result["delay_turns"])
        self._last_delay_calibration = result
        self.pll_design_label.setText(
            f"FPGA PLL delay={result['delay_turns']:.6f} turn 2단계 자동 보정 "
            f"(세밀 간격 {result['fine_step_turns']:.6f} turn), "
            f"f0={result['center_hz']:,.3f} Hz, Drive={result['drive_fraction']:.4f} FS, "
            f"error 잔차={result['error_counts']:,.0f} counts. "
            "양의 기울기 영점을 선택했습니다; LOCK 판정은 아닙니다.")
        if start_tracking:
            self._start_fpga_closed_loop()

    def _on_fpga_delay_calibration_failed(self, message):
        if self.fpga_delay_worker:
            self.fpga_delay_worker.wait(1000)
        self.fpga_delay_worker = None
        was_starting = self._start_pll_after_delay
        self._start_pll_after_delay = False
        self.pll_design_label.setText(f"FPGA PLL delay 보정 실패: {message}")
        if was_starting:
            self.pll_enable_btn.blockSignals(True)
            self.pll_enable_btn.setChecked(False)
            self.pll_enable_btn.blockSignals(False)
            self.pll_enable_btn.setText("PLL 활성화")
        if "중지" not in message:
            QMessageBox.warning(self, "FPGA PLL delay 보정 실패", message)

    def measure_fpga_step_response(self):
        if self._auto_busy():
            return
        """Measure the real resonator-PLL response instead of trusting target BW."""
        if self.fpga is None:
            QMessageBox.warning(self, "폐루프 응답", "FPGA 연결이 필요합니다.")
            return
        if not self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "폐루프 응답", "먼저 FPGA PLL tracking을 활성화하세요.")
            return
        if ((self.sweep_worker and self.sweep_worker.isRunning()) or
                (self.fpga_error_worker and self.fpga_error_worker.isRunning()) or
                (self.fpga_delay_worker and self.fpga_delay_worker.isRunning()) or
                (self.fpga_response_worker and self.fpga_response_worker.isRunning())):
            QMessageBox.warning(self, "폐루프 응답", "다른 FPGA 측정이 끝난 뒤 실행하세요.")
            return
        if self.baseline_active or self.pingpong_active:
            QMessageBox.warning(self, "폐루프 응답", "Baseline 또는 Ping-Pong 시험을 먼저 중지하세요.")
            return
        parameters = self.fpga.last_parameters
        if (parameters is None or not parameters.mode.run or
                parameters.mode.run_nco_at_fixed_frequency):
            QMessageBox.warning(self, "폐루프 응답", "FPGA가 tracking 모드인지 확인하세요.")
            return
        self._restart_plotter_after_response = bool(
            self.plotter_worker and self.plotter_worker.isRunning())
        if self._restart_plotter_after_response:
            self.stop_plotter()
        self.fpga_response_worker = FPGAClosedLoopStepWorker(
            self.fpga, self.fpga_pll_delay_spin.value(),
            self.fpga_step_size_spin.value(), self._expected_fpga_stream_rate_hz())
        self.fpga_response_worker.completed.connect(self._on_fpga_step_response)
        self.fpga_response_worker.failed.connect(self._on_fpga_step_response_failed)
        self.fpga_step_response_btn.setEnabled(False)
        self.fpga_step_response_btn.setText("폐루프 step 응답 측정 중…")
        self.fpga_step_response_label.setText(
            "작은 delay step을 적용해 실제 주파수 응답을 기록하고 있습니다…")
        self.fpga_response_worker.start()

    def _finish_fpga_step_response_worker(self):
        if self.fpga_response_worker:
            self.fpga_response_worker.wait(1000)
        self.fpga_response_worker = None
        self.fpga_step_response_btn.setEnabled(True)
        self.fpga_step_response_btn.setText("실제 폐루프 step 응답 · BW 측정")
        restart = self._restart_plotter_after_response
        self._restart_plotter_after_response = False
        if restart and self.pll_enable_btn.isChecked():
            self.start_plotter()

    def _on_fpga_step_response(self, payload):
        try:
            result = analyze_step_response(
                payload["time_s"], payload["frequency_hz"],
                step_time_s=payload["step_time_s"],
                restore_time_s=payload["restore_time_s"],
                delay_step_turns=payload["delay_step_turns"],
            )
            recovery = analyze_step_response(
                payload["time_s"],payload["frequency_hz"],
                step_time_s=payload["restore_time_s"],
                restore_time_s=float(payload["time_s"][-2]),
                delay_step_turns=-payload["delay_step_turns"])
            extra_reasons=[]
            if not recovery.accepted:
                extra_reasons.append("복귀 응답 검토 필요: "+"; ".join(recovery.reasons))
            if max(result.bandwidth_hz,recovery.bandwidth_hz)/min(result.bandwidth_hz,recovery.bandwidth_hz)>1.5:
                extra_reasons.append("상승/복귀 BW가 1.5배 이상 다릅니다.")
            extra_reasons.extend((payload.get("stream_health") or {}).get("warnings",[]))
            if extra_reasons:
                result=replace(result,accepted=False,reasons=result.reasons+tuple(extra_reasons))
            parameters = self.fpga.last_parameters if self.fpga else None
            metadata = {
                "step_time_s": payload["step_time_s"],
                "restore_time_s": payload["restore_time_s"],
                "base_delay_turns": payload["base_delay_turns"],
                "target_bandwidth_hz": (
                    parameters.loop_bandwidth_hz if parameters else self.pll_target_bw_spin.value()),
                "center_frequency_hz": self.pll_center_spin.value(),
                "resonator_q": self.pll_q_spin.value(),
                "cic_rate": self.fpga_cic_rate_spin.value(),
                "cic_bit_shift": self.fpga_cic_shift_spin.value(),
                "agc_enabled": self.fpga_agc_check.isChecked(),
                "stream_rate_hz_expected_from_cic": self._expected_fpga_stream_rate_hz(),
                "recovery_response": asdict(recovery),
                "stream_health": payload.get("stream_health"),
                "response_scope": "phase-reference-to-frequency response, not direct resonance-frequency disturbance",
            }
            self._last_step_response = result
            self._last_step_response_context = dict(metadata)
            csv_path, json_path, png_path = save_step_response(
                Path(__file__).resolve().parent / "diagnostics",
                payload["time_s"], payload["frequency_hz"], payload["segment"],
                result, metadata)
            target = float(metadata["target_bandwidth_hz"])
            ratio = result.bandwidth_hz / target if target > 0 else np.nan
            status = "통과" if result.accepted else "검토 필요"
            self.fpga_step_response_label.setText(
                f"실측 BW={result.bandwidth_hz:.3g} Hz / Target={target:.3g} Hz "
                f"(비={ratio:.3f}), τ={result.tau_s:.4g} s, "
                f"overshoot={result.overshoot_percent:.1f}%, R²={result.r_squared:.3f} · {status}")
            self.fpga_step_response_label.setText(source_widget_text(self.fpga_step_response_label)+
                f" · 복귀 BW={recovery.bandwidth_hz:.3g} Hz · drift={result.drift_hz_per_s:+.3g} Hz/s")
            details = (
                source_widget_text(self.fpga_step_response_label) +
                f"\n\nCSV: {csv_path}\nJSON: {json_path}\n그래프: {png_path}")
            if result.reasons:
                details += "\n\n" + "\n".join(f"• {reason}" for reason in result.reasons)
            if result.accepted:
                QMessageBox.information(self, "FPGA 폐루프 응답 측정 완료", details)
            else:
                QMessageBox.warning(self, "FPGA 폐루프 응답 검토 필요", details)
        except Exception as exc:
            self.fpga_step_response_label.setText(f"폐루프 응답 분석 실패: {exc}")
            QMessageBox.warning(self, "폐루프 응답 분석 실패", str(exc))
        finally:
            self._finish_fpga_step_response_worker()

    def _on_fpga_step_response_failed(self, message):
        self.fpga_step_response_label.setText(f"폐루프 step 응답 측정 실패: {message}")
        self._finish_fpga_step_response_worker()
        if "중지" not in message:
            QMessageBox.warning(self, "폐루프 step 응답 측정 실패", message)

    def run_sweep(self):
        if self.fpga is None:
            self.status_label.setText("FPGA 연결이 필요합니다.")
            return
        if self.sweep_worker and self.sweep_worker.isRunning():
            return
        if self.fpga_error_worker and self.fpga_error_worker.isRunning():
            QMessageBox.warning(self, "FPGA error 진단", "진단이 끝난 뒤 sweep을 시작하세요.")
            return
        if self.fpga_delay_worker and self.fpga_delay_worker.isRunning():
            QMessageBox.warning(self, "FPGA PLL delay", "보정이 끝난 뒤 sweep을 시작하세요.")
            return
        if self.plotter_worker and self.plotter_worker.isRunning():
            self.stop_plotter()
        if self.pll_enable_btn.isChecked():
            self.pll_enable_btn.setChecked(False)
        start, stop = self.f_start_spin.value(), self.f_stop_spin.value()
        if not 0 < start < stop < 6_250_000:
            QMessageBox.warning(self, "FPGA sweep 범위", "0 < 시작 < 끝 < 6.25 MHz가 필요합니다.")
            return
        self._ensure_drive_active()
        self._save_last_settings()
        n = self.n_points_spin.value()
        self.sweep_freqs = np.full(n, np.nan)
        self.sweep_R = np.full(n, np.nan)
        self.sweep_theta = np.full(n, np.nan)
        self.sweep_X = np.full(n, np.nan)
        self.sweep_Y = np.full(n, np.nan)
        self.sweep_amp_curve.setData([], [])
        self.sweep_phase_curve.setData([], [])
        self.sweep_fit_amp_curve.setData([], [])
        self.sweep_fit_phase_curve.setData([], [])
        self.peak_markers.setData([], [])
        self.peak_list.clear()
        self._peak_candidates = []
        self.last_resonance_fit = None
        self.tabs.setCurrentIndex(0)
        self._fpga_sweep_index = 0
        self.sweep_worker = FPGASweepWorker(
            self.fpga, self._fpga_parameters(fixed=True, minimum=start, maximum=stop),
            np.linspace(start, stop, n),
            max(0.05, self.sweep_min_settle_spin.value()),
            self.sweep_samples_spin.value())
        self.sweep_worker.point.connect(self._on_fpga_sweep_point)
        self.sweep_worker.completed.connect(self._on_fpga_sweep_completed)
        self.sweep_worker.failed.connect(self.on_sweep_failed)
        self.run_sweep_btn.setEnabled(False)
        self.stop_sweep_btn.setEnabled(True)
        self.sweep_worker.start()

    def _on_fpga_sweep_point(self, f, r, theta, x, y):
        index = self._fpga_sweep_index
        self._fpga_sweep_index += 1
        self.on_sweep_point(index, f, r, theta, x, y)
        self.on_sweep_progress(index + 1, self.n_points_spin.value())

    def _on_fpga_sweep_completed(self, result):
        count = len(result["freqs"])
        result = dict(result)
        result.update({
            "time_constants": np.full(count, np.nan),
            "bandwidths": np.full(count, np.nan),
            "settling_times": np.full(count, max(0.05, self.sweep_min_settle_spin.value())),
            "measurement_times": np.full(count, np.nan),
        })
        self.on_sweep_finished(result)

    def export_sweep_csv(self):
        if not len(self.sweep_freqs):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "FPGA sweep 저장", "fpga_sweep.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["# Metadata", "Backend", "SMR FPGA channel A"])
            writer.writerow(["# Metadata", "I_Q_unit", "uncalibrated FPGA counts"])
            writer.writerow(["Frequency_Hz", "Amplitude_R_counts", "Phase_deg",
                             "X_counts", "Y_counts", "X_SEM_counts",
                             "Y_SEM_counts", "N_samples"])
            writer.writerows(zip(self.sweep_freqs, self.sweep_R,
                                 self.sweep_theta, self.sweep_X, self.sweep_Y,
                                 self.sweep_X_sem, self.sweep_Y_sem,
                                 self.sweep_metadata.get("n_samples", [])))

    def detect_peaks(self, freqs, R, theta):
        super().detect_peaks(freqs, R, theta)
        if self.peak_list.count() == 1 and "noise" in self.peak_list.item(0).text():
            self.peak_list.item(0).setText(
                self.peak_list.item(0).text().replace(" V)", " counts)"))

    def toggle_pll(self, checked):
        if self.fpga is None:
            self.pll_enable_btn.blockSignals(True)
            self.pll_enable_btn.setChecked(False)
            self.pll_enable_btn.blockSignals(False)
            return
        if checked:
            try:
                started=self.start_pll_preflight()
            except Exception as exc:
                self._abort_pll_start(str(exc))
                return
            if not started:
                self.pll_enable_btn.blockSignals(True)
                self.pll_enable_btn.setChecked(False)
                self.pll_enable_btn.blockSignals(False)
        else:
            self._start_pll_after_delay = False
            self._stop_peak_refinement()
            if self.fpga_response_worker and self.fpga_response_worker.isRunning():
                self.fpga_response_worker.stop()
                self.fpga_response_worker.wait(3000)
            if self.fpga_delay_worker and self.fpga_delay_worker.isRunning():
                self.fpga_delay_worker.stop()
                self.fpga_delay_worker.wait(3000)
            if self.plotter_worker and self.plotter_worker.isRunning():
                self.stop_plotter()
            try:
                self.fpga.stop_output()
                self.fpga.stop_stream()
            except Exception as exc:
                self.status_label.setText(f"FPGA 정지 확인 실패: {exc}")
            self.pll_enable_btn.setText("PLL 활성화")
            self.pll_locked_label.setText("FPGA PLL: 비활성")
            self.pll_big_locked.setText("PLL: 비활성")

    def _start_fpga_closed_loop(self):
        initial=getattr(self,"_initial_parameters",None)
        if initial is not None and not self._pll_start_validated:
            current=self._fpga_parameters()
            if current != replace(initial,pll_delay_turns=current.pll_delay_turns):
                self._abort_pll_start("공진 검증 후 설정이 변경되었습니다. PLL 시작을 다시 실행하세요.")
                return
        try:
            if self.fpga_error_worker and self.fpga_error_worker.isRunning():
                raise RuntimeError("원시 error 진단이 끝난 뒤 PLL을 시작하세요.")
            if self.fpga_delay_worker and self.fpga_delay_worker.isRunning():
                raise RuntimeError("delay 보정이 끝난 뒤 PLL을 시작하세요.")
            if self.sweep_worker and self.sweep_worker.isRunning():
                raise RuntimeError("Sweep이 끝난 뒤 PLL을 시작하세요.")
            if self.plotter_worker and self.plotter_worker.isRunning():
                self.stop_plotter()
            self._ensure_drive_active()
            self.fpga.start_stream(signal="frequency")
            self.fpga.apply_parameters(self._fpga_parameters())
            self.fpga.start_tracking(stream=True)
            self.fpga.restart_all_plls()
            self.pll_enable_btn.setText("PLL 비활성화")
            self._update_fpga_pll_status()
            self.pll_big_locked.setText("LOCK 미확인")
            if self.fpga_peak_auto_check.isChecked():
                self.start_fpga_peak_refinement()
                return
            self.start_plotter()
            self.tabs.setCurrentIndex(3 if self.live_particle_enabled.isChecked() else 2)
        except Exception as exc:
            try:
                self.fpga.stop_output()
            except Exception:
                pass
            self.pll_enable_btn.blockSignals(True)
            self.pll_enable_btn.setChecked(False)
            self.pll_enable_btn.blockSignals(False)
            self.pll_enable_btn.setText("PLL 활성화")
            QMessageBox.critical(self, "FPGA PLL 시작 실패", str(exc))

    def start_plotter(self, baseline=False):
        if self._pll_validation_failed:
            self.pll_design_label.setText("공진 검증이 완료되지 않았습니다. 스윕/피크를 확인하고 PLL 시작을 다시 실행하세요.")
            return
        if self.pll_enable_btn.isChecked() and not self._pll_start_validated:
            self.pll_design_label.setText("공진점과 폐루프 진폭 검증 완료 후 기록할 수 있습니다.")
            return
        if self._auto_busy():
            return
        if self.fpga is None or (self.plotter_worker and self.plotter_worker.isRunning()):
            return
        if self.fpga_response_worker and self.fpga_response_worker.isRunning():
            QMessageBox.warning(self, "폐루프 응답", "step 응답 측정이 끝난 뒤 Plotter를 시작하세요.")
            return
        if self.fpga_error_worker and self.fpga_error_worker.isRunning():
            QMessageBox.warning(self, "FPGA error 진단", "진단이 끝난 뒤 Plotter를 시작하세요.")
            return
        if self.fpga_delay_worker and self.fpga_delay_worker.isRunning():
            QMessageBox.warning(self, "FPGA PLL delay", "보정이 끝난 뒤 Plotter를 시작하세요.")
            return
        if not self.pll_enable_btn.isChecked():
            try:
                center = self.pll_center_spin.value()
                self._ensure_drive_active()
                self.fpga.start_stream(signal="frequency")
                self.fpga.apply_parameters(self._fpga_parameters(fixed=True,
                                             minimum=center, maximum=center))
                self.fpga.start_fixed(stream=True)
                self._fpga_fixed_plotter = True
            except Exception as exc:
                QMessageBox.critical(self, "FPGA Plotter 시작 실패", str(exc))
                return
        else:
            self._fpga_fixed_plotter = False
            try:
                # A stopped Plotter must not reuse an old multicast receiver.
                # Keep the PLL running while opening a fresh frequency session.
                self.fpga.stop_stream()
                self.fpga.start_stream(signal="frequency")
                if not self.fpga.last_parameters.mode.send_data_to_pc:
                    self.fpga.start_tracking(stream=True)
            except Exception as exc:
                QMessageBox.critical(self, "FPGA Plotter 시작 실패", str(exc))
                return
        self.baseline_active = bool(baseline)
        try:
            self._reset_plotter_logs()
        except OSError as exc:
            QMessageBox.critical(self, "기록 파일 오류", str(exc))
            return
        self.baseline_analysis_result = None
        self.baseline_analysis_source = None
        self.plotter_actual_rate_hz = np.nan
        self._last_plotter_ui_update_t = -np.inf
        self._last_baseline_ui_update_t = -np.inf
        self._last_filtered_ui_update_t = -np.inf
        self.measurement_metadata = self._capture_measurement_metadata(self.baseline_active)
        self.measurement_metadata["Temporary_raw_recording"] = bool(self.plotter_spool)
        self.plotter_freq_curve.setData([], [])
        self.plotter_r_curve.setData([], [])
        # Legend clicks can hide a trace, and a restarted worker starts its
        # time axis at zero while the ViewBox may retain the old session's zoom.
        self.live_frequency_curve.setVisible(True)
        self.live_filtered_frequency_curve.setVisible(
            self.live_savgol_enabled.isChecked())
        self.live_particle_plot.enableAutoRange(axis="y", enable=True)
        self.live_particle_plot.setXRange(
            0, self.plotter_window_spin.value(), padding=0)
        self.live_filtered_frequency_curve.setData([], [])
        self.reset_live_particle_detector()
        self.tabs.setCurrentIndex(4 if baseline else 3)
        self.plotter_worker = FPGAPlotterWorker(
            self.fpga, self._expected_fpga_stream_rate_hz())
        worker = self.plotter_worker
        worker.batch_ready.connect(
            lambda batch, source=worker: self._on_current_plotter_batch(source, batch))
        worker.stream_started.connect(
            lambda requested, actual, source=worker:
            self.on_plotter_stream_started(requested, actual)
            if self.plotter_worker is source and source.isRunning() else None)
        worker.failed.connect(
            lambda message, source=worker: self.on_plotter_failed(message)
            if self.plotter_worker is source else None)
        worker.health_ready.connect(lambda stats,source=worker:
            self._on_fpga_stream_health(stats) if self.plotter_worker is source else None)
        self.plotter_rate_spin.setEnabled(False)
        self.plotter_start_btn.setEnabled(False)
        self.plotter_stop_btn.setEnabled(True)
        self._update_plotter_pause_ui(running=True)
        self.plotter_record_check.setEnabled(False)
        self.save_plotter_btn.setEnabled(self.plotter_spool is not None)
        self.baseline_start_btn.setEnabled(not baseline)
        self.baseline_stop_btn.setEnabled(baseline)
        self.plotter_worker.start()

    def stop_plotter(self):
        super().stop_plotter()
        self.plotter_worker = None
        self.plotter_rate_spin.setEnabled(False)
        if self.fpga:
            try:
                if self._fpga_fixed_plotter:
                    self.fpga.stop_output()
                self.fpga.stop_stream()
            finally:
                self._fpga_fixed_plotter = False

    def _on_current_plotter_batch(self, source, batch):
        if self.plotter_worker is source and source.isRunning():
            self.on_plotter_batch(batch)

    def _on_fpga_stream_health(self, stats):
        self.measurement_metadata.update({"FPGA_stream_"+k:v for k,v in stats.items()})
        delivered=stats["host_delivery_rate_hz"]
        text=f"CIC 예상 {stats['expected_rate_hz']:,.2f} Sa/s · host 수신 {delivered:,.1f} Sa/s"
        if stats["warnings"]:
            text += " · "+" / ".join(stats["warnings"])
        self.plotter_actual_rate_label.setText(text)
        self.plotter_actual_rate_label.setToolTip("host 도착률이며 장치 clock 실측값이 아닙니다. 미검증 UDP header는 누락 판정에 사용하지 않습니다.")

    def _refresh_plotter_ui(self, t):
        super()._refresh_plotter_ui(t)
        # Keep the live chart on the latest samples even after manual panning
        # or a Plotter restart. The worker's time axis is local to each run.
        window = self.plotter_window_spin.value()
        self.live_particle_plot.setXRange(
            max(0.0, t - window), max(window, t), padding=0)

    def _capture_measurement_metadata(self, baseline):
        metadata = super()._capture_measurement_metadata(baseline)
        parameters = self.fpga.last_parameters if self.fpga else None
        gains = (user_parameters_to_registers(parameters)
                 if parameters is not None else None)
        response_context = self._last_step_response_context or {}
        peak=self._last_peak_refinement
        peak_context=(peak or {}).get("context",{})
        peak_current=bool(peak and parameters and
            abs(peak["delay_turns"]-parameters.pll_delay_turns)<1e-6 and
            peak_context.get("center_hz")==parameters.frequency_hz and
            peak_context.get("cic_rate")==parameters.cic_rate and
            peak_context.get("drive_fraction")==parameters.drive_amplitude_fraction and
            peak_context.get("agc")==parameters.mode.enable_agc)
        response_current = bool(
            self._last_step_response and parameters and
            abs(response_context.get("target_bandwidth_hz", np.inf) -
                parameters.loop_bandwidth_hz) < 1e-9 and
            abs(response_context.get("center_frequency_hz", np.inf) -
                parameters.frequency_hz) < 0.1 and
            abs(response_context.get("base_delay_turns", np.inf) -
                parameters.pll_delay_turns) < 1e-6 and
            response_context.get("cic_rate") == parameters.cic_rate and
            response_context.get("cic_bit_shift") == parameters.cic_bit_shift and
            response_context.get("agc_enabled") == parameters.mode.enable_agc)
        metadata.update({
            "FPGA_initial_resonance_validation": getattr(self,"_initial_resonance_result",None),
            "FPGA_PLL_start_validated": self._pll_start_validated,
            "Backend": "SMR FPGA channel A",
            "Acquisition_mode": "UDP PLL frequency; host reconstructed time",
            "PLL_status_logging": "lock bit unavailable; raw error requires a separate selected UDP stream",
            "PLL_update_rate_Hz_expected_from_CIC": self._expected_fpga_stream_rate_hz(),
            "FPGA_PLL_datarate_decimation": 0,
            "PLL_Kp_Hz_per_deg": np.nan,
            "PLL_Ki_Hz2_per_deg": np.nan,
            "PLL_Kd_per_deg": np.nan,
            "PLL_filter_time_constant_s": np.nan,
            "PLL_target_bandwidth_Hz": (
                parameters.loop_bandwidth_hz if parameters else self.pll_target_bw_spin.value()),
            "FPGA_PLL_target_bandwidth_UI_Hz": self.pll_target_bw_spin.value(),
            "PLL_target_phase_deg": np.nan,
            "Excitation_amplitude_Vpk": np.nan,
            "FPGA_drive_fraction_full_scale": self.drive_v_spin.value(),
            "FPGA_PLL_delay_turns": self.fpga_pll_delay_spin.value(),
            "FPGA_CIC_rate": self.fpga_cic_rate_spin.value(),
            "FPGA_CIC_bit_shift": self.fpga_cic_shift_spin.value(),
            "FPGA_PLL_gain_formula": "LabVIEW VI frequency-and-CIC-dependent",
            "FPGA_PLL_gain_proportional_word": (
                gains.gain_proportional if gains else np.nan),
            "FPGA_PLL_gain_integral_word": (
                gains.gain_integral if gains else np.nan),
            "FPGA_AGC_enabled": (
                parameters.mode.enable_agc if parameters else self.fpga_agc_check.isChecked()),
            "FPGA_AGC_UI_enabled": self.fpga_agc_check.isChecked(),
            "FPGA_closed_loop_peak_refinement": peak if peak_current else None,
            "FPGA_PLL_delay_auto_calibrated": bool(
                self._last_delay_calibration and
                abs(self._last_delay_calibration["center_hz"] - self.pll_center_spin.value()) < 0.1 and
                abs(self._last_delay_calibration["drive_fraction"] - self.drive_v_spin.value()) < 1e-5 and
                abs(self._last_delay_calibration["delay_turns"] - self.fpga_pll_delay_spin.value()) < 1e-5),
            "FPGA_PLL_raw_error_at_calibration_counts": (
                self._last_delay_calibration["error_counts"]
                if self._last_delay_calibration else np.nan),
            "FPGA_PLL_measured_closed_loop_bandwidth_Hz": (
                self._last_step_response.bandwidth_hz
                if response_current else np.nan),
            "FPGA_PLL_measured_step_tau_s": (
                self._last_step_response.tau_s
                if response_current else np.nan),
            "FPGA_PLL_step_response_accepted": (
                self._last_step_response.accepted
                if response_current else False),
            "Amplitude_R_unit": "unavailable in simultaneous frequency stream",
        })
        return metadata

    def on_plotter_stream_started(self, requested_rate_hz, assumed_rate_hz):
        self.plotter_actual_rate_hz = float(assumed_rate_hz)
        self.measurement_metadata["FPGA_stream_rate_Hz_expected_from_CIC"] = float(assumed_rate_hz)
        self.measurement_metadata["Demod_stream_rate_Hz_actual"] = np.nan
        self.measurement_metadata["Demod_stream_rate_Hz_requested"] = np.nan
        self.plotter_actual_rate_label.setText(
            f"FPGA CIC 계산 속도: {assumed_rate_hz:,.2f} Sa/s (실측 미검증)")

    def _record_plotter_sample(self, t, r, theta, f, x, y, locked,
                               phase_error, correction, demod_latency, pll_latency):
        row = super()._record_plotter_sample(
            t, r, theta, f, x, y, False, np.nan, np.nan,
            demod_latency, pll_latency)
        if row is not None:
            row[7] = np.nan  # HF2LI phase setpoint is not an FPGA observation
            row[11] = np.nan  # lock state unknown, not false
        if self.log_t and self.log_t[-1] == t:
            self.log_pll_target_phase[-1] = np.nan
            self.log_pll_locked[-1] = np.nan
        return row

    def on_plotter_batch(self, batch):
        if self.plotter_paused:
            return
        super().on_plotter_batch(batch)
        if batch and self.pll_enable_btn.isChecked():
            frequency = float(batch[-1][3])
            self.pll_big_freq.setText(f"{frequency:,.4f}")
            self.pll_big_shift.setText(
                f"{frequency - self.pll_center_spin.value():+.4f}")
            self.pll_big_error.setText("측정 불가")
            self.pll_big_locked.setText("LOCK 미확인")
            self._update_fpga_pll_status()
            p=self.fpga.last_parameters if self.fpga else None
            if p:
                margin=.02*(p.maximum_frequency_hz-p.minimum_frequency_hz)
                values=np.array([row[3] for row in batch],float)
                if np.any((values<=p.minimum_frequency_hz+margin)|(values>=p.maximum_frequency_hz-margin)):
                    self.pll_locked_label.setText("FPGA 추적 주의: capture range 경계 도달 · LOCK 미확인")
                    self.measurement_metadata["FPGA_capture_boundary_seen"]=True
                elif len(values)>1 and np.any(abs(np.diff(values))>.25*p.frequency_hz/p.resonator_q):
                    self.pll_locked_label.setText("FPGA 추적 주의: 선폭 대비 큰 주파수 점프 · LOCK 미확인")
                    self.measurement_metadata["FPGA_large_frequency_jump_seen"]=True

    def start_pingpong(self):
        if self.fpga is None or not self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "FPGA PLL 필요",
                                "주파수 추적 PLL을 먼저 활성화하세요.")
            return
        if self.baseline_active:
            QMessageBox.warning(self, "Baseline 모드",
                                "Baseline 테스트를 중지한 뒤 시작하세요.")
            return
        if self.plotter_worker is None or not self.plotter_worker.isRunning():
            self.start_plotter()
        self.pingpong_state = "baseline"
        self.pingpong_baseline_f = None
        self.pingpong_transit_start_t = None
        self.pingpong_transit_min_f = None
        self.pingpong_forward = True
        self.pingpong_cycle_count = 0
        self.pingpong_settle_until_t = None
        self.pingpong_events = []
        self.pp_event_list.clear()
        self.pingpong_markers.setData([], [])
        self._pingpong_apply_direction(forward=True)
        self.pingpong_active = True
        self.pp_start_btn.setEnabled(False)
        self.pp_stop_btn.setEnabled(True)
        self.pp_status_label.setText("상태: 기준선 수집 중...")
        self.tabs.setCurrentIndex(3 if self.live_particle_enabled.isChecked() else 2)

    def emergency_stop(self):
        self._stop_peak_refinement()
        daq_was_active = self.pingpong_active or any(
            abs(value) > 1e-12 for value in self.pressure_commands.values())
        self.pingpong_active = False
        self.pp_start_btn.setEnabled(True)
        self.pp_stop_btn.setEnabled(False)
        if self.sweep_worker:
            self.sweep_worker.stop()
            self.sweep_worker.wait(3000)
        if self.fpga_error_worker:
            self.fpga_error_worker.stop()
            self.fpga_error_worker.wait(3000)
        if self.fpga_delay_worker:
            self.fpga_delay_worker.stop()
            self.fpga_delay_worker.wait(3000)
        if self.fpga_response_worker:
            self.fpga_response_worker.stop()
            self.fpga_response_worker.wait(3000)
        if self.plotter_worker:
            self.plotter_worker.stop()
            self.plotter_worker.wait(1000)
        self.pll_enable_btn.setChecked(False)
        self.output_on_btn.setChecked(False)
        self.amp_enable_btn.setChecked(False)
        self.plotter_start_btn.setEnabled(True)
        self.plotter_stop_btn.setEnabled(False)
        self.plotter_paused = False
        self._update_plotter_pause_ui(running=False)
        self.plotter_record_check.setEnabled(True)
        if self.fpga:
            try:
                self.fpga.stop_output()
                self.fpga.stop_stream()
            except Exception as exc:
                self.status_label.setText(f"FPGA 정지 명령 실패: {exc}")
        if daq_was_active:
            self.zero_daq_channels()
        self.status_label.setText(source_widget_text(self.status_label) + " [긴급 정지 실행됨]")

    def closeEvent(self, event):
        if not self._stop_peak_refinement():
            event.ignore()
            return
        if self._auto_busy():
            self.stop_auto_sweep()
            event.ignore()
            self.auto_status.setText("자동 스윕 중지·저장 중입니다. 완료 후 창을 닫아주세요.")
            return
        super().closeEvent(event)
        if not event.isAccepted():
            return
        if hasattr(self,"language_manager"):
            self.language_manager.shutdown()
        if self.fpga:
            self.fpga.disconnect()
            self.fpga = None


def main():
    app = QApplication(sys.argv)
    window = FPGAMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
