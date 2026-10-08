"""Python GUI for the supplied DE2-115 SMR lock-in/PLL firmware.

Start in Simulator mode to exercise the entire sweep -> fit -> tracking ->
save path.  Hardware mode never connects or energizes an output at startup.
Unknown legacy protocol values remain editable in the Connection tab.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import sys
import time

import numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

from fpga_backend import FPGAController, SimulatedFPGAController
from fpga_registers import ModeFields, PLLUserParameters
from fpga_transport import NetworkProfile
from resonance_fit import assess_fit_quality, fit_complex_resonance
from sweep_math import average_complex_samples


APP_VERSION = "2026.09.17.1"
SIGNAL_OPTIONS = [
    ("PLL frequency", 0), ("error signal", 1), ("magnitude", 2),
    ("AGC error", 3), ("mixdown (I/Q, experimental)", 4),
]


def write_tracking_csv(path, rows, *, backend, time_source, stream_rate_hz,
                       local_queue_dropped_batches=0, stream_signal="frequency"):
    """Write an analysis-compatible tracking file without inventing fields."""
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# Metadata", "Backend", backend])
        writer.writerow(["# Metadata", "Time_source", time_source])
        writer.writerow(["# Metadata", "Stream_rate_Hz", stream_rate_hz])
        writer.writerow(["# Metadata", "Stream_signal", stream_signal])
        writer.writerow(["# Metadata", "Local_queue_dropped_batches",
                         local_queue_dropped_batches])
        writer.writerow(["Time_s", "Frequency_Hz", "Amplitude_R_V", "Phase_deg",
                         "X_V", "Y_V", "PacketIndex", "SequenceClass",
                         "MissingPacketsBefore", "SegmentID", "Valid",
                         "Amplitude_counts"])
        writer.writerows(tuple(row) + ((np.nan,) if len(row) == 11 else ())
                         for row in rows)


def _double(value, minimum, maximum, decimals=6, suffix=""):
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(suffix)
    return widget


def _integer(value, minimum, maximum):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class SweepWorker(QThread):
    point = pyqtSignal(float, float, float, float, float)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, controller, parameters, frequencies, settle_s, samples_per_point):
        super().__init__()
        self.controller = controller
        self.parameters = parameters
        self.frequencies = np.asarray(frequencies, float)
        self.settle_s = float(settle_s)
        self.samples_per_point = int(samples_per_point)
        self.stop_requested = False
        self.last_result = {}

    def _pause(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stop_requested and time.monotonic() < deadline:
            self.msleep(max(1, min(20, int((deadline - time.monotonic()) * 1000))))

    def stop(self):
        self.stop_requested = True

    def _hardware_complex_samples(self):
        needed = 2 * self.samples_per_point
        words = []
        deadline = time.monotonic() + max(2.0, self.settle_s + 2.0)
        while len(words) < needed and time.monotonic() < deadline and not self.stop_requested:
            for batch in self.controller.drain_stream():
                if batch.sequence_classification in ("duplicate", "reordered"):
                    continue
                words.extend(batch.values.tolist())
            self.msleep(5)
        if self.stop_requested:
            return None
        if len(words) < needed:
            raise RuntimeError(
                f"mixdown payload 부족: {len(words)} words 수신, {needed} 필요. "
                "UDP port/Signal of interest/전송 형식을 확인하세요.")
        raw = np.asarray(words[:needed], float)
        # The working FPGA_Python GUI fits mean signed ADC counts. There is
        # no measured ADC-count-to-volt calibration in the supplied code.
        return raw[0::2] + 1j * raw[1::2]

    def measure(self):
        """Blocking segment acquisition; cleanup completes before returning.

        May be called by an owning sequence worker without starting another
        thread. last_result retains partial data even on stop or I/O failure.
        """
        rows = []
        requested = []
        hardware = isinstance(self.controller, FPGAController)
        try:
            if self.stop_requested:
                return
            if hardware:
                self.controller.start_stream(signal="signed")
                self.controller.apply_verified_sweep_preset(
                    float(self.frequencies[0]), float(self.frequencies[-1]),
                    self.parameters.drive_amplitude_fraction)
                self._pause(0.350)  # discard the captured preset's transition packets
                self.controller.drain_stream(1000)
            else:
                mode = replace(self.parameters.mode, run=False, send_data_to_pc=False,
                               run_nco_at_fixed_frequency=True)
                self.controller.apply_parameters(replace(self.parameters, mode=mode))
            if self.stop_requested:
                return
            self.controller.start_fixed(stream=hardware)
            if hardware:
                self._pause(0.250)
                self.controller.drain_stream(1000)
            for frequency in self.frequencies:
                if self.stop_requested:
                    break
                actual = self.controller.set_fixed_frequency(float(frequency))
                actual = float(frequency) if actual is None else float(actual)
                self._pause(max(0.001, self.settle_s))
                if self.stop_requested:
                    break
                if hardware:
                    # Old packets may span the NCO transition. Only samples
                    # received after the requested settle interval belong to
                    # this sweep point.
                    self.controller.drain_stream(1000)
                    samples = self._hardware_complex_samples()
                else:
                    samples = self.controller.complex_response(frequency, self.samples_per_point)
                if samples is None or self.stop_requested:
                    break
                z, x_sem, y_sem, covariance, count = average_complex_samples(
                    samples.real, samples.imag)
                requested.append(float(frequency))
                rows.append((actual, abs(z), np.angle(z, deg=True), z.real, z.imag,
                             x_sem, y_sem, covariance, count))
                self.point.emit(actual, abs(z), np.angle(z, deg=True), z.real, z.imag)
        finally:
            names = ("freqs", "R", "theta", "X", "Y", "X_sem", "Y_sem", "XY_cov", "n_samples")
            columns = list(zip(*rows)) if rows else [[] for _ in names]
            self.last_result = {name: np.asarray(values, float)
                                for name, values in zip(names, columns)}
            self.last_result["requested_freqs"] = np.asarray(requested, float)
            try:
                self.controller.stop_output()
            finally:
                if hardware:
                    self.controller.stop_stream()
        return self.last_result

    def run(self):
        try:
            self.measure()
            self.completed.emit(self.last_result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FPGAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FPGA Resonance PLL GUI v{APP_VERSION}")
        self.resize(1380, 900)
        self.controller = None
        self.sweep_worker = None
        self.last_sweep = None
        self.last_fit = None
        self.last_fit_quality = None
        self.tracking_rows = []
        self.tracking_t, self.tracking_f = [], []
        self.tracking_sample_index = 0
        self.tracking_first_monotonic = None
        self.tracking_segment = 0
        self.tracking_last_packet_monotonic = None
        self.tracking_ignore_before_monotonic = None
        self.stream_error_reported = False
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll_tracking)
        self._set_connected(False)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._connection_tab(), "1. 연결")
        self.tabs.addTab(self._sweep_tab(), "2. Sweep / Fit")
        self.tabs.addTab(self._pll_tab(), "3. PLL Tracking")
        root.addWidget(self.tabs)
        self.status = QLabel("연결되지 않음 — 시작 시 출력은 활성화되지 않습니다.")
        self.status.setStyleSheet("font-weight: 700; padding: 6px;")
        root.addWidget(self.status)
        self.setCentralWidget(central)

    def _connection_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        connection = QGroupBox("Backend / Network profile")
        form = QFormLayout(connection)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Simulator", "FPGA hardware"])
        self.host_edit = QLineEdit("192.168.100.2")
        self.tcp_port = _integer(30, 1, 65535)
        self.group_edit = QLineEdit("224.1.1.1")
        self.udp_port = _integer(5007, 1, 65535)
        self.interface_edit = QLineEdit("192.168.100.1")
        self.channel_spin = _integer(0, 0, 255)
        self.stride_spin = _integer(0x20, 0, 65535)
        form.addRow("Backend", self.backend_combo)
        form.addRow("FPGA host", self.host_edit)
        form.addRow("TCP port (미검증 기본값)", self.tcp_port)
        form.addRow("Multicast group", self.group_edit)
        form.addRow("UDP port (미검증 기본값)", self.udp_port)
        form.addRow("PC Ethernet address", self.interface_edit)
        form.addRow("SMR channel", self.channel_spin)
        form.addRow("Channel register stride", self.stride_spin)
        layout.addWidget(connection)

        hardware = QGroupBox("PLL / NCO parameters")
        grid = QGridLayout(hardware)
        self.center = _double(29400, 0, 12_500_000, 6, " Hz")
        self.minimum = _double(29200, 0, 12_500_000, 6, " Hz")
        self.maximum = _double(29600, 0, 12_500_000, 6, " Hz")
        self.q = _double(1200, 1, 1e9, 3)
        self.loop_bw = _double(3, 0.001, 50000, 4, " Hz")
        self.loop_order = _integer(1, 1, 5)
        self.cic_rate = _integer(32767, 16, 32767)
        self.cic_shift = _integer(21, 0, 31)
        self.delay_turns = _double(0.079, 0, 1, 9, " turn")
        self.drive_fraction = _double(0.02, 0, 1, 6, " FS")
        self.stream_rate = _double(3051, 0.01, 1e7, 3, " Sa/s")
        fields = [
            ("Center", self.center), ("Min", self.minimum), ("Max", self.maximum),
            ("Resonator Q", self.q), ("Loop bandwidth", self.loop_bw),
            ("Loop order", self.loop_order), ("CIC rate", self.cic_rate),
            ("CIC shift", self.cic_shift), ("PLL delay", self.delay_turns),
            ("Drive amplitude", self.drive_fraction),
            ("Measured stream rate", self.stream_rate),
        ]
        for index, (label, widget) in enumerate(fields):
            row, col = divmod(index, 2)
            grid.addWidget(QLabel(label), row, 2*col)
            grid.addWidget(widget, row, 2*col+1)
        layout.addWidget(hardware)

        warning = QLabel(
            "FPGA_Python의 검증 프로파일: UDP 5007, little-endian, channel A, "
            "Type-2 PI PLL의 목표 폐루프 응답 1차. Drive는 full-scale 비율이며 전압이 아닙니다. "
            "PIO 읽기는 설정값 확인에 사용할 수 없습니다.")
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a5b00; padding: 8px;")
        layout.addWidget(warning)
        buttons = QHBoxLayout()
        self.connect_button = QPushButton("연결")
        self.connect_button.clicked.connect(self.connect_backend)
        self.disconnect_button = QPushButton("출력 정지 후 연결 해제")
        self.disconnect_button.clicked.connect(self.disconnect_backend)
        buttons.addWidget(self.connect_button); buttons.addWidget(self.disconnect_button)
        layout.addLayout(buttons); layout.addStretch(1)
        return tab

    def _sweep_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.sweep_start = _double(29200, 0, 12_500_000, 3, " Hz")
        self.sweep_stop = _double(29600, 0, 12_500_000, 3, " Hz")
        self.sweep_points = _integer(401, 9, 100000)
        self.sweep_settle = _double(0.02, 0.001, 10, 4, " s")
        self.sweep_samples = _integer(256, 1, 100000)
        for label, widget in (("Start", self.sweep_start), ("Stop", self.sweep_stop),
                              ("Points", self.sweep_points), ("Settle", self.sweep_settle),
                              ("Samples/point", self.sweep_samples)):
            controls.addWidget(QLabel(label)); controls.addWidget(widget)
        layout.addLayout(controls)
        buttons = QHBoxLayout()
        self.sweep_button = QPushButton("Sweep 실행")
        self.sweep_button.clicked.connect(self.run_sweep)
        self.sweep_stop_button = QPushButton("Sweep 중지")
        self.sweep_stop_button.clicked.connect(self.stop_sweep)
        self.use_fit_button = QPushButton("Fit 결과를 PLL 설정에 적용")
        self.use_fit_button.clicked.connect(self.use_fit)
        self.use_fit_button.setEnabled(False)
        self.export_sweep_button = QPushButton("Sweep CSV 저장")
        self.export_sweep_button.clicked.connect(self.export_sweep)
        for button in (self.sweep_button, self.sweep_stop_button,
                       self.use_fit_button, self.export_sweep_button): buttons.addWidget(button)
        layout.addLayout(buttons)
        self.sweep_plot = pg.PlotWidget(title="Amplitude / Phase")
        self.sweep_plot.showGrid(x=True, y=True, alpha=0.25)
        self.sweep_plot.setLabel("bottom", "Frequency", units="Hz")
        self.sweep_plot.setLabel("left", "Amplitude", units="relative")
        self.amplitude_curve = self.sweep_plot.plot(pen=pg.mkPen("#1b65d6", width=2))
        self.phase_view = pg.ViewBox()
        self.sweep_plot.scene().addItem(self.phase_view)
        self.sweep_plot.getAxis("right").setLabel("Phase", units="deg")
        self.sweep_plot.showAxis("right")
        self.sweep_plot.getAxis("right").linkToView(self.phase_view)
        self.phase_view.setXLink(self.sweep_plot)
        self.phase_curve = pg.PlotCurveItem(pen=pg.mkPen("#d13b73", width=2))
        self.phase_view.addItem(self.phase_curve)
        self.sweep_plot.getViewBox().sigResized.connect(self._sync_phase_view)
        layout.addWidget(self.sweep_plot)
        self.fit_label = QLabel("아직 sweep/fit 결과가 없습니다.")
        self.fit_label.setWordWrap(True); layout.addWidget(self.fit_label)
        return tab

    def _sync_phase_view(self):
        view = self.sweep_plot.getViewBox()
        self.phase_view.setGeometry(view.sceneBoundingRect())
        self.phase_view.linkedViewChanged(view, self.phase_view.XAxis)

    def _pll_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        buttons = QHBoxLayout()
        self.stream_choice = QComboBox()
        self.stream_choice.addItems(["PLL frequency", "PLL amplitude"])
        buttons.addWidget(QLabel("FPGA stream")); buttons.addWidget(self.stream_choice)
        self.track_button = QPushButton("설정 적용 + PLL Tracking 시작")
        self.track_button.clicked.connect(self.start_tracking)
        self.track_stop_button = QPushButton("출력 및 Tracking 정지")
        self.track_stop_button.clicked.connect(self.stop_tracking)
        self.save_tracking_button = QPushButton("Tracking CSV 저장")
        self.save_tracking_button.clicked.connect(self.save_tracking)
        buttons.addWidget(self.track_button); buttons.addWidget(self.track_stop_button)
        buttons.addWidget(self.save_tracking_button); layout.addLayout(buttons)
        self.lock_label = QLabel("상태: 정지")
        self.lock_label.setAlignment(Qt.AlignCenter)
        self.lock_label.setStyleSheet("font-size: 24px; font-weight: 800; padding: 10px;")
        layout.addWidget(self.lock_label)
        self.frequency_label = QLabel("Frequency: -")
        self.frequency_label.setAlignment(Qt.AlignCenter)
        self.frequency_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.frequency_label)
        self.track_plot = pg.PlotWidget(title="Tracked frequency")
        self.track_plot.showGrid(x=True, y=True, alpha=0.25)
        self.track_plot.setLabel("bottom", "Relative time", units="s")
        self.track_plot.setLabel("left", "Frequency", units="Hz")
        self.track_curve = self.track_plot.plot(pen=pg.mkPen("#147d64", width=2))
        layout.addWidget(self.track_plot)
        note = QLabel(
            "현재 배포 펌웨어에서 hardware lock bit와 device timestamp는 확인되지 않았습니다. "
            "Hardware 화면의 상태는 데이터 최신성과 frequency rail만 이용한 추정이며, CSV 시간은 "
            "입력한 stream rate로 복원됩니다. Frequency와 amplitude는 보드에서 선택 전송하므로 동시에 측정되지 않습니다.")
        note.setWordWrap(True); note.setStyleSheet("color: #555;")
        layout.addWidget(note)
        return tab

    def _set_connected(self, connected):
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.sweep_button.setEnabled(connected)
        self.track_button.setEnabled(connected)
        self.track_stop_button.setEnabled(connected)

    def _network_profile(self):
        return NetworkProfile(
            board_host=self.host_edit.text().strip(), tcp_port=self.tcp_port.value(),
            multicast_group=self.group_edit.text().strip(), udp_port=self.udp_port.value(),
            interface_host=self.interface_edit.text().strip())

    def _parameters(self, *, tracking=False, streaming=False):
        return PLLUserParameters(
            frequency_hz=self.center.value(),
            minimum_frequency_hz=self.minimum.value(),
            maximum_frequency_hz=self.maximum.value(), resonator_q=self.q.value(),
            loop_bandwidth_hz=self.loop_bw.value(), loop_order=self.loop_order.value(),
            cic_rate=self.cic_rate.value(), cic_bit_shift=self.cic_shift.value(),
            pll_delay_turns=self.delay_turns.value(),
            drive_amplitude_fraction=self.drive_fraction.value(),
            feedback_gain=0.1,
            mode=ModeFields(run=False, enable_agc=True,
                            send_data_to_pc=streaming,
                            run_nco_at_fixed_frequency=not tracking,
                            dac_a_output=1,
                            signal_of_interest=(2 if tracking and
                                                self.stream_choice.currentText() == "PLL amplitude"
                                                else 0)))

    def connect_backend(self):
        try:
            if self.backend_combo.currentText() == "Simulator":
                controller = SimulatedFPGAController()
            else:
                if self.channel_spin.value() != 0:
                    raise ValueError("FPGA_Python 검증 프로파일은 channel A (0)만 지원합니다.")
                controller = FPGAController(
                    self._network_profile(), channel=self.channel_spin.value(),
                    channel_stride=self.stride_spin.value())
            banner = controller.connect()
            self.controller = controller
            self._set_connected(True)
            self.status.setText(f"연결됨: {banner.strip() or type(controller).__name__} · 출력 정지 상태")
        except Exception as exc:
            self.controller = None
            QMessageBox.critical(self, "연결 실패", str(exc))

    def disconnect_backend(self):
        if self.controller is None:
            return
        error = None
        try:
            if self.sweep_worker is not None:
                self.sweep_worker.stop()
                self.sweep_worker.wait(3000)
            self.stop_tracking()
            self.controller.stop_output()
        except Exception as exc:
            error = exc
        try:
            self.controller.disconnect()
        finally:
            self.controller = None
            self._set_connected(False)
            self.status.setText("연결 해제됨")
        if error:
            QMessageBox.warning(self, "출력 정지 확인 실패", str(error))

    def run_sweep(self):
        if self.controller is None or self.sweep_worker is not None:
            return
        if self.timer.isActive():
            QMessageBox.warning(self, "PLL 실행 중", "PLL Tracking을 먼저 정지하세요.")
            return
        try:
            frequencies = np.linspace(self.sweep_start.value(), self.sweep_stop.value(),
                                      self.sweep_points.value())
            parameters = self._parameters()
            parameters = replace(parameters, frequency_hz=float(frequencies[0]),
                                 minimum_frequency_hz=float(min(frequencies)),
                                 maximum_frequency_hz=float(max(frequencies)))
            self.last_sweep = None; self.last_fit = None
            self.last_fit_quality = None
            self.use_fit_button.setEnabled(False)
            self.amplitude_curve.setData([], []); self.phase_curve.setData([], [])
            self._sweep_live = []
            worker = SweepWorker(self.controller, parameters, frequencies,
                                 self.sweep_settle.value(), self.sweep_samples.value())
            self.sweep_worker = worker
            worker.point.connect(self._on_sweep_point)
            worker.completed.connect(self._on_sweep_completed)
            worker.failed.connect(self._on_sweep_failed)
            worker.start()
            self.sweep_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.status.setText("Sweep 실행 중")
        except Exception as exc:
            QMessageBox.critical(self, "Sweep 시작 실패", str(exc))

    def _on_sweep_point(self, frequency, amplitude, phase, x, y):
        self._sweep_live.append((frequency, amplitude, phase, x, y))
        a = np.asarray(self._sweep_live)
        self.amplitude_curve.setData(a[:, 0], a[:, 1])
        self.phase_curve.setData(a[:, 0], a[:, 2])

    def _on_sweep_completed(self, result):
        self.sweep_worker = None
        self.sweep_button.setEnabled(True)
        self.track_button.setEnabled(True)
        if not len(result["freqs"]):
            self.status.setText("Sweep 중지됨")
            return
        self.last_sweep = result
        try:
            peak = int(np.nanargmax(result["R"]))
            f0 = float(result["freqs"][peak])
            span = float(np.ptp(result["freqs"]))
            fit = fit_complex_resonance(
                result["freqs"], result["X"], result["Y"],
                result["X_sem"], result["Y_sem"], f0_guess=f0,
                f0_bounds=(max(result["freqs"][0], f0-span/4),
                           min(result["freqs"][-1], f0+span/4)))
            quality = assess_fit_quality(result["freqs"], result["X"], result["Y"], fit)
            self.last_fit = fit
            self.last_fit_quality = quality
            self.use_fit_button.setEnabled(quality.accepted)
            reasons = " / ".join(quality.reasons) if quality.reasons else "품질 기준 통과"
            self.fit_label.setText(
                f"f0={fit.f0_hz:.6f} Hz · Q={fit.q:.3f} · linewidth={fit.linewidth_hz:.6g} Hz · "
                f"phase={fit.phase_deg:.3f}° · R²={quality.complex_r_squared:.5f} · {reasons}")
            self.status.setText("Sweep와 complex fit 완료")
        except Exception as exc:
            self.last_fit = None
            self.last_fit_quality = None
            self.use_fit_button.setEnabled(False)
            self.fit_label.setText(f"Sweep 완료, complex fit 실패: {exc}")
            self.status.setText("Sweep 완료, fit 확인 필요")

    def _on_sweep_failed(self, message):
        self.sweep_worker = None
        self.last_fit = None
        self.last_fit_quality = None
        self.use_fit_button.setEnabled(False)
        self.sweep_button.setEnabled(True)
        self.track_button.setEnabled(True)
        self.status.setText("Sweep 실패")
        QMessageBox.critical(self, "Sweep 실패", message)

    def stop_sweep(self):
        if self.sweep_worker is not None:
            self.sweep_worker.stop()

    def use_fit(self):
        if self.last_fit is None or self.last_fit_quality is None or not self.last_fit_quality.accepted:
            return
        self.center.setValue(self.last_fit.f0_hz)
        self.q.setValue(self.last_fit.q)
        half_range = max(5*self.last_fit.linewidth_hz, self.loop_bw.value()*5)
        self.minimum.setValue(max(0, self.last_fit.f0_hz-half_range))
        self.maximum.setValue(min(12_500_000, self.last_fit.f0_hz+half_range))
        self.tabs.setCurrentIndex(2)

    def export_sweep(self):
        if self.last_sweep is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Sweep CSV 저장", "fpga_sweep.csv", "CSV (*.csv)")
        if not path: return
        names = list(self.last_sweep)
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle); writer.writerow(names)
            writer.writerows(zip(*[self.last_sweep[name] for name in names]))

    def start_tracking(self):
        if self.controller is None:
            return
        if self.sweep_worker is not None:
            QMessageBox.warning(self, "Sweep 실행 중", "Sweep을 먼저 중지하고 완료될 때까지 기다리세요.")
            return
        try:
            self.stop_tracking()
            amplitude_mode = self.stream_choice.currentText() == "PLL amplitude"
            if amplitude_mode and not isinstance(self.controller, FPGAController):
                raise ValueError("Amplitude stream 선택은 실장비에서만 지원합니다.")
            parameters = self._parameters(tracking=True, streaming=True)
            self.tracking_rows = []; self.tracking_t = []; self.tracking_f = []
            self.tracking_sample_index = 0; self.tracking_first_monotonic = None
            self.tracking_last_packet_monotonic = None
            self.tracking_ignore_before_monotonic = None
            self.stream_error_reported = False
            if isinstance(self.controller, FPGAController):
                self.controller.start_stream(
                    signal="unsigned" if amplitude_mode else "frequency")
            self.controller.apply_parameters(parameters)
            self.controller.start_tracking(stream=True)
            if isinstance(self.controller, FPGAController):
                self.controller.restart_all_plls()
                self.tracking_ignore_before_monotonic = time.monotonic() + 0.5
            self.stream_choice.setEnabled(False)
            self.track_plot.setLabel("left", "Amplitude counts" if amplitude_mode
                                     else "Frequency", units="" if amplitude_mode else "Hz")
            self.timer.start()
            self.lock_label.setText("상태: tracking 데이터 대기")
            self.status.setText("PLL tracking 실행 중")
        except Exception as exc:
            try: self.controller.stop_output()
            except Exception: pass
            QMessageBox.critical(self, "PLL 시작 실패", str(exc))

    def _poll_tracking(self):
        if self.controller is None:
            return
        stale = False
        if isinstance(self.controller, SimulatedFPGAController):
            t, f, z = self.controller.sample_tracking(10, self.stream_rate.value())
            for ti, fi, zi in zip(t, f, z):
                self.tracking_rows.append(
                    (ti, fi, abs(zi), np.angle(zi, deg=True), zi.real, zi.imag,
                     0, "simulated", 0, 0, 1))
            self.tracking_t.extend(t.tolist()); self.tracking_f.extend(f.tolist())
            self.tracking_last_packet_monotonic = time.monotonic()
        else:
            rate = self.stream_rate.value()
            amplitude_mode = self.stream_choice.currentText() == "PLL amplitude"
            for batch in self.controller.drain_stream(200):
                if (self.tracking_ignore_before_monotonic is not None and
                        batch.host_monotonic_s < self.tracking_ignore_before_monotonic):
                    continue
                if batch.sequence_classification in ("duplicate", "reordered"):
                    continue
                if self.tracking_first_monotonic is None:
                    self.tracking_first_monotonic = batch.host_monotonic_s
                self.tracking_last_packet_monotonic = time.monotonic()
                if batch.missing_packets:
                    self.tracking_sample_index += batch.missing_packets * len(batch.values)
                for offset, value in enumerate(batch.values):
                    t = self.tracking_sample_index/rate
                    valid = int(np.isfinite(value))
                    self.tracking_rows.append(
                        (t, np.nan if amplitude_mode else value,
                         np.nan, np.nan, np.nan, np.nan,
                         batch.packet_index,
                         batch.sequence_classification if offset == 0 else "payload",
                         batch.missing_packets if offset == 0 else 0,
                         batch.segment_id, valid,
                         value if amplitude_mode else np.nan))
                    self.tracking_t.append(t); self.tracking_f.append(float(value))
                    self.tracking_sample_index += 1
            receiver = self.controller.receiver
            if (receiver is not None and receiver.error is not None
                    and not self.stream_error_reported):
                self.stream_error_reported = True
                self.timer.stop()
                try:
                    self.controller.stop_output()
                except Exception:
                    pass
                QMessageBox.critical(self, "UDP 수신 중단", str(receiver.error))
                return
            if (self.tracking_last_packet_monotonic is not None
                    and time.monotonic() - self.tracking_last_packet_monotonic > 1.0):
                stale = True
                self.lock_label.setText("데이터 stale · tracking 상태 확인 불가")
                self.lock_label.setStyleSheet(
                    "font-size: 24px; font-weight: 800; padding: 10px; color: #c62828;")
        if self.tracking_f:
            keep = max(0, len(self.tracking_f)-100000)
            self.track_curve.setData(self.tracking_t[keep:], self.tracking_f[keep:])
            latest = self.tracking_f[-1]
            amplitude_mode = (isinstance(self.controller, FPGAController) and
                              self.stream_choice.currentText() == "PLL amplitude")
            self.frequency_label.setText(
                f"Amplitude: {latest:,.0f} counts" if amplitude_mode else
                f"Frequency: {latest:,.6f} Hz")
            within = self.minimum.value() <= latest <= self.maximum.value()
            simulated = isinstance(self.controller, SimulatedFPGAController)
            if not stale:
                self.lock_label.setText("Amplitude stream · lock 미확인" if amplitude_mode else
                                        "LOCKED (simulated)" if simulated and within else
                                        "Tracking 추정 · hardware lock 미확인" if within else
                                        "Frequency rail 이탈")
            self.lock_label.setStyleSheet(
                    "font-size: 24px; font-weight: 800; padding: 10px; color: "
                    + ("#555;" if amplitude_mode else
                       "#147d64;" if within else "#c62828;"))

    def stop_tracking(self):
        self.timer.stop()
        self.stream_choice.setEnabled(True)
        if self.controller is not None:
            try: self.controller.stop_output()
            finally:
                if isinstance(self.controller, FPGAController):
                    self.controller.stop_stream()
        self.lock_label.setText("상태: 정지")

    def save_tracking(self):
        if not self.tracking_rows: return
        path, _ = QFileDialog.getSaveFileName(self, "Tracking CSV 저장", "fpga_tracking.csv", "CSV (*.csv)")
        if not path: return
        write_tracking_csv(
            path, self.tracking_rows,
            backend=self.controller.capabilities.backend if self.controller else "disconnected",
            time_source=("simulator" if isinstance(self.controller, SimulatedFPGAController)
                         else "reconstructed_from_user_stream_rate"),
            stream_rate_hz=self.stream_rate.value(),
            local_queue_dropped_batches=getattr(self.controller, "dropped_local_batches", 0),
            stream_signal=("amplitude_counts" if self.stream_choice.currentText() ==
                           "PLL amplitude" else "frequency_hz"),
        )

    def closeEvent(self, event):
        try:
            self.stop_sweep()
            if self.sweep_worker is not None:
                self.sweep_worker.wait(3000)
            self.disconnect_backend()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = FPGAWindow(); window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
