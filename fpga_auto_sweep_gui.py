"""Theory sweep controls and persistent multi-mode result browser."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
                             QHeaderView, QAbstractItemView)

from beam_modes import TubeBeam, calculate_modes
from fpga_auto_sweep import AutoSweepSettings, AutoModeSweepWorker, build_mode_plans, load_auto_session


THEORY_WIDGETS = ("theory_length_spin", "theory_outer_spin", "theory_inner_spin",
                  "theory_e_spin", "theory_density_spin", "theory_fluid_density_spin",
                  "theory_count_spin", "theory_show_check")


class AutoSweepMixin:
    def _build_auto_controls(self, parent):
        self.auto_worker = None
        self.auto_stages = []
        self.auto_records = []
        self.auto_directory = None
        self._auto_locked_widgets = []
        group = QGroupBox("이론 모드 자동 스윕 · Coarse → Fine")
        grid = QGridLayout(group)
        self.auto_fields = {}
        defaults = AutoSweepSettings()
        specifications = [
            ("first_mode", "시작 차수", 1, 30, 0),
            ("last_mode", "종료 차수", 1, 30, 0),
            ("coarse_percent", "Coarse ± (%)", 0.01, 100, 2),
            ("coarse_half_hz", "Coarse ± Hz (0=비율)", 0, 6249999, 2),
            ("coarse_step_hz", "Coarse 간격 (Hz)", 0.01, 100000, 3),
            ("fine_step_hz", "Fine 간격 (Hz)", 0.01, 10000, 3),
            ("fallback_half_hz", "선폭 미검출 시 ± (Hz)", 1, 100000, 2),
            ("linewidth_factor", "Fine 반폭 / 선폭", 1, 10, 1),
            ("coarse_settle_s", "Coarse 정착 (s)", 0.05, 60, 3),
            ("fine_settle_s", "Fine 정착 (s)", 0.05, 60, 3),
            ("coarse_samples", "Coarse 평균 샘플", 1, 100000, 0),
            ("fine_samples", "Fine 평균 샘플", 1, 100000, 0),
        ]
        for row, (key, label, low, high, decimals) in enumerate(specifications):
            field = QDoubleSpinBox() if decimals else QSpinBox()
            field.setRange(low, high)
            if decimals:
                field.setDecimals(decimals)
            field.setValue(getattr(defaults, key))
            self.auto_fields[key] = field
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(field, row, 1)
        row = len(specifications)
        self.auto_preview_btn = QPushButton("범위 미리보기 / 전체 결과")
        self.auto_preview_btn.clicked.connect(self.preview_auto_sweep)
        grid.addWidget(self.auto_preview_btn, row, 0, 1, 2)
        self.auto_run_btn = QPushButton("자동 스윕 시작 · 저장 폴더 선택")
        self.auto_run_btn.clicked.connect(self.start_auto_sweep)
        grid.addWidget(self.auto_run_btn, row+1, 0, 1, 2)
        self.auto_stop_btn = QPushButton("자동 스윕 중지")
        self.auto_stop_btn.setEnabled(False)
        self.auto_stop_btn.clicked.connect(self.stop_auto_sweep)
        grid.addWidget(self.auto_stop_btn, row+2, 0, 1, 2)
        self.auto_status = QLabel("준비 · 현재 이론 입력으로 모드별 탐색")
        self.auto_status.setWordWrap(True)
        grid.addWidget(self.auto_status, row+3, 0, 1, 2)
        parent.layout().addWidget(group, parent.layout().rowCount(), 0, 1, 2)

    def _build_auto_results(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        buttons = QHBoxLayout()
        self.auto_view = QComboBox()
        self.auto_view.addItem("전체 모드", None)
        self.auto_view.currentIndexChanged.connect(self._draw_auto_results)
        buttons.addWidget(QLabel("보기"))
        buttons.addWidget(self.auto_view)
        self.auto_fine_zoom_btn = QPushButton("선택 모드 Fine 확대")
        self.auto_fine_zoom_btn.clicked.connect(self._zoom_auto_fine)
        buttons.addWidget(self.auto_fine_zoom_btn)
        self.auto_open_btn = QPushButton("저장된 자동 스윕 열기")
        self.auto_open_btn.clicked.connect(self.open_auto_session)
        buttons.addWidget(self.auto_open_btn)
        layout.addLayout(buttons)
        self.auto_path_label = QLabel("실행마다 별도 폴더에 coarse/fine CSV, 설정, 요약 저장")
        self.auto_path_label.setWordWrap(True)
        layout.addWidget(self.auto_path_label)
        self.auto_table = QTableWidget(0, 7)
        self.auto_table.setHorizontalHeaderLabels(["모드", "이론 (Hz)", "Coarse 범위 / 상태", "피크 (Hz)", "Fit f0 (Hz)", "Q", "비고"])
        self.auto_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.auto_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.auto_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.auto_table.horizontalHeader().setStretchLastSection(True)
        self.auto_table.setMaximumHeight(310)
        self.auto_table.cellClicked.connect(self._select_auto_row)
        layout.addWidget(self.auto_table)
        self.auto_amp_plot = pg.PlotWidget(title="전체 모드 / Coarse · Fine")
        self.auto_amp_plot.setLabel("left", "Amplitude", units="counts")
        self.auto_amp_plot.setLabel("bottom", "Frequency", units="Hz")
        self.auto_amp_plot.addLegend()
        self.auto_phase_plot = pg.PlotWidget(title="위상")
        self.auto_phase_plot.setLabel("left", "Phase", units="deg")
        self.auto_phase_plot.setLabel("bottom", "Frequency", units="Hz")
        self.auto_phase_plot.setXLink(self.auto_amp_plot)
        layout.addWidget(self.auto_amp_plot, 3)
        layout.addWidget(self.auto_phase_plot, 2)
        self.auto_tab_index = self.tabs.addTab(panel, "⑥ 모드 자동 스윕")
        self._auto_live = []
        self._auto_live_curve = None
        self._auto_live_phase = None
        self._auto_current_stage = None

    def _auto_settings_widgets(self):
        return {**{"auto_"+key: widget for key, widget in self.auto_fields.items()},
                **{name: getattr(self, name) for name in THEORY_WIDGETS},
                "drive_v_spin": self.drive_v_spin}

    def _auto_configuration(self):
        settings = AutoSweepSettings(**{key: field.value() for key, field in self.auto_fields.items()})
        settings.validate()
        beam = TubeBeam(self.theory_length_spin.value()*1e-3,
                        self.theory_outer_spin.value()*1e-3, self.theory_inner_spin.value()*1e-3,
                        self.theory_e_spin.value()*1e9, self.theory_density_spin.value(),
                        self.theory_fluid_density_spin.value())
        # Calculate the next neighbor even when it is outside the selected run.
        modes = calculate_modes(beam, max(settings.last_mode+1, self.theory_count_spin.value()))
        self.theory_modes = modes
        self.theory_mode_list.clear()
        for mode in modes:
            self.theory_mode_list.addItem(f"Mode {mode.index}: {mode.frequency_hz:,.3f} Hz")
        self.update_theory_lines()
        return settings, beam, modes, build_mode_plans(modes, settings)

    def preview_auto_sweep(self):
        try:
            _, _, _, plans = self._auto_configuration()
            if not self.auto_stages:
                self.auto_records = [dict(mode=p["mode"], theory_hz=p["theory_hz"],
                    status=f"{p['low']:.1f}~{p['high']:.1f} ({p['points']}점)", notes=[]) for p in plans]
                self._update_auto_table()
            self.tabs.setCurrentIndex(self.auto_tab_index)
            points = sum(p["points"] for p in plans)
            self.auto_status.setText(f"{plans[0]['mode']}~{plans[-1]['mode']}차 · 최초 coarse {points:,}점 · fine 범위는 검출 후 결정")
        except Exception as exc:
            QMessageBox.warning(self, "자동 스윕 설정", str(exc))

    def _auto_busy(self):
        worker = getattr(self, "auto_worker", None)
        return worker is not None and worker.isRunning()

    def start_auto_sweep(self, checked=False, *, directory=None):
        if self._auto_busy():
            return
        try:
            if self.fpga is None:
                raise RuntimeError("FPGA 연결이 필요합니다.")
            for name in ("sweep_worker", "fpga_error_worker", "fpga_delay_worker", "fpga_response_worker"):
                worker = getattr(self, name, None)
                if worker and worker.isRunning():
                    raise RuntimeError("진행 중인 FPGA 측정이 끝난 뒤 자동 스윕을 시작하세요.")
            settings, beam, modes, plans = self._auto_configuration()
            if directory is None:
                parent = QFileDialog.getExistingDirectory(self, "자동 스윕 저장 위치", str(Path.cwd()))
                if not parent:
                    return
                directory = Path(parent) / datetime.now().strftime("auto_sweep_%Y%m%d_%H%M%S_%f")
            if Path(directory).exists():
                raise ValueError("실행 폴더는 새 이름이어야 합니다. 기존 측정은 덮어쓰지 않습니다.")
            if self.plotter_worker and self.plotter_worker.isRunning():
                self.stop_plotter()
                if self.plotter_worker and self.plotter_worker.isRunning():
                    raise RuntimeError("Plotter 종료 대기 중입니다.")
            if self.pll_enable_btn.isChecked():
                self.pll_enable_btn.setChecked(False)
            parameters = self._fpga_parameters(fixed=True, minimum=plans[0]["low"], maximum=plans[0]["high"])
            worker = AutoModeSweepWorker(self.fpga, parameters, modes, settings, directory, {"beam": asdict(beam)})
            self._save_last_settings()
            self.auto_stages = []
            self.auto_records = [dict(mode=p["mode"], theory_hz=p["theory_hz"], status="pending", notes=[]) for p in plans]
            self.auto_directory = Path(directory)
            self.auto_path_label.setText(str(directory))
            self.auto_view.blockSignals(True)
            self.auto_view.clear()
            self.auto_view.addItem("전체 모드", None)
            for p in plans:
                self.auto_view.addItem(f"{p['mode']}차 상세", p["mode"])
            self.auto_view.blockSignals(False)
            self._update_auto_table()
            self._draw_auto_results()
            self.auto_worker = worker
            # Shared sweep reference lets emergency stop and shutdown own this
            # worker too; GUI uses separate handlers for its structured signals.
            self.sweep_worker = worker
            self._set_auto_drive_indicator(True)
            worker.stage_started.connect(self._auto_stage_started)
            worker.point.connect(self._auto_point)
            worker.stage_done.connect(self._auto_stage_done)
            worker.mode_done.connect(self._auto_mode_done)
            worker.finished.connect(self._auto_finished)
            self._lock_auto_controls(True)
            self.tabs.setCurrentIndex(self.auto_tab_index)
            worker.start()
        except Exception as exc:
            QMessageBox.critical(self, "자동 스윕 시작 실패", str(exc))

    def _lock_auto_controls(self, locked):
        if locked:
            self._auto_locked_widgets = []
            for widget in self.findChildren(QWidget):
                if not isinstance(widget, (QPushButton, QAbstractSpinBox, QComboBox, QCheckBox, QLineEdit)):
                    continue
                if widget in (self.auto_stop_btn, self.stop_sweep_btn, self.auto_view, self.auto_fine_zoom_btn):
                    continue
                if isinstance(widget, QPushButton) and "긴급 정지" in widget.text():
                    continue
                self._auto_locked_widgets.append((widget, widget.isEnabled()))
            # Snapshot children before disabling their parent spinboxes.
            for widget, _ in self._auto_locked_widgets:
                widget.setEnabled(False)
        else:
            for widget, enabled in self._auto_locked_widgets:
                widget.setEnabled(enabled)
            self._auto_locked_widgets = []
        self.auto_stop_btn.setEnabled(locked)
        self.stop_sweep_btn.setEnabled(locked)

    def stop_auto_sweep(self):
        if self.auto_worker:
            self.auto_worker.stop()
            self.auto_status.setText("중지 요청 · 수집 데이터 저장 및 출력 정리 중")

    def _set_auto_drive_indicator(self, enabled):
        # Display the worker's ownership without issuing competing register writes.
        for widget, label in ((self.output_on_btn, "On"), (self.amp_enable_btn, "Amp")):
            widget.blockSignals(True)
            widget.setChecked(enabled)
            widget.setText(label + (" ✓" if enabled else ""))
            widget.blockSignals(False)

    def _auto_stage_started(self, meta):
        self._set_auto_drive_indicator(True)
        for record in self.auto_records:
            if record["mode"] == meta["mode"]:
                record["status"] = meta["stage"] + " 측정 중"
        self._update_auto_table()
        self.auto_status.setText(f"{meta['mode']}차 · {meta['stage']} · {meta['total']:,}점")
        self._auto_current_stage = meta
        self._auto_live = []
        self.auto_view.setCurrentIndex(self.auto_view.findData(meta["mode"]))
        self._draw_auto_results()

    def _auto_point(self, point):
        self._auto_live.append(point["values"])
        if point["count"] % 5 == 0 or point["count"] == point["total"]:
            if self._auto_live_curve is not None and self.auto_view.currentData() in (None, point["mode"]):
                values = np.asarray(self._auto_live)
                self._auto_live_curve.setData(values[:, 0], values[:, 1])
                self._auto_live_phase.setData(values[:, 0], values[:, 2])
            self.auto_status.setText(f"{point['mode']}차 · {point['stage']} · {point['count']:,}/{point['total']:,}")

    def _auto_stage_done(self, stage):
        self._auto_current_stage = None
        self._set_auto_drive_indicator(False)
        self.auto_stages.append(stage)
        self._draw_auto_results()

    def _auto_mode_done(self, record):
        self.auto_records = [record if r["mode"] == record["mode"] else r for r in self.auto_records]
        self._update_auto_table()

    def _auto_finished(self):
        outcome = self.auto_worker.outcome
        self.auto_records = outcome["records"]
        self._set_auto_drive_indicator(False)
        self._lock_auto_controls(False)
        self._update_auto_table()
        self.auto_view.setCurrentIndex(0)
        self._draw_auto_results()
        accepted = sum(r.get("fit_accepted", False) for r in self.auto_records)
        self.auto_status.setText(f"{outcome['status']} · Fit 통과 {accepted}/{len(self.auto_records)} · {outcome['elapsed_s']:.1f}s")
        if outcome["error"]:
            self.auto_status.setText(self.auto_status.text() + " · " + outcome["error"])
        self.status_label.setText(self.auto_status.text())

    def _update_auto_table(self):
        self.auto_table.setRowCount(len(self.auto_records))
        for row, record in enumerate(self.auto_records):
            values = [record["mode"], f"{record['theory_hz']:.3f}", record["status"],
                      record.get("peak_hz", ""), record.get("f0_hz", ""), record.get("q", ""), "; ".join(record["notes"])]
            for column, value in enumerate(values):
                text = f"{value:.4f}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.auto_table.setItem(row, column, item)

    def _select_auto_row(self, row, column):
        del column
        if 0 <= row < len(self.auto_records):
            self.auto_view.setCurrentIndex(self.auto_view.findData(self.auto_records[row]["mode"]))

    def _zoom_auto_fine(self):
        selected = self.auto_view.currentData()
        if selected is None:
            self.auto_status.setText("확대할 모드를 목록 또는 결과 표에서 선택하세요.")
            return
        stages = [stage for stage in self.auto_stages if stage["mode"] == selected
                  and stage["stage"].startswith("fine") and len(stage["data"].get("freqs", []))]
        if stages:
            f = stages[-1]["data"]["freqs"]
            self.auto_amp_plot.setXRange(float(f[0]), float(f[-1]), padding=.05)

    def _draw_auto_results(self):
        self.auto_amp_plot.clear()
        self.auto_phase_plot.clear()
        self._auto_live_curve = self._auto_live_phase = None
        selected = self.auto_view.currentData()
        for stage in self.auto_stages:
            if selected is not None and stage["mode"] != selected:
                continue
            data = stage["data"]
            if not data or not len(data["freqs"]):
                continue
            palette = ("#0072b2", "#d55e00", "#009e73", "#7b3294",
                       "#a6611a", "#c51b7d", "#4d4d4d", "#008b8b")
            color = palette[(stage["mode"]-1) % len(palette)]
            fine = stage["stage"].startswith("fine")
            pen = pg.mkPen(color, width=2 if fine else 1)
            name = f"M{stage['mode']} {stage['stage']}"
            self.auto_amp_plot.plot(data["freqs"], data["R"], pen=pen, name=name)
            self.auto_phase_plot.plot(data["freqs"], data["theta"], pen=pen)
        for record in self.auto_records:
            if selected is not None and record["mode"] != selected:
                continue
            theory = record["theory_hz"]
            self.auto_amp_plot.addItem(pg.InfiniteLine(theory, pen=pg.mkPen((140, 140, 140)), label=f"M{record['mode']} theory"))
            fit_path = self.auto_directory / f"mode_{record['mode']:02d}_fit.npz" if self.auto_directory else None
            if fit_path and fit_path.exists():
                with np.load(fit_path, allow_pickle=False) as fit:
                    self.auto_amp_plot.plot(fit["freqs"], fit["R"], pen=pg.mkPen("#222222", width=1), name=f"M{record['mode']} fit")
                    self.auto_phase_plot.plot(fit["freqs"], fit["theta"], pen=pg.mkPen("#222222", width=1))
        self.auto_amp_plot.enableAutoRange()
        self.auto_phase_plot.enableAutoRange()
        live = getattr(self, "_auto_current_stage", None)
        if live and (selected is None or selected == live["mode"]):
            values = np.asarray(self._auto_live)
            self._auto_live_curve = self.auto_amp_plot.plot(pen=pg.mkPen("#e66101", width=2), name="현재 측정")
            self._auto_live_phase = self.auto_phase_plot.plot(pen=pg.mkPen("#e66101", width=2))
            if len(values):
                self._auto_live_curve.setData(values[:, 0], values[:, 1])
                self._auto_live_phase.setData(values[:, 0], values[:, 2])

    def open_auto_session(self, checked=False, *, directory=None):
        if self._auto_busy():
            return
        directory = directory or QFileDialog.getExistingDirectory(self, "자동 스윕 실행 폴더 열기")
        if not directory:
            return
        try:
            session, stages = load_auto_session(directory)
            settings_file = Path(directory) / "settings.json"
            version = (json.loads(settings_file.read_text(encoding="utf-8"))
                       .get("peak_selection_version", 1))
            self.auto_directory = Path(directory)
            self.auto_records, self.auto_stages = session["records"], stages
            if version < 2:
                replay_file = Path(directory) / "selection_replay.json"
                if replay_file.exists():
                    proposals = json.loads(replay_file.read_text(encoding="utf-8"))
                    by_mode = {item["mode"]: item for item in proposals}
                    for record in self.auto_records:
                        proposal = by_mode.get(record["mode"])
                        if proposal and proposal["revised_coarse_peak_hz"] is not None:
                            difference = abs(proposal["revised_coarse_peak_hz"] - (record.get("peak_hz") or 0))
                            if difference > 20:
                                record["notes"] = list(record["notes"]) + [
                                    f"새 coarse 후보 {proposal['revised_coarse_peak_hz']:.1f} Hz · 1 Hz 재측정 필요"]
            self.auto_view.blockSignals(True)
            self.auto_view.clear()
            self.auto_view.addItem("전체 모드", None)
            for record in self.auto_records:
                self.auto_view.addItem(f"{record['mode']}차 상세", record["mode"])
            self.auto_view.blockSignals(False)
            self.auto_path_label.setText(str(directory))
            self._update_auto_table()
            self._draw_auto_results()
            self.tabs.setCurrentIndex(self.auto_tab_index)
            self.auto_status.setText(
                f"저장 결과 · {session['status']}" +
                (" · 이전 피크 판정 결과: 선택된 f0/Q를 재확인하세요. 원본은 유지됩니다."
                 if version < 2 else ""))
        except Exception as exc:
            QMessageBox.warning(self, "자동 스윕 열기 실패", str(exc))
