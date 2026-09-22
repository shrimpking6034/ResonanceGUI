"""Standalone baseline noise GUI. No hardware modules or connections required."""
from pathlib import Path
import sys

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from baseline_offline import analyze_noise, load_noise_csv, save_noise_result


class Worker(QtCore.QThread):
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, function, *args):
        super().__init__()
        self.function, self.args = function, args

    def run(self):
        try:
            self.completed.emit(self.function(*self.args))
        except Exception as exc:
            self.failed.emit(str(exc))


class BaselineOfflineWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.source = self.result = self.worker = None
        self.setWindowTitle("Baseline 노이즈 분석 — Offline")
        self.resize(1200, 850)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self.controls = QtWidgets.QWidget()
        controls = QtWidgets.QHBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.open_button = QtWidgets.QPushButton("노이즈 CSV 열기")
        self.open_button.clicked.connect(self.open_csv)
        controls.addWidget(self.open_button)
        self.full_range = QtWidgets.QCheckBox("전체 구간")
        self.full_range.setChecked(True)
        controls.addWidget(self.full_range)
        self.start_spin, self.end_spin = self.time_spin(), self.time_spin()
        for label, widget in (("시작 (s)", self.start_spin), ("끝 (s)", self.end_spin)):
            controls.addWidget(QtWidgets.QLabel(label))
            controls.addWidget(widget)
            widget.valueChanged.connect(self.invalidate)
        self.rolling_spin = QtWidgets.QDoubleSpinBox()
        self.rolling_spin.setRange(.001, 36000)
        self.rolling_spin.setDecimals(3)
        self.rolling_spin.setValue(10)
        self.rolling_spin.valueChanged.connect(self.invalidate)
        controls.addWidget(QtWidgets.QLabel("Rolling (s)"))
        controls.addWidget(self.rolling_spin)
        self.analyze_button = QtWidgets.QPushButton("분석 실행")
        self.analyze_button.clicked.connect(self.analyze)
        controls.addWidget(self.analyze_button)
        self.save_button = QtWidgets.QPushButton("결과 저장")
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        layout.addWidget(self.controls)
        self.full_range.toggled.connect(self.range_changed)
        self.path_label = QtWidgets.QLabel("저장된 CSV를 선택하세요. 장비 연결 없이 실행됩니다.")
        self.path_label.setWordWrap(True)
        self.path_label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.path_label)
        self.summary = QtWidgets.QLabel("필수 열: Time_s, Frequency_Hz · 진폭/위상 열은 선택 사항")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        raw_widget = pg.GraphicsLayoutWidget()
        self.raw_plot = raw_widget.addPlot(row=0, col=0)
        self.raw_plot.addLegend()
        self.raw_curve = self.curve(self.raw_plot, "Frequency", "Hz", "#3b82f6", "Raw")
        self.trend_curve = self.raw_plot.plot(pen=pg.mkPen("#f59e0b", width=2), name="Linear trend")
        self.residual_plot = raw_widget.addPlot(row=1, col=0)
        self.residual_curve = self.curve(self.residual_plot, "Residual", "Hz", "#14b8a6")
        self.residual_plot.setXLink(self.raw_plot)
        self.tabs.addTab(raw_widget, "주파수 / Drift 제거")
        noise_widget = pg.GraphicsLayoutWidget()
        self.rolling_plot = noise_widget.addPlot(row=0, col=0)
        self.rolling_curve = self.curve(self.rolling_plot, "Rolling detrended σ", "Hz", "#8b5cf6")
        self.allan_plot = noise_widget.addPlot(row=1, col=0)
        self.allan_plot.setLabel("left", "Allan deviation", units="Hz")
        self.allan_plot.getAxis("left").setWidth(110)
        self.allan_plot.getAxis("left").enableAutoSIPrefix(False)
        self.allan_plot.setLabel("bottom", "Averaging time", units="s")
        self.allan_plot.setLogMode(x=True, y=True)
        self.allan_plot.showGrid(x=True, y=True, alpha=.25)
        self.allan_curve = self.allan_plot.plot(pen="#f59e0b", symbol="o", symbolSize=6)
        self.tabs.addTab(noise_widget, "Rolling σ / Allan")
        aux_widget = pg.GraphicsLayoutWidget()
        self.amp_plot = aux_widget.addPlot(row=0, col=0)
        self.amp_curve = self.curve(self.amp_plot, "Amplitude", "V", "#3b82f6")
        self.phase_plot = aux_widget.addPlot(row=1, col=0)
        self.phase_curve = self.curve(self.phase_plot, "Phase", "deg", "#14b8a6")
        self.phase_plot.setXLink(self.amp_plot)
        self.tabs.addTab(aux_widget, "진폭 / 위상")
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, "상세 통계 / 메타데이터")
        note = QtWidgets.QLabel(
            "분석은 전체 해상도 원본 샘플을 사용합니다. 평활화·재샘플링 없음. "
            "Allan은 drift를 포함한 raw 주파수 기준이며, 시간 간격 누락의 영향을 받을 수 있습니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.range_changed()
        self.update_buttons()

    @staticmethod
    def time_spin():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        return spin

    @staticmethod
    def curve(plot, label, units, color, name=None):
        plot.setLabel("left", label, units=units)
        plot.getAxis("left").setWidth(110)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Time", units="s")
        plot.showGrid(x=True, y=True, alpha=.25)
        curve = plot.plot(pen=pg.mkPen(color, width=1), name=name)
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        return curve

    def range_changed(self):
        for spin in (self.start_spin, self.end_spin):
            spin.setEnabled(not self.full_range.isChecked())
        self.invalidate()

    def invalidate(self):
        if self.result is not None:
            self.result = None
            self.summary.setText("설정이 변경되었습니다. 분석 실행을 눌러 갱신하세요.")
            self.clear_plots()
        self.update_buttons()

    def clear_plots(self):
        for curve in (self.raw_curve, self.trend_curve, self.residual_curve,
                      self.rolling_curve, self.allan_curve, self.amp_curve, self.phase_curve):
            curve.setData([], [])
        self.details.clear()

    def update_buttons(self):
        self.analyze_button.setEnabled(self.source is not None)
        self.save_button.setEnabled(self.result is not None)

    def run_job(self, function, args, callback, message):
        if self.worker is not None:
            return
        self.controls.setEnabled(False)
        self.statusBar().showMessage(message)
        self.worker = Worker(function, *args)
        self.worker.completed.connect(callback)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.controls.setEnabled(True)
        self.update_buttons()

    def failed(self, message):
        self.statusBar().showMessage("작업 실패")
        if self.result is None:
            self.summary.setText("작업에 실패했습니다. 입력 파일과 분석 구간을 확인하세요.")
        QtWidgets.QMessageBox.warning(self, "Baseline 분석", message)

    def open_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "노이즈 CSV 선택", str(Path(__file__).parent / "Noise Data"), "CSV (*.csv)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        self.run_job(load_noise_csv, (path,), self.loaded, "CSV 불러오는 중…")

    def loaded(self, source):
        self.source, self.result = source, None
        self.clear_plots()
        t = source["columns"]["Time_s"]
        self.start_spin.setValue(float(t[0]))
        self.end_spin.setValue(float(t[-1]))
        self.full_range.setChecked(True)
        self.path_label.setText(source["path"])
        self.raw_curve.setData(t, source["columns"]["Frequency_Hz"])
        self.summary.setText(f"{len(t):,}개 샘플 · {t[0]:.6f} ~ {t[-1]:.6f} s · 분석 실행을 누르세요.")
        self.statusBar().showMessage("CSV 불러오기 완료")

    def analyze(self):
        if self.source is None:
            return
        self.result = None
        self.clear_plots()
        self.summary.setText("Baseline 분석 중…")
        start = None if self.full_range.isChecked() else self.start_spin.value()
        end = None if self.full_range.isChecked() else self.end_spin.value()
        self.run_job(analyze_noise, (self.source, start, end, self.rolling_spin.value()),
                     self.analyzed, "Baseline 분석 중…")

    def analyzed(self, result):
        self.result = result
        g, quality, cols = result["global"], result["timing"], result["columns"]
        t = cols["Time_s"]
        self.raw_curve.setData(t, cols["Frequency_Hz"])
        self.trend_curve.setData(t, g["trend_hz"])
        self.residual_curve.setData(t, g["residual_hz"])
        self.rolling_curve.setData(t, result["rolling"]["metrics"]["detrended_std_hz"])
        positive = [r for r in result["allan"] if r["allan_deviation_hz"] > 0]
        self.allan_curve.setData([r["tau_s"] for r in positive], [r["allan_deviation_hz"] for r in positive])
        for key, curve in (("Amplitude_R_V", self.amp_curve), ("Phase_deg", self.phase_curve)):
            if key in cols:
                curve.setData(t, cols[key])
        self.summary.setText(
            f"N={len(t):,} · 평균 {g['mean_hz']:.8g} Hz · raw σ {g['raw_std_hz']:.6g} Hz · "
            f"detrended σ {g['detrended_std_hz']:.6g} Hz · drift {g['drift_hz_per_min']:.6g} Hz/min\n"
            f"p-p {g['peak_to_peak_hz']:.6g} Hz · 3σ {g['threshold_3sigma_hz']:.6g} Hz · "
            f"5σ {g['threshold_5sigma_hz']:.6g} Hz · 유효 rate {quality['effective_rate_hz']:.6g} Hz · "
            f"긴 시간 간격 {quality['long_gap_count']}개")
        lines = [result["processing"], "", "[Timing]"]
        lines += [f"{key}: {value}" for key, value in quality.items()]
        lines += ["", "[Correlation]"]
        lines += [f"{key}: {value}" for key, value in result["correlation"].items()]
        lines += ["", "[Allan: raw Hz]"]
        lines += [str(row) for row in result["allan"]] or ["계산 가능한 Allan 구간이 없습니다."]
        lines += ["", "[Metadata]"]
        lines += [f"{key}: {value}" for key, value in result["metadata"].items()]
        self.details.setPlainText("\n".join(lines))
        self.statusBar().showMessage("분석 완료 · 결과 저장: Summary JSON / Derived CSV / Allan CSV·PNG")

    def save(self):
        if self.result is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "분석 결과를 저장할 상위 폴더")
        if folder:
            self.run_job(save_noise_result, (self.result, folder), self.saved, "결과 저장 중…")

    def saved(self, folder):
        self.statusBar().showMessage(f"저장 완료: {folder}")
        QtWidgets.QMessageBox.information(self, "저장 완료", str(folder))

    def closeEvent(self, event):
        if self.worker is not None:
            self.statusBar().showMessage("현재 작업이 끝난 뒤 창을 닫아 주세요.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = BaselineOfflineWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load_path(sys.argv[1])
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
