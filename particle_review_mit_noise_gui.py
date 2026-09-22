"""Qt panel for the MIT-paper-compatible inter-peak noise analysis."""
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from particle_review_mit_noise import save_mit_noise_analysis


class MitNoiseSaveWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, folder, analysis, source, metadata):
        super().__init__()
        self.args = (folder, analysis, source, metadata)

    def run(self):
        try:
            self.done.emit(str(save_mit_noise_analysis(*self.args)))
        except Exception as exc:
            self.failed.emit(str(exc))


class MitNoisePanel(QtWidgets.QWidget):
    """Show inter-peak segments and their high-pass residuals."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis = None
        self.save_worker = None
        layout = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel('왼쪽의 「MIT 논문 방식 비교 계산」을 누르세요.')
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['구간', '시작(s)', '끝(s)', '길이(s)', '표본 수', 'Fs(Hz)', 'MIT σ(Hz)', '상태 / 안내'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.draw)
        layout.addWidget(self.table, 1)

        self.plot = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot, 2)
        self.raw = self.plot.addPlot(row=0, col=0)
        self.raw.setLabel('left', 'Frequency', units='Hz')
        self.residual = self.plot.addPlot(row=1, col=0)
        self.residual.setLabel('left', 'MIT HP residual', units='Hz')
        self.residual.setLabel('bottom', 'Time', units='s')
        self.residual.setXLink(self.raw)
        for item in (self.raw, self.residual):
            item.showGrid(x=True, y=True, alpha=.2)
            item.getAxis('left').enableAutoSIPrefix(False)
            item.getAxis('left').setWidth(110)
        note = QtWidgets.QLabel(
            '선택 구간만 표시합니다. 파랑=원시 주파수, 주황=1 Hz·2차 Butterworth high-pass 후 평균 제거 residual. '
            '이 탭은 기존 전후 선형 baseline SNR 분석을 바꾸지 않습니다.'
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.save_btn = QtWidgets.QPushButton('MIT 비교 CSV / JSON / 보고서 저장')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        layout.addWidget(self.save_btn)

    def invalidate(self, *args):
        self.analysis = None
        self.table.setRowCount(0)
        self.raw.clear(); self.residual.clear()
        self.save_btn.setEnabled(False)
        self.info.setText('후보 또는 MIT 설정이 바뀌었습니다. 「MIT 논문 방식 비교 계산」으로 갱신하세요.')

    def set_result(self, analysis, data, source, metadata):
        self.analysis, self.data, self.source, self.metadata = analysis, data, source, metadata
        s, q = analysis['settings'], analysis['summary']
        pooled = q['pooled_sigma_hz']; ratio = q['paper_ratio']
        pooled_text = f'{pooled:.5f} Hz' if np.isfinite(pooled) else '계산 불가'
        ratio_text = f'{ratio:.2f}×' if np.isfinite(ratio) else '—'
        self.info.setText(
            f"MIT 방식: 피크 사이 baseline / {s['cutoff_hz']:g} Hz / {s['order']}차 Butterworth HP / ddof={s['std_ddof']}\n"
            f"통합 σ {pooled_text} · Supplement Fig. S5 평균 {q['paper_reference_hz']:.4f} Hz 대비 {ratio_text} · "
            f"유효 {q['valid_segment_count']} / {q['total_segment_count']} 구간 · 총 {q['total_baseline_duration_s']:.2f} s"
        )
        segments = analysis['segments']
        self.table.setRowCount(len(segments))
        for i, row in enumerate(segments.itertuples()):
            status = row.status if not row.warning else f'{row.status}: {row.warning}'
            values = [row.segment_index, row.start_s, row.end_s, row.duration_s, row.sample_count,
                      row.sampling_rate_hz, row.sigma_mit_hz, status]
            for j, value in enumerate(values):
                if isinstance(value, (float, np.floating)):
                    text = f'{value:.5f}' if np.isfinite(value) else '—'
                else:
                    text = str(value)
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        if len(segments):
            self.table.selectRow(0)
        self.save_btn.setEnabled(not (self.save_worker and self.save_worker.isRunning()))

    def draw(self):
        index = self.table.currentRow()
        if self.analysis is None or not 0 <= index < len(self.analysis['segments']):
            return
        segment = self.analysis['segments'].iloc[index]
        self.raw.clear(); self.residual.clear()
        start, end = float(segment.start_s), float(segment.end_s)
        part = self.data[(self.data.Time_s >= start) & (self.data.Time_s <= end)]
        self.raw.plot(part.Time_s.to_numpy(), part.Frequency_Hz.to_numpy(), pen='#175aaa')
        residuals = self.analysis['residuals']
        r = residuals[residuals.segment_index == segment.segment_index]
        if len(r):
            self.residual.plot(r.time_s.to_numpy(), r.residual_hz.to_numpy(), pen='#df7d17', symbol='o', symbolSize=3)
            self.residual.addLine(y=0, pen=pg.mkPen('#606060', style=QtCore.Qt.DashLine))
        self.raw.setXRange(start, end, padding=.02)
        self.raw.enableAutoRange(axis='y'); self.residual.enableAutoRange(axis='y')

    def save(self):
        if self.analysis is None:
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, 'MIT 비교 결과 저장 위치', str(Path(self.source).parent))
        if not parent:
            return
        folder = Path(parent) / ('mit_noise_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.save_worker = MitNoiseSaveWorker(folder, self.analysis, self.source, self.metadata)
        self.save_btn.setEnabled(False)
        self.save_worker.done.connect(lambda path: QtWidgets.QMessageBox.information(self, '저장 완료', f'MIT 비교 수치와 설정을 저장했습니다.\n{path}'))
        self.save_worker.failed.connect(lambda error: QtWidgets.QMessageBox.warning(self, '저장 실패', error))
        self.save_worker.finished.connect(lambda: self.save_btn.setEnabled(self.analysis is not None))
        self.save_worker.start()
