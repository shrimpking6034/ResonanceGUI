"""Qt result panel for MIT-style seventh-mode peak processing."""
from datetime import datetime
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from particle_review_mit_signal import save_mit_signal_analysis


class MitSignalPanel(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.analysis=None
        layout=QtWidgets.QVBoxLayout(self)
        self.info=QtWidgets.QLabel('Savitzky–Golay 비교 탭에서 「MIT 피크 분석」을 실행하세요.')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        self.table=QtWidgets.QTableWidget(0,10)
        self.table.setHorizontalHeaderLabels(['후보','시각(s)','antinode','방향','첫 Δf','마지막 Δf','대표 Δf','차이(%)','MIT SNR','상태 / 안내'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.draw); layout.addWidget(self.table,1)
        self.graph=pg.GraphicsLayoutWidget(); layout.addWidget(self.graph,2)
        self.raw=self.graph.addPlot(row=0,col=0); self.raw.setLabel('left','Frequency',units='Hz')
        self.residual=self.graph.addPlot(row=1,col=0); self.residual.setLabel('left','Baseline Δf',units='Hz'); self.residual.setLabel('bottom','Time',units='s'); self.residual.setXLink(self.raw)
        for p in (self.raw,self.residual):p.showGrid(x=True,y=True,alpha=.2);p.getAxis('left').enableAutoSIPrefix(False);p.getAxis('left').setWidth(105)
        note=QtWidgets.QLabel('회색=raw · 주황=SG · 검정 점선=3차 baseline · 점=7 antinode. MIT noise는 논문 비교를 위해 SG 전 raw peak 사이 구간에서 계산합니다.')
        note.setWordWrap(True);layout.addWidget(note)
        self.save_btn=QtWidgets.QPushButton('MIT 피크 분석 CSV / JSON / 보고서 저장')
        self.save_btn.setEnabled(False);self.save_btn.clicked.connect(self.save);layout.addWidget(self.save_btn)

    def invalidate(self):
        self.analysis=None;self.table.setRowCount(0);self.raw.clear();self.residual.clear()
        self.info.setText('후보 또는 SG/MIT 설정이 바뀌었습니다. 분석을 다시 실행하세요.');self.save_btn.setEnabled(False)

    def set_result(self,analysis,source=None,metadata=None):
        self.analysis=analysis;self.source=source;self.metadata=metadata or {};s=analysis['settings'];out=analysis['summary']
        sigma=out.mit_noise_sigma_hz.iloc[0] if len(out) else np.nan
        self.info.setText(f"SG order {s['sg_order']} / window {s['sg_window']} ({s['sg_window_ms']:.3f} ms) · prominence {s['prominence_sigma']:g}σ(현재 데이터 추정, 논문 약 5–10σ) · local fit ±{s['local_fit_fraction']:.2f} 간격(추정)\nMIT raw inter-peak σ {sigma:.5f} Hz · 유효 peak {(out.status=='ok').sum()} / {len(out)}")
        self.table.setRowCount(len(out))
        for i,r in enumerate(out.itertuples()):
            status=r.status if not r.warning else f'{r.status}: {r.warning}'
            vals=[r.event_index,r.time_s,r.antinode_count,r.direction,r.first_delta_hz,r.last_delta_hz,r.mean_delta_hz,r.relative_difference_percent,r.snr_mit,status]
            for j,v in enumerate(vals):
                text=f'{v:.5f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else ('—' if isinstance(v,(float,np.floating)) else str(v))
                self.table.setItem(i,j,QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        if len(out):self.table.selectRow(0)
        self.save_btn.setEnabled(source is not None)

    def draw(self):
        self.raw.clear();self.residual.clear()
        if self.analysis is None:return
        row=self.table.currentRow()
        if row<0:return
        event_index=int(self.analysis['summary'].iloc[row].event_index)
        tr=self.analysis['traces'];tr=tr[tr.event_index==event_index]
        if not len(tr):return
        self.raw.plot(tr.time_s.to_numpy(),tr.raw_hz.to_numpy(),pen='#aaaaaa')
        self.raw.plot(tr.time_s.to_numpy(),tr.sg_hz.to_numpy(),pen=pg.mkPen('#df7d17',width=2))
        self.raw.plot(tr.time_s.to_numpy(),tr.baseline_hz.to_numpy(),pen=pg.mkPen('k',style=QtCore.Qt.DashLine))
        a=self.analysis['antinodes'];a=a[a.event_index==event_index]
        if len(a):
            self.raw.plot(a.time_s.to_numpy(),a.frequency_hz.to_numpy(),pen=None,symbol='o',symbolBrush='#175aaa',symbolSize=7)
            self.residual.plot(a.time_s.to_numpy(),a.delta_hz.to_numpy(),pen=None,symbol='o',symbolBrush='#175aaa',symbolSize=7)
        self.residual.plot(tr.time_s.to_numpy(),tr.residual_hz.to_numpy(),pen='#238b57')
        self.residual.addLine(y=0,pen=pg.mkPen('#606060',style=QtCore.Qt.DashLine))
        self.raw.setXRange(float(tr.time_s.min()),float(tr.time_s.max()),padding=.02)
        self.raw.enableAutoRange(axis='y');self.residual.enableAutoRange(axis='y')

    def save(self):
        if self.analysis is None or self.source is None:return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'MIT 피크 분석 저장 위치',str(Path(self.source).parent))
        if not parent:return
        folder=Path(parent)/('mit_signal_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        try:
            save_mit_signal_analysis(folder,self.analysis,self.source,self.metadata)
            QtWidgets.QMessageBox.information(self,'저장 완료',f'MIT 피크, antinode, 처리 trace와 설정을 저장했습니다.\n{folder}')
        except Exception as exc:QtWidgets.QMessageBox.warning(self,'저장 실패',str(exc))
