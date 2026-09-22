"""Noise/SNR view embedded in the particle review window."""
from datetime import datetime
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from particle_review_noise import save_noise_analysis


class NoiseSaveWorker(QtCore.QThread):
    done=QtCore.pyqtSignal(str)
    failed=QtCore.pyqtSignal(str)

    def __init__(self, folder, analysis, source, metadata):
        super().__init__(); self.args=(folder,analysis,source,metadata)

    def run(self):
        try:self.done.emit(str(save_noise_analysis(*self.args)))
        except Exception as exc:self.failed.emit(str(exc))


class NoisePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis=None; self.save_worker=None
        layout=QtWidgets.QVBoxLayout(self)
        self.info=QtWidgets.QLabel('왼쪽의 「노이즈 + SNR 계산」을 누르세요.'); self.info.setWordWrap(True); layout.addWidget(self.info)
        self.table=QtWidgets.QTableWidget(0,9)
        self.table.setHorizontalHeaderLabels(['시각(s)','Δf(Hz)','전 σ(Hz)','후 σ(Hz)','통합 σ(Hz)','10σ(Hz)','SNR','판정','안내'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.draw)
        layout.addWidget(self.table,1)
        self.plot=pg.GraphicsLayoutWidget(); layout.addWidget(self.plot,2)
        self.raw=self.plot.addPlot(row=0,col=0); self.raw.setLabel('left','Frequency',units='Hz')
        self.residual=self.plot.addPlot(row=1,col=0); self.residual.setLabel('left','Residual',units='Hz')
        self.residual.setXLink(self.raw); self.residual.setLabel('bottom','Time',units='s')
        for p in (self.raw,self.residual):
            p.showGrid(x=True,y=True,alpha=.2); p.getAxis('left').enableAutoSIPrefix(False)
            p.getAxis('left').setWidth(100)
        note=QtWidgets.QLabel('파랑=전 / 주황=후 · 검정 점선=drift · 회색=제외 구간\nSNR = |저장된 깊이| / 통합 σ. 이벤트 전체가 회색 구간 안에 들어가는지 확인하세요.')
        note.setWordWrap(True); layout.addWidget(note)
        self.save_btn=QtWidgets.QPushButton('노이즈·SNR CSV / JSON / 보고서 / 그래프 저장')
        self.save_btn.setEnabled(False); self.save_btn.clicked.connect(self.save); layout.addWidget(self.save_btn)

    def invalidate(self, *args):
        self.analysis=None; self.table.setRowCount(0); self.raw.clear(); self.residual.clear()
        self.save_btn.setEnabled(False)
        self.info.setText('후보 또는 설정이 바뀌었습니다. 「노이즈 + SNR 계산」으로 갱신하세요.')

    def set_result(self, analysis, data, source, metadata):
        self.analysis=analysis; self.data=data; self.source=source; self.metadata=metadata
        out=analysis['summary']; s=analysis['settings']
        source=s.get('signal_source','raw')
        self.info.setText(f"입력 {source} · {len(out)}개 후보 | 각 방향 {s['window_s']:g} s / 중심 ±{s['guard_s']:g} s 제외 / {s['degree']}차 drift\nSNR 평균 {out.snr_pooled.mean():.2f} · 중앙값 {out.snr_pooled.median():.2f} · 10σ 충족 {(out.meets_10sigma==True).sum()}개 / 유효 {out.snr_pooled.notna().sum()}개")
        self.table.setRowCount(len(out))
        for i,r in enumerate(out.itertuples()):
            verdict='계산 불가' if not np.isfinite(r.snr_pooled) else ('충족' if r.meets_10sigma else '미달')
            vals=[r.time_s,r.depth_hz,r.sigma_before_hz,r.sigma_after_hz,r.sigma_pooled_hz,r.threshold_10sigma_hz,r.snr_pooled,verdict,r.warning]
            for j,v in enumerate(vals):
                text=(f'{v:.4f}' if np.isfinite(v) else '—') if isinstance(v,(float,np.floating)) else str(v)
                self.table.setItem(i,j,QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents(); self.table.selectRow(0)
        self.save_btn.setEnabled(not(self.save_worker and self.save_worker.isRunning()))

    def draw(self):
        i=self.table.currentRow()
        if self.analysis is None or not 0<=i<len(self.analysis['summary']):return
        r=self.analysis['summary'].iloc[i]; s=self.analysis['settings']; c=r.time_s; g=s['guard_s']; w=s['window_s']
        for p in (self.raw,self.residual):
            p.clear(); p.addItem(pg.LinearRegionItem([c-g,c+g],movable=False,brush=(140,140,140,35)))
        d=self.data; part=d[(d.Time_s>=c-g-w)&(d.Time_s<=c+g+w)]
        self.raw.plot(part.Time_s.to_numpy(),part.Frequency_Hz.to_numpy(),pen='#aaaaaa')
        residuals=self.analysis['residuals']
        for label,color in [('before','#175aaa'),('after','#df7d17')]:
            p=residuals[(residuals.event_index==r.event_index)&(residuals.context==label)]
            self.raw.plot(p.time_s.to_numpy(),p.frequency_hz.to_numpy(),pen=color)
            self.raw.plot(p.time_s.to_numpy(),p.drift_hz.to_numpy(),pen=pg.mkPen('k',style=QtCore.Qt.DashLine))
            self.residual.plot(p.time_s.to_numpy(),p.residual_hz.to_numpy(),pen=color,symbol='o',symbolSize=3)
        self.raw.setXRange(c-g-w,c+g+w,padding=.02)
        for p in (self.raw,self.residual):p.enableAutoRange(axis='y')

    def save(self):
        if self.analysis is None:return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'노이즈·SNR 결과 저장 위치',str(Path(self.source).parent))
        if not parent:return
        folder=Path(parent)/('particle_noise_snr_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.save_worker=NoiseSaveWorker(folder,self.analysis,self.source,self.metadata)
        self.save_btn.setEnabled(False)
        self.save_worker.done.connect(lambda p:QtWidgets.QMessageBox.information(self,'저장 완료',f'노이즈·SNR 수치, 설정, 보고서, 그래프를 저장했습니다.\n{p}'))
        self.save_worker.failed.connect(lambda e:QtWidgets.QMessageBox.warning(self,'저장 실패',e))
        self.save_worker.finished.connect(lambda:self.save_btn.setEnabled(self.analysis is not None))
        self.save_worker.start()
