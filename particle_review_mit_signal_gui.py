"""Qt result panel for MIT-style seventh-mode peak processing."""
from datetime import datetime
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from particle_review_mit_signal import save_mit_signal_analysis


class MitSignalPanel(QtWidgets.QWidget):
    manualAntinodesRequested = QtCore.pyqtSignal(int, object)
    automaticAntinodesRequested = QtCore.pyqtSignal(int)
    sensitivityRequested = QtCore.pyqtSignal()
    eventBoundsRequested = QtCore.pyqtSignal(int,float,float)

    def __init__(self,parent=None):
        super().__init__(parent); self.analysis=None
        layout=QtWidgets.QVBoxLayout(self)
        self.info=QtWidgets.QLabel('Savitzky–Golay 비교 탭에서 「MIT 피크 분석」을 실행하세요.')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        baseline_controls=QtWidgets.QHBoxLayout();layout.addLayout(baseline_controls)
        baseline_controls.addWidget(QtWidgets.QLabel('Baseline 차수'))
        self.baseline_degree=QtWidgets.QSpinBox();self.baseline_degree.setRange(1,3);self.baseline_degree.setValue(3);baseline_controls.addWidget(self.baseline_degree)
        baseline_controls.addWidget(QtWidgets.QLabel('추가 제외 guard (ms, SG 반폭 자동 추가)'))
        self.baseline_guard=QtWidgets.QDoubleSpinBox();self.baseline_guard.setRange(0,10000);self.baseline_guard.setDecimals(2);baseline_controls.addWidget(self.baseline_guard)
        self.sensitivity_btn=QtWidgets.QPushButton('선택 후보 SG / baseline 민감도 비교');self.sensitivity_btn.clicked.connect(self.sensitivityRequested);baseline_controls.addWidget(self.sensitivity_btn)
        baseline_controls.addStretch(1)
        interval=QtWidgets.QHBoxLayout();layout.addLayout(interval)
        interval.addWidget(QtWidgets.QLabel('이벤트 시작 (s)'))
        self.event_start=QtWidgets.QDoubleSpinBox();self.event_start.setDecimals(6);self.event_start.setSingleStep(.01);interval.addWidget(self.event_start)
        interval.addWidget(QtWidgets.QLabel('끝 (s)'))
        self.event_end=QtWidgets.QDoubleSpinBox();self.event_end.setDecimals(6);self.event_end.setSingleStep(.01);interval.addWidget(self.event_end)
        self.apply_bounds_btn=QtWidgets.QPushButton('구간 적용 후 재분석');self.apply_bounds_btn.clicked.connect(self.apply_event_bounds);interval.addWidget(self.apply_bounds_btn)
        self.interval_info=QtWidgets.QLabel('');interval.addWidget(self.interval_info,1)
        manual=QtWidgets.QHBoxLayout(); layout.addLayout(manual)
        manual.addWidget(QtWidgets.QLabel('수동 시각(s)'))
        self.manual_time=QtWidgets.QDoubleSpinBox();self.manual_time.setDecimals(6);self.manual_time.setSingleStep(.001)
        manual.addWidget(self.manual_time)
        self.add_btn=QtWidgets.QPushButton('이 위치에 추가');self.add_btn.clicked.connect(self.add_antinode);manual.addWidget(self.add_btn)
        self.move_btn=QtWidgets.QPushButton('가장 가까운 점 이동');self.move_btn.clicked.connect(self.move_antinode);manual.addWidget(self.move_btn)
        self.delete_btn=QtWidgets.QPushButton('가장 가까운 점 삭제');self.delete_btn.clicked.connect(self.delete_antinode);manual.addWidget(self.delete_btn)
        self.auto_btn=QtWidgets.QPushButton('선택 후보 자동값 복원');self.auto_btn.clicked.connect(self.restore_auto);manual.addWidget(self.auto_btn)
        manual.addStretch(1)
        self.table=QtWidgets.QTableWidget(0,10)
        self.table.setHorizontalHeaderLabels(['후보','시각(s)','antinode','방향','첫 Δf','마지막 Δf','대표 Δf','차이(%)','MIT SNR','상태 / 안내'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.draw); layout.addWidget(self.table,1)
        self.quality=QtWidgets.QLabel('');self.quality.setWordWrap(True);layout.addWidget(self.quality)
        # The two charts share the available width, giving the frequency plot a
        # near-square viewport in the normal review-window size.
        self.graph=pg.GraphicsLayoutWidget(); layout.addWidget(self.graph,3)
        self.raw=self.graph.addPlot(row=0,col=0); self.raw.setLabel('left','Frequency',units='Hz'); self.raw.setLabel('bottom','Time',units='s')
        self.residual=self.graph.addPlot(row=0,col=1); self.residual.setLabel('left','Baseline Δf',units='Hz'); self.residual.setLabel('bottom','Time',units='s'); self.residual.setXLink(self.raw)
        for p in (self.raw,self.residual):p.showGrid(x=True,y=True,alpha=.2);p.getAxis('left').enableAutoSIPrefix(False);p.getAxis('left').setWidth(80)
        self.cursor=pg.InfiniteLine(angle=90,movable=True,pen=pg.mkPen('#c33d3d',width=1))
        self.cursor.sigPositionChanged.connect(lambda:self.manual_time.setValue(float(self.cursor.value())))
        self.manual_time.valueChanged.connect(self.set_cursor_time)
        note=QtWidgets.QLabel('회색=raw · 주황=SG · 검정 점선=baseline · 파란 음영=baseline 사용 구간 · 자홍 곡선=외곽 2차 fit · 파란점=자동 / 보라점=수동 antinode. '
                              'Noise는 SG 전 raw에서 계산하며 후보 1개이면 전후 local noise로 구분합니다.')
        note.setWordWrap(True);layout.addWidget(note)
        self.save_btn=QtWidgets.QPushButton('MIT 피크 분석 CSV / JSON / 보고서 저장')
        self.save_btn.setEnabled(False);self.save_btn.clicked.connect(self.save);layout.addWidget(self.save_btn)
        self.baseline_degree.valueChanged.connect(self.invalidate)
        self.baseline_guard.valueChanged.connect(self.invalidate)

    def invalidate(self):
        self.analysis=None;self.table.setRowCount(0);self.raw.clear();self.residual.clear()
        self.quality.clear()
        self.info.setText('후보 또는 SG/MIT 설정이 바뀌었습니다. 분석을 다시 실행하세요.');self.save_btn.setEnabled(False)

    def set_result(self,analysis,source=None,metadata=None):
        self.analysis=analysis;self.source=source;self.metadata=metadata or {};s=analysis['settings'];out=analysis['summary']
        sigma=out.mit_noise_sigma_hz.iloc[0] if len(out) else np.nan
        manual_count=sum(1 for value in out.selection_source if value=='manual') if 'selection_source' in out else 0
        noise_source='전후 local noise (논문 inter-peak 방식의 확장)' if out.noise_source.iloc[0]=='local_before_after' else 'raw inter-peak noise'
        self.info.setText(f"선택 모드 {s['antinode_count']}차 → 기대 antinode {s['antinode_count']}개 · SG order {s['sg_order']} / window {s['sg_window']} ({s['sg_window_ms']:.3f} ms) · prominence {s['prominence_sigma']:g}σ\n{noise_source} σ {sigma:.5f} Hz · baseline {s['baseline_degree']}차 / 전체 이벤트 + {s['effective_baseline_guard_s']*1000:.2f} ms 제외 · 유효 {(out.status=='ok').sum()}/{len(out)} · 수동 {manual_count}")
        self.table.setRowCount(len(out))
        for i,r in enumerate(out.itertuples()):
            status=r.status if not r.warning else f'{r.status}: {r.warning}'
            if getattr(r,'selection_source','auto')=='manual':status='수동 · '+status
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
        summary=self.analysis['summary'].iloc[row]
        s=self.analysis['settings'];center=float(summary.time_s)
        start=float(summary.get('event_start_s',center));end=float(summary.get('event_end_s',center))
        valid=np.isfinite([start,end]).all() and end>start
        if not valid:start,end=center-.25,center+.25
        for spin,value in ((self.event_start,start),(self.event_end,end)):
            spin.setRange(s.get('data_start_s',0.),s.get('data_end_s',1e9));spin.setValue(value)
        self.interval_info.setText('저장된 구간' if valid else '구간 미지정 — 표시 값은 편집용 제안이며 아직 적용되지 않았습니다.')
        self.quality.setText(f"Baseline RMSE {summary.get('baseline_rmse_hz',np.nan):.5f} Hz · 검출용 SG σ {summary.get('detection_sigma_sg_hz',np.nan):.5f} Hz · 간격 CV {summary.get('antinode_spacing_cv',np.nan):.3f} · SG/간격 {summary.get('sg_spacing_ratio',np.nan):.2f}\n{summary.warning}")
        if summary.get('noise_warning',''):self.quality.setText(self.quality.text()+'\nNoise: '+str(summary.noise_warning))
        tr=self.analysis['traces'];tr=tr[tr.event_index==event_index]
        if not len(tr):return
        self.raw.plot(tr.time_s.to_numpy(),tr.raw_hz.to_numpy(),pen='#aaaaaa')
        self.raw.plot(tr.time_s.to_numpy(),tr.sg_hz.to_numpy(),pen=pg.mkPen('#df7d17',width=2))
        self.raw.plot(tr.time_s.to_numpy(),tr.baseline_hz.to_numpy(),pen=pg.mkPen('k',style=QtCore.Qt.DashLine))
        if 'baseline_used' in tr:
            times=tr.time_s.to_numpy();used=tr.baseline_used.to_numpy(dtype=bool)
            edges=np.diff(np.r_[False,used,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
            for left,right in zip(starts,ends):
                for plot in (self.raw,self.residual):
                    plot.addItem(pg.LinearRegionItem([times[left],times[right]],movable=False,brush=(40,115,190,30),pen=None))
        curves=self.analysis.get('fit_curves')
        if curves is not None and len(curves):
            for _,curve in curves[curves.event_index==event_index].groupby('antinode_index'):
                self.raw.plot(curve.time_s.to_numpy(),curve.frequency_hz.to_numpy(),pen=pg.mkPen('#bc329c',width=2))
                self.residual.plot(curve.time_s.to_numpy(),curve.delta_hz.to_numpy(),pen=pg.mkPen('#bc329c',width=2))
        a=self.analysis['antinodes'];a=a[a.event_index==event_index]
        if len(a):
            manual='selection_source' in a and (a.selection_source=='manual').any()
            brush='#8e44ad' if manual else '#175aaa'
            self.raw.plot(a.time_s.to_numpy(),a.frequency_hz.to_numpy(),pen=None,symbol='o',symbolBrush=brush,symbolSize=7)
            self.residual.plot(a.time_s.to_numpy(),a.delta_hz.to_numpy(),pen=None,symbol='o',symbolBrush=brush,symbolSize=7)
            if 'fit_rmse_hz' in a:
                fit=a[a.fit_rmse_hz.notna()]
                self.quality.setText(self.quality.text()+'\n외곽 2차 fit RMSE: '+', '.join(f'#{int(r.antinode_index)} {r.fit_rmse_hz:.5f} Hz' for r in fit.itertuples())+' (SG 잔차, 통계적 신뢰구간 아님)')
        self.residual.plot(tr.time_s.to_numpy(),tr.residual_hz.to_numpy(),pen='#238b57')
        self.residual.addLine(y=0,pen=pg.mkPen('#606060',style=QtCore.Qt.DashLine))
        self.manual_time.setRange(float(tr.time_s.min()),float(tr.time_s.max()))
        self.manual_time.blockSignals(True);self.manual_time.setValue(float(self.analysis['summary'].iloc[row].time_s));self.manual_time.blockSignals(False)
        self.cursor.setValue(self.manual_time.value());self.raw.addItem(self.cursor)
        self.raw.setXRange(float(tr.time_s.min()),float(tr.time_s.max()),padding=.02)
        self.raw.enableAutoRange(axis='y');self.residual.enableAutoRange(axis='y')

    def set_cursor_time(self,value):
        if self.cursor.value()!=value:self.cursor.setValue(value)

    def apply_event_bounds(self):
        if self.analysis is None or self.table.currentRow()<0:return
        start,end=self.event_start.value(),self.event_end.value()
        if end<=start:
            QtWidgets.QMessageBox.warning(self,'이벤트 구간','끝 시각은 시작 시각보다 커야 합니다.');return
        index=int(self.analysis['summary'].iloc[self.table.currentRow()].event_index)
        self.eventBoundsRequested.emit(index,start,end)

    def _selected(self):
        if self.analysis is None or self.table.currentRow()<0:return None,[]
        event_index=int(self.analysis['summary'].iloc[self.table.currentRow()].event_index)
        points=self.analysis['antinodes'];points=points[points.event_index==event_index]
        return event_index,sorted(float(value) for value in points.time_s)

    def add_antinode(self):
        event_index,times=self._selected()
        if event_index is None:return
        expected=int(self.analysis['settings']['antinode_count'])
        if len(times)>=expected:
            QtWidgets.QMessageBox.warning(self,'antinode 추가',f'이미 기대 개수 {expected}개입니다. 가장 가까운 점 이동을 사용하세요.');return
        times.append(float(self.manual_time.value()))
        self.manualAntinodesRequested.emit(event_index,sorted(times))

    def move_antinode(self):
        event_index,times=self._selected()
        if event_index is None or not times:return
        value=float(self.manual_time.value());nearest=int(np.argmin(abs(np.asarray(times)-value)))
        times[nearest]=value;self.manualAntinodesRequested.emit(event_index,sorted(times))

    def delete_antinode(self):
        event_index,times=self._selected()
        if event_index is None or not times:return
        value=float(self.manual_time.value());nearest=int(np.argmin(abs(np.asarray(times)-value)))
        times.pop(nearest);self.manualAntinodesRequested.emit(event_index,times)

    def restore_auto(self):
        event_index,_=self._selected()
        if event_index is not None:self.automaticAntinodesRequested.emit(event_index)

    def save(self):
        if self.analysis is None or self.source is None:return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'MIT 피크 분석 저장 위치',str(Path(self.source).parent))
        if not parent:return
        folder=Path(parent)/('mit_signal_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        try:
            save_mit_signal_analysis(folder,self.analysis,self.source,self.metadata)
            QtWidgets.QMessageBox.information(self,'저장 완료',f'MIT 피크, antinode, 처리 trace와 설정을 저장했습니다.\n{folder}')
        except Exception as exc:QtWidgets.QMessageBox.warning(self,'저장 실패',str(exc))
