"""Standalone offline particle-event GUI: python particle_review_gui.py [CSV]."""
import sys
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
from particle_review_core import (Settings, ParticlePropertySettings, load_raw, pressure_changes,
                                  detect, local_baseline, manual_event, settings_dict,
                                  batch_context_noise, estimate_particle_properties,
                                  load_sensitivity_profiles, save_sensitivity_profile)
from particle_review_noise import analyze_noise, recompute_event_depths
from particle_review_noise_gui import NoisePanel
from particle_review_mit_noise import analyze_mit_noise
from particle_review_mit_noise_gui import MitNoisePanel
from particle_review_savgol_gui import SavgolPanel
from particle_review_mit_signal import analyze_mit_signal
from particle_review_mit_signal_gui import MitSignalPanel
from particle_review_core import sample_window_stats


class Worker(QtCore.QThread):
    done = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, data, settings):
        super().__init__(); self.data=data; self.settings=settings

    def run(self):
        try: self.done.emit(*detect(self.data, self.settings))
        except Exception as exc: self.failed.emit(str(exc))


class FigureExportWorker(QtCore.QThread):
    progress=QtCore.pyqtSignal(int,int)
    done=QtCore.pyqtSignal(str,int)
    failed=QtCore.pyqtSignal(str)

    def __init__(self,data,events,parent,source,span,residual,settings,review_context):
        super().__init__(); self.args=(data,deepcopy(events),parent,source,span,residual,settings)
        self.review_context=deepcopy(review_context)

    def run(self):
        try:
            from particle_review_export import export_event_figures
            folder,count=export_event_figures(*self.args,progress=self.progress.emit,cancelled=self.isInterruptionRequested,review_context=self.review_context)
            self.done.emit(str(folder),count)
        except Exception as exc:self.failed.emit(str(exc))


class ReviewWindow(QtWidgets.QMainWindow):
    def __init__(self, calibration_store_path=None):
        super().__init__()
        self.setWindowTitle('입자 원시 데이터 분석기 — Offline Particle Review')
        self.resize(1550, 1000)
        self.data=None; self.path=None; self.events=[]; self.thresholds=[]; self.worker=None
        self.metadata={}; self.last_settings=None; self._editing=False
        self.property_calibration=None
        config_root=Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericConfigLocation))
        default_store=config_root/'ResonanceGUI'/'OfflineParticleReview'/'mass_sensitivity.json'
        self.calibration_store_path=Path(calibration_store_path) if calibration_store_path else default_store
        self.saved_sensitivity_profiles={}
        self.removed_stack=[]; self.figure_worker=None
        self.noise_window=self.spin(1.0,.1,120); self.noise_guard=self.spin(.25,0,30); self.noise_degree=QtWidgets.QComboBox(); self.noise_degree.addItems(['1 (선형 drift)','2 (완만한 곡률)'])
        self.mit_guard=self.spin(.25,0,30); self.mit_min_duration=self.spin(1.0,.1,120)
        self.mit_prominence=self.spin(2.0,.5,20); self.mit_local_fit=self.spin(.20,.05,.5)
        pg.setConfigOptions(background='w', foreground='#263344', antialias=False)
        root=QtWidgets.QWidget(); self.setCentralWidget(root)
        layout=QtWidgets.QVBoxLayout(root)
        toolbar=QtWidgets.QHBoxLayout(); layout.addLayout(toolbar)
        self.open_btn=QtWidgets.QPushButton('CSV 열기'); self.open_btn.clicked.connect(self.choose_file)
        toolbar.addWidget(self.open_btn)
        self.path_label=QtWidgets.QLabel('파일을 열거나 실행 시 CSV 경로를 지정하세요.'); toolbar.addWidget(self.path_label,1)
        self.export_btn=QtWidgets.QPushButton('결과 CSV + JSON 저장'); self.export_btn.clicked.connect(self.export_results); toolbar.addWidget(self.export_btn)
        resume=QtWidgets.QPushButton('검토 불러오기'); resume.clicked.connect(self.import_review); toolbar.addWidget(resume)
        screenshot=QtWidgets.QPushButton('상세 그래프 PNG'); screenshot.clicked.connect(self.export_plot); toolbar.addWidget(screenshot)

        # Keep the calibrated-property result visible independently of the long
        # control column.  The individual controls can scroll; this compact
        # strip remains in view and is also restored from a saved review.
        summary=QtWidgets.QFrame(); summary.setObjectName('propertySummary')
        summary.setStyleSheet('''
            QFrame#propertySummary { background: #f4f8fc; border: 1px solid #b9cadc; border-radius: 6px; }
            QFrame#propertySummary QLabel { border: none; background: transparent; }
            QLabel#propertySummaryTitle { color: #173a5e; font-size: 11pt; font-weight: bold; }
            QLabel#propertySummaryState { color: #ffffff; background: #607d96; border-radius: 9px; padding: 2px 8px; font-weight: bold; }
            QLabel[propertyValue="true"] { color: #102f4e; font-size: 10pt; font-weight: bold; }
            QLabel[propertyCaption="true"] { color: #617487; }
        ''')
        sg=QtWidgets.QGridLayout(summary); sg.setContentsMargins(12,7,12,8); sg.setHorizontalSpacing(18); sg.setVerticalSpacing(2)
        title=QtWidgets.QLabel('물성치 역산 결과'); title.setObjectName('propertySummaryTitle'); sg.addWidget(title,0,0)
        self.property_summary_state=QtWidgets.QLabel('계산 전'); self.property_summary_state.setObjectName('propertySummaryState'); self.property_summary_state.setAlignment(QtCore.Qt.AlignCenter); sg.addWidget(self.property_summary_state,0,1,1,4,QtCore.Qt.AlignLeft)
        summary_fields=[('적용 감도','summary_sensitivity'),('대표 신호','summary_signal'),('대표 부유질량','summary_mass'),('추가 역산','summary_inferred'),('신뢰도','summary_quality')]
        for column,(caption,name) in enumerate(summary_fields):
            cap=QtWidgets.QLabel(caption); cap.setProperty('propertyCaption',True); sg.addWidget(cap,1,column)
            value=QtWidgets.QLabel('—'); value.setProperty('propertyValue',True); value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse); sg.addWidget(value,2,column)
            setattr(self,name,value); sg.setColumnStretch(column,1)
        layout.addWidget(summary)

        split=QtWidgets.QSplitter(); layout.addWidget(split,1)
        self.left_scroll=QtWidgets.QScrollArea(); self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_scroll.setMinimumWidth(500)
        left=QtWidgets.QWidget(); left.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Maximum)
        ll=QtWidgets.QVBoxLayout(left); ll.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.left_scroll.setWidget(left); split.addWidget(self.left_scroll)
        controls=QtWidgets.QGroupBox('탐색 설정 (원본은 평균 처리하지 않음)'); form=QtWidgets.QFormLayout(controls); ll.addWidget(controls)
        self.widths=QtWidgets.QLineEdit('0.03, 0.05, 0.10, 0.20'); form.addRow('short 탐색 폭 (s, ≤0.20)',self.widths)
        self.ref_start=self.spin(1); self.ref_end=self.spin(6)
        self.start=self.spin(0); self.end=self.spin(1000)
        form.addRow('잡음 기준 시작 (s)',self.ref_start); form.addRow('잡음 기준 끝 (s)',self.ref_end)
        form.addRow('분석 시작 (s)',self.start); form.addRow('분석 끝 (s)',self.end)
        self.floor=self.spin(.15,0,1000); self.sigma=self.spin(6,.1,100); self.merge=self.spin(.3,.02,10)
        form.addRow('최소 깊이 (Hz)',self.floor); form.addRow('blank 깊이 잡음 배수',self.sigma); form.addRow('short 후보 합치기 (s)',self.merge)
        self.run_btn=QtWidgets.QPushButton('short 깊이 기준 분석'); self.run_btn.clicked.connect(self.analyze); form.addRow(self.run_btn)
        self.info=QtWidgets.QLabel(); self.info.setWordWrap(True); ll.addWidget(self.info)
        note=QtWidgets.QLabel('자동 결과는 입자 확정이 아닌 검토 후보입니다.\nshort(≤0.20 s)만 탐색하며, 국소 2차 배경 대비 원시 주파수 깊이로 판정합니다.\n점수 열은 이전 검토 CSV 호환용이며 새 자동 탐색에서는 비어 있습니다.'); note.setWordWrap(True); ll.addWidget(note)
        tabs=QtWidgets.QTabWidget(); tabs.setMinimumHeight(230); ll.addWidget(tabs)
        self.table=QtWidgets.QTableWidget(0,15)
        self.table.setMinimumHeight(200)
        self.table.setHorizontalHeaderLabels(['대표시각(s)','깊이(Hz)','탐색폭(s)','이전점수(호환)','압력후(s)','종류','검토','메모','시작(s)','끝(s)','실제구간(s)',
                                              '왕복쌍','방향','쌍평균(Hz)','쌍완료'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self.select_event)
        self.table.itemChanged.connect(self.edit_event)
        tabs.addTab(self.table,'후보 목록')
        self.pressure_table=QtWidgets.QTableWidget(0,3); self.pressure_table.setHorizontalHeaderLabels(['변경 시각(s)','A(V)','B(V)'])
        self.pressure_table.itemSelectionChanged.connect(self.select_pressure)
        self.pressure_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tabs.addTab(self.pressure_table,'압력 변경')
        self.meta=QtWidgets.QPlainTextEdit(); self.meta.setReadOnly(True); tabs.addTab(self.meta,'메타데이터 / 문턱')
        actions=QtWidgets.QHBoxLayout(); ll.addLayout(actions)
        remove=QtWidgets.QPushButton('선택 행 제거'); remove.clicked.connect(self.remove_selected); actions.addWidget(remove)
        keep=QtWidgets.QPushButton('검토 상태로 남기기'); keep.clicked.connect(self.keep_status); actions.addWidget(keep)
        self.undo_btn=QtWidgets.QPushButton('제거 취소'); self.undo_btn.setEnabled(False); self.undo_btn.clicked.connect(self.undo_remove); actions.addWidget(self.undo_btn)
        self.batch_btn=QtWidgets.QPushButton('남은 후보 그래프 전체 저장'); self.batch_btn.clicked.connect(self.export_all_plots); ll.addWidget(self.batch_btn)
        prop_box=QtWidgets.QGroupBox('왕복쌍 대표 신호 / 물성 역산'); pf=QtWidgets.QFormLayout(prop_box); ll.addWidget(prop_box)
        self.mode_number=QtWidgets.QSpinBox(); self.mode_number.setRange(1,30); self.mode_number.setValue(6)
        self.diameter_min=self.spin(190,.001,100000); self.diameter_max=self.spin(220,.001,100000)
        self.fluid_density=self.spin(997,0,50000); self.reference_density=self.spin(7930,.001,50000)
        self.mass_method=QtWidgets.QComboBox(); self.mass_method.addItem('기준 입자로 새 감도 보정 (1회)','reference'); self.mass_method.addItem('저장된/직접 감도 사용','sensitivity')
        self.mass_sensitivity=self.spin(1,.0001,1e9); self.mass_sensitivity.setDecimals(3)
        self.representative_method=QtWidgets.QComboBox(); self.representative_method.addItem('왕복쌍 평균들의 평균','mean'); self.representative_method.addItem('왕복쌍 평균들의 중앙값 (이상치 강건)','median')
        self.inference_method=QtWidgets.QComboBox(); self.inference_method.addItem('대표 부유질량만','mass_only'); self.inference_method.addItem('알려진 지름 → 밀도','density_from_diameter'); self.inference_method.addItem('알려진 밀도 → 등가지름','diameter_from_density')
        pf.addRow('측정 모드',self.mode_number); pf.addRow('알려진/기준 지름 최소 (µm)',self.diameter_min); pf.addRow('알려진/기준 지름 최대 (µm)',self.diameter_max)
        pf.addRow('유체 밀도 (kg/m³)',self.fluid_density); pf.addRow('보정 방식',self.mass_method)
        pf.addRow('알려진/기준 입자 밀도 (kg/m³)',self.reference_density); pf.addRow('질량 감도 (pg/Hz)',self.mass_sensitivity)
        pf.addRow('입자 대표 신호',self.representative_method); pf.addRow('외부 정보로 추가 계산',self.inference_method)
        self.mass_btn=QtWidgets.QPushButton('왕복쌍 대표값·부유질량 계산'); self.mass_btn.clicked.connect(self.calculate_particle_properties); pf.addRow(self.mass_btn)
        self.mass_info=QtWidgets.QLabel('개별 통과값은 품질 확인용이며, 질량은 연속 왕복쌍의 대표 신호에 한 번만 적용합니다.'); self.mass_info.setWordWrap(True); pf.addRow(self.mass_info)
        self.mass_method.currentIndexChanged.connect(self.update_mass_method)
        self.inference_method.currentIndexChanged.connect(self.update_mass_method)
        self.mode_number.valueChanged.connect(self.change_property_mode)
        self.update_mass_method()
        noise_box=QtWidgets.QGroupBox('후보 전후 주파수 노이즈 / SNR'); nf=QtWidgets.QFormLayout(noise_box); ll.addWidget(noise_box)
        nf.addRow('각 방향 구간 길이 (s)',self.noise_window); nf.addRow('후보 제외 guard (s)',self.noise_guard); nf.addRow('drift 모델',self.noise_degree)
        self.noise_btn=QtWidgets.QPushButton('남은 후보 노이즈 + SNR 계산'); self.noise_btn.clicked.connect(self.calculate_context_noise); nf.addRow(self.noise_btn)
        mit_box=QtWidgets.QGroupBox('MIT 논문 Fig. 4 방식 노이즈 비교'); mf=QtWidgets.QFormLayout(mit_box); ll.addWidget(mit_box)
        mf.addRow('피크 양쪽 추가 제외 guard (s)',self.mit_guard)
        mf.addRow('최소 baseline 길이 (s)',self.mit_min_duration)
        mf.addRow('필터',QtWidgets.QLabel('1 Hz, 2차 Butterworth high-pass (고정)'))
        mf.addRow('Antinode prominence (σ, 현재 데이터 2 / 논문 약 5–10)',self.mit_prominence)
        mf.addRow('국소 2차 fit 반폭 / antinode 간격 (추정)',self.mit_local_fit)
        self.mit_noise_btn=QtWidgets.QPushButton('MIT 논문 방식 비교 계산'); self.mit_noise_btn.clicked.connect(self.calculate_mit_noise); mf.addRow(self.mit_noise_btn)
        manual=QtWidgets.QHBoxLayout(); ll.addLayout(manual)
        self.manual_time=self.spin(0); manual.addWidget(self.manual_time)
        btn=QtWidgets.QPushButton('시각으로 수동 후보 추가'); btn.clicked.connect(self.add_manual); manual.addWidget(btn)
        self.right_tabs=QtWidgets.QTabWidget(); split.addWidget(self.right_tabs)
        right=QtWidgets.QWidget(); rl=QtWidgets.QVBoxLayout(right); self.right_tabs.addTab(right,'입자 파형 검토')
        self.noise_panel=NoisePanel(self); self.right_tabs.addTab(self.noise_panel,'노이즈 / SNR')
        self.mit_noise_panel=MitNoisePanel(self); self.right_tabs.addTab(self.mit_noise_panel,'MIT 논문 비교')
        self.savgol_panel=SavgolPanel(self); self.right_tabs.addTab(self.savgol_panel,'Savitzky–Golay 비교')
        self.mit_signal_panel=MitSignalPanel(self); self.right_tabs.addTab(self.mit_signal_panel,'MIT 피크 분석')
        self.savgol_panel.noiseRequested.connect(self.calculate_sg_context_noise)
        self.savgol_panel.mitRequested.connect(self.calculate_mit_signal)
        self.savgol_panel.propertiesRequested.connect(self.calculate_sg_particle_properties)
        self.noise_window.valueChanged.connect(self.noise_panel.invalidate)
        self.noise_guard.valueChanged.connect(self.noise_panel.invalidate)
        self.noise_degree.currentIndexChanged.connect(self.noise_panel.invalidate)
        self.mit_guard.valueChanged.connect(self.mit_noise_panel.invalidate)
        self.mit_min_duration.valueChanged.connect(self.mit_noise_panel.invalidate)
        self.mit_prominence.valueChanged.connect(self.mit_signal_panel.invalidate)
        self.mit_local_fit.valueChanged.connect(self.mit_signal_panel.invalidate)
        self.overview=pg.PlotWidget(); self.overview.setLabel('left','Frequency',units='Hz'); self.overview.setLabel('bottom','Time',units='s')
        self.overview.getAxis('left').enableAutoSIPrefix(False)
        self.overview.setMaximumHeight(220); rl.addWidget(self.overview)
        self.region=pg.LinearRegionItem([0,3]); self.region.setZValue(20); self.overview.addItem(self.region)
        self.region.sigRegionChanged.connect(self.region_changed)
        self.markers=pg.ScatterPlotItem(size=8,pen=None,brush='#d84242'); self.overview.addItem(self.markers)
        self.markers.sigClicked.connect(self.marker_clicked)
        detailbar=QtWidgets.QHBoxLayout(); rl.addLayout(detailbar)
        detailbar.addWidget(QtWidgets.QLabel('후보 클릭 또는 위 선택 구간 이동 / 휠 확대 / 우클릭 축 메뉴'))
        self.span=self.spin(1.4,.1,120); self.span.valueChanged.connect(self.select_event); detailbar.addWidget(QtWidgets.QLabel('확대 폭(s)')); detailbar.addWidget(self.span)
        self.residual=QtWidgets.QCheckBox('국소 배경 제거'); self.residual.toggled.connect(self.select_event); detailbar.addWidget(self.residual)
        samplebar=QtWidgets.QHBoxLayout(); rl.addLayout(samplebar)
        self.sample_start=self.spin(0); self.sample_end=self.spin(.2)
        for spin in (self.sample_start,self.sample_end):spin.setDecimals(7); spin.setSingleStep(.001)
        samplebar.addWidget(QtWidgets.QLabel('표본 계산 구간(s)')); samplebar.addWidget(self.sample_start); samplebar.addWidget(QtWidgets.QLabel('~')); samplebar.addWidget(self.sample_end)
        self.sample_visible=QtWidgets.QCheckBox('구간 표시'); self.sample_visible.setChecked(True); samplebar.addWidget(self.sample_visible)
        sample_all=QtWidgets.QPushButton('현재 화면 범위'); sample_all.clicked.connect(self.sample_use_view); samplebar.addWidget(sample_all)
        sample_apply=QtWidgets.QPushButton('선택 구간을 현재 후보에 적용'); sample_apply.clicked.connect(self.apply_selected_region); samplebar.addWidget(sample_apply)
        sample_save=QtWidgets.QPushButton('표본 통계 저장'); sample_save.clicked.connect(self.export_sample_stats); samplebar.addWidget(sample_save)
        self.sample_info=QtWidgets.QLabel(); self.sample_info.setWordWrap(True); rl.addWidget(self.sample_info)
        self.graph=pg.GraphicsLayoutWidget(); rl.addWidget(self.graph,1)
        self.plots=[]
        for i,label in enumerate(['Frequency (Hz)','Phase (deg)','Amplitude (mV)','Pressure command (V)']):
            p=self.graph.addPlot(row=i,col=0); p.setLabel('left',label); p.showGrid(x=True,y=True,alpha=.2)
            p.getAxis('left').enableAutoSIPrefix(False)
            if i:p.setXLink(self.plots[0])
            self.plots.append(p)
        self.plots[-1].setLabel('bottom','Time',units='s')
        self.sample_region=pg.LinearRegionItem([0,.2],brush=(30,160,170,35),pen=pg.mkPen('#159ca6'))
        self.sample_region.setZValue(30)
        self.sample_region.sigRegionChanged.connect(self.sample_region_changed)
        self.sample_start.valueChanged.connect(self.sample_spins_changed)
        self.sample_end.valueChanged.connect(self.sample_spins_changed)
        self.sample_visible.toggled.connect(self.sample_region.setVisible)
        self.plots[0].sigXRangeChanged.connect(self.update_sample_stats)
        self.plots[0].scene().sigMouseClicked.connect(self.graph_click)
        self.detail_note=QtWidgets.QLabel(); self.detail_note.setWordWrap(True); rl.addWidget(self.detail_note)
        split.setSizes([610,940]); self.statusBar().showMessage('장비 연결 없이 CSV만 분석합니다.')
        self.load_saved_sensitivities()
        self.update_property_summary()

    @staticmethod
    def spin(value, low=0, high=1e9):
        s=QtWidgets.QDoubleSpinBox(); s.setRange(low,high); s.setDecimals(4); s.setValue(value); return s

    def error(self, message):
        QtWidgets.QMessageBox.warning(self,'분석 안내',message)

    def choose_file(self):
        if self.events and QtWidgets.QMessageBox.question(self,'다른 파일 열기','현재 후보 검토 내용은 저장하지 않으면 사라집니다. 계속할까요?')!=QtWidgets.QMessageBox.Yes:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'원시 CSV 선택','','CSV (*.csv)')
        if path:self.load_file(path)

    def load_file(self,path):
        try:d,metadata=load_raw(path)
        except Exception as exc:self.error(str(exc)); return
        self.data=d; self.path=Path(path).resolve(); self.metadata=metadata; self.events=[]; self.thresholds=[]; self.last_settings=None; self.property_calibration=None
        self.update_property_summary('분석 대기')
        self.sample_times=d.Time_s.to_numpy()
        self.sample_region.setBounds([float(d.Time_s.iloc[0]),float(d.Time_s.iloc[-1])])
        self.sample_region.setRegion([float(d.Time_s.iloc[0]),min(float(d.Time_s.iloc[-1]),float(d.Time_s.iloc[0])+.2)])
        self.removed_stack.clear(); self.undo_btn.setEnabled(False)
        self.path_label.setText(self.path.name)
        t=d.Time_s.to_numpy(); self.start.setValue(t[0]); self.end.setValue(t[-1])
        self.ref_start.setValue(t[0]+1); self.ref_end.setValue(min(t[-1]-.6,t[0]+6))
        self.overview.clear(); self.overview.addItem(self.region); self.overview.addItem(self.markers)
        self.overview.plot(t,d.Frequency_Hz.to_numpy(),pen=pg.mkPen('#175aaa',width=1))
        self.changes=pressure_changes(d)
        self.overview.setXRange(t[0],t[-1],padding=.01); self.region.setBounds([t[0],t[-1]])
        self.region.setRegion([t[0],min(t[0]+10,t[-1])])
        self.changes=pressure_changes(d)
        self.pressure_table.setRowCount(len(self.changes))
        for row,values in enumerate(self.changes.to_numpy()):
            for col,value in enumerate(values):self.pressure_table.setItem(row,col,QtWidgets.QTableWidgetItem(f'{value:.4f}'))
        self.info.setText(f'{len(d):,}표본 / {t[-1]-t[0]:.2f}초 / 중앙 간격 {np.median(np.diff(t))*1000:.3f} ms\n압력 명령 변경 {len(self.changes)}회 (양쪽 정지 포함)\n잡음 기준 구간이 안정적인지 그래프로 확인하세요.')
        self.update_meta(); self.refresh_table(); self.draw_detail()

    def get_settings(self):
        widths=tuple(float(v.strip()) for v in self.widths.text().split(',') if v.strip())
        if not widths:raise ValueError('탐색 시간 폭을 입력하세요.')
        if any(not np.isfinite(v) or v<=0 or v>.20 for v in widths):raise ValueError('short 전용이므로 탐색 폭은 0초 초과, 0.20초 이하로 입력하세요.')
        return Settings(widths=widths,reference_start=self.ref_start.value(),reference_end=self.ref_end.value(),
                        analyze_start=self.start.value(),analyze_end=self.end.value(),sigma_multiplier=self.sigma.value(),
                        floor_hz=self.floor.value(),merge_s=self.merge.value())

    def analyze(self):
        if self.data is None:return
        if self.events and QtWidgets.QMessageBox.question(self,'다시 분석','새 분석은 현재 후보 목록과 검토 메모를 교체합니다. 계속할까요?')!=QtWidgets.QMessageBox.Yes:return
        try:settings=self.get_settings()
        except Exception as exc:self.error(str(exc)); return
        self.pending_settings=settings
        self.run_btn.setEnabled(False); self.open_btn.setEnabled(False)
        self.worker=Worker(self.data,settings); self.worker.done.connect(self.analysis_done); self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.worker_finished); self.worker.start(); self.statusBar().showMessage('분석 중…')

    def worker_finished(self):
        self.run_btn.setEnabled(True); self.open_btn.setEnabled(True)
        if self.statusBar().currentMessage()=='분석 중…':self.statusBar().showMessage('분석을 완료하지 못했습니다. 안내 메시지의 설정/데이터 조건을 확인하세요.')

    def analysis_done(self,events,thresholds):
        self.events=events; self.thresholds=thresholds; self.last_settings=self.pending_settings
        self.removed_stack.clear(); self.undo_btn.setEnabled(False)
        self.refresh_table(); self.update_meta()
        self.statusBar().showMessage(f'자동 후보 {len(events)}개. 원시 파형을 보고 검토 상태를 지정하세요.')
        if len(events)>=2 and self.mass_method.currentData()=='sensitivity':
            self.calculate_particle_properties(automatic=True)
        elif events and self.mass_method.currentData()=='sensitivity':
            self.mass_info.setText('완전한 왕복쌍이 없어 대표 신호를 계산하지 않았습니다. 정방향·역방향 후보가 모두 필요합니다.')
            self.update_property_summary('왕복쌍 부족')
        if events:self.table.selectRow(0)

    def refresh_table(self):
        self.noise_panel.invalidate()
        self.mit_noise_panel.invalidate()
        self.mit_signal_panel.invalidate()
        self._editing=True; self.table.setRowCount(len(self.events))
        for row,e in enumerate(self.events):
            values=[e['time_s'],e['depth_hz'],e['width_s'],e['contrast_hz'],e['pressure_delay_s'],e['family'],e['status'],e['note'],
                    e.get('event_start_s',e['time_s']),e.get('event_end_s',e['time_s']),e.get('duration_s',0.),
                    e.get('round_trip_pair',''),{'forward':'정방향','reverse':'역방향'}.get(e.get('direction'),''),
                    e.get('pair_depth_hz',np.nan),'예' if e.get('pair_complete') else ('아니오' if 'pair_complete' in e else '')]
            for col,v in enumerate(values):
                text=f'{v:.4f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else ('' if isinstance(v,(float,np.floating)) else str(v))
                item=QtWidgets.QTableWidgetItem(text)
                if col not in (6,7):item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(row,col,item)
        self.table.resizeColumnsToContents()
        for col in range(15):self.table.setColumnWidth(col,max(62,self.table.columnWidth(col)))
        self._editing=False
        if self.data is not None:
            t=self.data.Time_s.to_numpy(); f=self.data.Frequency_Hz.to_numpy()
            self.markers.setData([dict(pos=(e['time_s'],float(np.interp(e['time_s'],t,f))),data=i) for i,e in enumerate(self.events)])
        self.savgol_panel.set_source(self.data,self.events,self.table.currentRow())

    def edit_event(self,item):
        if not self._editing and item.column() in (6,7):
            self.events[item.row()]['status' if item.column()==6 else 'note']=item.text()
            self.noise_panel.invalidate()

    def select_event(self,*args):
        if self._editing:return
        row=self.table.currentRow()
        if not 0<=row<len(self.events):return
        event=self.events[row]; center=event['time_s']; self.manual_time.setValue(center)
        self.savgol_panel.set_selected_event(row)
        lo=event.get('event_start_s',center-.1); hi=event.get('event_end_s',center+.1)
        if not np.isfinite([lo,hi]).all() or hi<=lo:lo,hi=center-.1,center+.1
        self.sample_region.setRegion([lo,hi])
        self.region.setRegion([center-self.span.value()/2,center+self.span.value()/2]); self.draw_detail()

    def marker_clicked(self,item,points,*args):
        if points:self.table.selectRow(int(points[0].data()))

    def select_pressure(self):
        row=self.pressure_table.currentRow()
        if row>=0:
            center=float(self.changes.Time_s.iloc[row]); self.region.setRegion([center-1,center+5])

    def region_changed(self):
        if self.data is not None:self.draw_detail()

    def draw_detail(self):
        if self.data is None:return
        lo,hi=self.region.getRegion(); d=self.data; t=d.Time_s.to_numpy()
        mask=(t>=lo)&(t<=hi)
        for p in self.plots:p.clear()
        self.plots[0].addItem(self.sample_region)
        self.sample_region.setVisible(self.sample_visible.isChecked())
        f=d.Frequency_Hz.to_numpy(); baseline=None; center=self.manual_time.value()
        if self.residual.isChecked():
            try:baseline=local_baseline(d,center)
            except ValueError:pass
        self.plots[0].plot(t[mask],(f-baseline if baseline is not None else f)[mask],pen='#175aaa',symbol='o',symbolSize=3)
        self.plots[0].setLabel('left','Residual (Hz)' if baseline is not None else 'Frequency (Hz)')
        for i,col,scale,color in [(1,'Phase_deg',1,'#8b45a6'),(2,'Amplitude_R_V',1000,'#df7d17'),(3,'PressureCommand_A_V',1,'#239648'),(3,'PressureCommand_B_V',1,'#cf3d45')]:
            if col in d:self.plots[i].plot(t[mask],d[col].to_numpy()[mask]*scale,pen=color)
        for change in self.changes.Time_s:
            if lo<=change<=hi:
                for p in self.plots:p.addItem(pg.InfiniteLine(change,pen=pg.mkPen('#888888',style=QtCore.Qt.DashLine)))
        row=self.table.currentRow()
        if 0<=row<len(self.events):
            event=self.events[row]
            event_lo=event.get('event_start_s',event['time_s']); event_hi=event.get('event_end_s',event['time_s'])
            if np.isfinite([event_lo,event_hi]).all() and event_hi>event_lo:
                for p in self.plots:
                    p.addItem(pg.InfiniteLine(event_lo,pen=pg.mkPen('#159ca6',style=QtCore.Qt.DashLine)))
                    p.addItem(pg.InfiniteLine(event_hi,pen=pg.mkPen('#159ca6',style=QtCore.Qt.DashLine)))
        self.plots[0].setXRange(lo,hi,padding=0)
        for p in self.plots:p.enableAutoRange(axis='y')
        self.detail_note.setText('압력 명령: A=녹색 / B=빨강 · 회색 점선=명령 변경 · 청록 점선=저장될 이벤트 시작/끝\n청록 구간을 실제 이동 범위에 맞춘 뒤 「선택 구간을 현재 후보에 적용」하면 대표시각·깊이·구간이 함께 저장됩니다.')
        self.update_sample_stats()

    def sample_region_changed(self):
        lo,hi=self.sample_region.getRegion()
        for spin,value in [(self.sample_start,lo),(self.sample_end,hi)]:
            spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
        self.update_sample_stats()

    def sample_spins_changed(self):
        self.sample_region.setRegion(sorted((self.sample_start.value(),self.sample_end.value())))

    def sample_use_view(self):
        if self.data is not None:self.sample_region.setRegion(self.plots[0].viewRange()[0])

    def apply_selected_region(self):
        """Persist the draggable interval and its midpoint on the selected event."""
        if self.data is None:return
        row=self.table.currentRow()
        if not 0<=row<len(self.events):self.error('먼저 구간을 적용할 후보 행을 선택하세요.');return
        lo,hi=sorted(self.sample_region.getRegion())
        mask=(self.data.Time_s>=lo)&(self.data.Time_s<=hi)
        if np.count_nonzero(mask)<3:self.error('선택 구간에 최소 3개 원시 표본이 필요합니다.');return
        center=float((lo+hi)/2)
        try:baseline=local_baseline(self.data,center)
        except ValueError as exc:self.error(str(exc));return
        ids=np.flatnonzero(mask.to_numpy()); residual=self.data.Frequency_Hz.to_numpy()-baseline
        peak=ids[np.argmin(residual[ids])]
        event=self.events[row]
        event.update(time_s=center,peak_time_s=float(self.data.Time_s.iloc[peak]),
                     event_start_s=float(lo),event_end_s=float(hi),duration_s=float(hi-lo),
                     selection_source='manual',depth_hz=float(-residual[peak]))
        changes=self.changes.Time_s.to_numpy(); previous=changes[changes<=center]
        event['pressure_delay_s']=float(center-previous[-1]) if len(previous) else np.nan
        self.invalidate_particle_properties()
        self.refresh_table(); self.table.selectRow(row)
        self.statusBar().showMessage(f'{row+1}번 후보에 수동 구간 {lo:.4f}–{hi:.4f} s를 적용했습니다. 결과 저장 시 CSV/JSON에 포함됩니다.')

    def update_sample_stats(self,*args):
        if self.data is None:return
        stats=sample_window_stats(self.sample_times,*self.sample_region.getRegion())
        shown=sample_window_stats(self.sample_times,*self.plots[0].viewRange()[0])
        text=f"선택 구간: {stats['sample_count']:,}개 / {stats['window_duration_s']*1000:.3f} ms"
        if np.isfinite(stats['sampling_rate_hz']):
            text+=f" · 중앙 간격 {stats['median_interval_s']*1000:.4f} ms · {stats['sampling_rate_hz']:,.1f} Sa/s · 최대 간격 {stats['max_interval_s']*1000:.4f} ms"
        text+=f"\n현재 화면: {shown['sample_count']:,}개 · 청록색 양 끝을 드래그해 실제 이벤트 범위를 지정하세요."
        if stats['large_gap_count']:text+=f" 간격 >1.5배: {stats['large_gap_count']}곳"
        self.sample_info.setText(text)

    def export_sample_stats(self):
        if self.data is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'선택 구간 원시 표본 통계',str(self.path.with_name(self.path.stem+'_sample_stats.csv')),'CSV (*.csv)')
        if not path:return
        if Path(path).resolve()==self.path:self.error('원본 CSV에는 덮어쓸 수 없습니다.');return
        try:
            stats=sample_window_stats(self.sample_times,*self.sample_region.getRegion())
            pd.DataFrame([dict(source=str(self.path),**stats)]).to_csv(path,index=False,encoding='utf-8-sig')
            self.statusBar().showMessage(f'선택 구간 표본 통계 저장: {path}')
        except Exception as exc:self.error(str(exc))

    def graph_click(self,event):
        if event.double() and self.plots[0].sceneBoundingRect().contains(event.scenePos()):
            self.manual_time.setValue(self.plots[0].vb.mapSceneToView(event.scenePos()).x())

    def add_manual(self):
        if self.data is None:return
        try:e=manual_event(self.data,self.manual_time.value())
        except Exception as exc:self.error(str(exc));return
        self.events.append(e); self.events.sort(key=lambda e:e['time_s']); self.invalidate_particle_properties(); self.refresh_table(); self.table.selectRow(self.events.index(e))

    def invalidate_particle_properties(self):
        self.property_calibration=None
        for event in self.events:
            for key in ('round_trip_pair','direction','pair_depth_hz','pair_complete'):
                event.pop(key,None)
        self.mass_info.setText('후보 목록 또는 깊이가 바뀌었습니다. 왕복쌍 대표값을 다시 계산하세요.')
        self.update_property_summary('재계산 필요')
        self.update_meta()

    def remove_rows(self,rows):
        rows=sorted(set(r for r in rows if 0<=r<len(self.events)))
        if not rows:return
        self.removed_stack.append([(r,self.events[r]) for r in rows])
        removed=set(rows); self.events=[e for r,e in enumerate(self.events) if r not in removed]
        self.invalidate_particle_properties()
        self.table.clearSelection(); self.refresh_table(); self.undo_btn.setEnabled(True)
        if self.events:self.table.selectRow(min(rows[0],len(self.events)-1))
        self.statusBar().showMessage(f'목록에서 {len(rows)}행 제거 · 남은 {len(self.events)}행 · 제거 취소로 복구 가능 (원본 변경 없음)')

    def remove_selected(self):
        self.remove_rows([index.row() for index in self.table.selectionModel().selectedRows()])

    def keep_status(self):
        if not self.events:return
        statuses=list(dict.fromkeys(str(e.get('status','')).strip() for e in self.events))
        chosen,ok=QtWidgets.QInputDialog.getItem(self,'남길 검토 상태','이 검토 상태와 일치하는 행만 남깁니다. 메모 값은 필터 조건이 아닙니다.',statuses,0,False)
        if not ok:return
        rows=[r for r,e in enumerate(self.events) if str(e.get('status','')).strip()!=chosen]
        if rows and QtWidgets.QMessageBox.question(self,'후보 목록 정리',f'검토 상태가 「{chosen}」인 {len(self.events)-len(rows)}행만 남기고 {len(rows)}행을 제거할까요? 제거 취소로 복구할 수 있습니다.')==QtWidgets.QMessageBox.Yes:self.remove_rows(rows)

    def undo_remove(self):
        if not self.removed_stack:return
        restored=self.removed_stack.pop()
        for index,event in restored:self.events.insert(min(index,len(self.events)),event)
        self.invalidate_particle_properties()
        self.table.clearSelection(); self.refresh_table(); self.undo_btn.setEnabled(bool(self.removed_stack))
        self.statusBar().showMessage(f'{len(restored)}행 복구 · 현재 {len(self.events)}행')

    def update_meta(self):
        text=json.dumps(self.metadata,ensure_ascii=False,indent=2)
        if self.thresholds:text+='\n\nshort 깊이 기준 잡음 / 문턱\n'+pd.DataFrame(self.thresholds).to_string(index=False)
        if self.property_calibration:text+='\n\n질량 / 밀도 역산\n'+json.dumps(self.property_calibration,ensure_ascii=False,indent=2)
        self.meta.setPlainText(text)

    def update_mass_method(self):
        direct=self.mass_method.currentData()=='sensitivity'
        self.mass_sensitivity.setEnabled(direct)
        inference=self.inference_method.currentData()
        self.reference_density.setEnabled(not direct or inference=='diameter_from_density')
        diameter_needed=not direct or inference=='density_from_diameter'
        self.diameter_min.setEnabled(diameter_needed); self.diameter_max.setEnabled(diameter_needed)

    @staticmethod
    def _summary_number(value, pattern):
        try:
            value=float(value)
            return format(value,pattern) if np.isfinite(value) else '—'
        except (TypeError,ValueError):
            return '—'

    def update_property_summary(self, state=None):
        """Render the current calibration and inference as a compact dashboard."""
        cal=self.property_calibration
        if cal:
            method='기준 입자 보정' if cal.get('calibration_method')=='reference' else '저장 감도 적용'
            sensitivity=self._summary_number(cal.get('sensitivity_pg_hz'),',.3f')
            signal=self._summary_number(cal.get('representative_depth_hz'),'.4f')
            mass=self._summary_number(cal.get('representative_buoyant_mass_pg'),'.4g')
            inference=cal.get('inference_method')
            if inference=='density_from_diameter':
                density=self._summary_number(cal.get('representative_density_kg_m3'),',.1f')
                low=self._summary_number(cal.get('representative_density_low_kg_m3'),',.1f')
                high=self._summary_number(cal.get('representative_density_high_kg_m3'),',.1f')
                inferred=f'{density} kg/m³  ({low}–{high})'
            elif inference=='diameter_from_density':
                diameter=self._summary_number(cal.get('representative_equivalent_diameter_um'),'.2f')
                inferred=f'{diameter} µm'
            else:
                inferred='부유질량만 계산'
            pair_cv=self._summary_number(cal.get('pair_stats',{}).get('cv_percent'),'.2f')
            pair_count=int(cal.get('complete_pair_count',0))
            excluded=int(cal.get('excluded_event_count',0))
            quality=f'{pair_count}쌍 · CV {pair_cv}%'
            if excluded:quality+=f' · {excluded}개 제외'
            self.summary_sensitivity.setText(f'{sensitivity} pg/Hz\n{method}')
            self.summary_signal.setText(f'{signal} Hz')
            self.summary_mass.setText(f'{mass} pg')
            self.summary_inferred.setText(inferred)
            self.summary_quality.setText(quality)
            self.property_summary_state.setText(state or '역산 완료')
            self.property_summary_state.setStyleSheet('background: #25734f;')
            return

        profile=self.saved_sensitivity_profiles.get(str(int(self.mode_number.value()))) if self.saved_sensitivity_profiles else None
        if profile:
            sensitivity=self._summary_number(profile.get('sensitivity_pg_hz'),',.3f')
            self.summary_sensitivity.setText(f'{sensitivity} pg/Hz\n{self.mode_number.value()}차 모드 저장값')
        else:
            self.summary_sensitivity.setText('저장된 감도 없음')
        self.summary_signal.setText('—'); self.summary_mass.setText('—')
        self.summary_inferred.setText('—'); self.summary_quality.setText('—')
        self.property_summary_state.setText(state or ('역산 대기' if profile else '캘리브레이션 필요'))
        self.property_summary_state.setStyleSheet('background: #607d96;')

    def load_saved_sensitivities(self):
        try:self.saved_sensitivity_profiles=load_sensitivity_profiles(self.calibration_store_path)
        except Exception as exc:
            self.saved_sensitivity_profiles={}
            self.mass_info.setText(f'저장된 감도 보정 파일을 읽지 못했습니다: {exc}')
            return
        self.apply_saved_sensitivity(self.mode_number.value())

    def change_property_mode(self,mode):
        if self.property_calibration is not None:
            self.property_calibration=None
            for event in self.events:
                for key in ('round_trip_pair','direction','pair_depth_hz','pair_complete'):
                    event.pop(key,None)
            self.update_meta()
        self.apply_saved_sensitivity(mode)

    def apply_saved_sensitivity(self,mode):
        profile=self.saved_sensitivity_profiles.get(str(int(mode)))
        if profile:
            self.mass_sensitivity.setValue(profile['sensitivity_pg_hz'])
            index=self.mass_method.findData('sensitivity')
            if index >= 0:self.mass_method.setCurrentIndex(index)
            when=profile.get('calibrated_at','날짜 미기록')
            self.mass_info.setText(f"저장된 {int(mode)}차 모드 감도 {profile['sensitivity_pg_hz']:,.3f} pg/Hz 사용 중 · 보정 {when}")
            if self.property_calibration is None:self.update_property_summary('역산 대기')
        elif self.mass_method.currentData()=='sensitivity':
            index=self.mass_method.findData('reference')
            if index >= 0:self.mass_method.setCurrentIndex(index)
            self.mass_info.setText(f'{int(mode)}차 모드에 저장된 감도가 없습니다. 기준 입자로 한 번 보정하세요.')
            if self.property_calibration is None:self.update_property_summary('캘리브레이션 필요')

    def get_property_settings(self):
        return ParticlePropertySettings(mode_number=self.mode_number.value(),
            diameter_min_um=self.diameter_min.value(),diameter_max_um=self.diameter_max.value(),
            fluid_density_kg_m3=self.fluid_density.value(),calibration_method=self.mass_method.currentData(),
            reference_density_kg_m3=self.reference_density.value(),sensitivity_pg_hz=self.mass_sensitivity.value(),
            representative_method=self.representative_method.currentData(),inference_method=self.inference_method.currentData())

    def calculate_particle_properties(self, automatic=False, signal_label=None):
        if not self.events:self.error('질량·밀도를 계산할 후보가 없습니다.');return
        settings=self.get_property_settings()
        try:self.events,self.property_calibration=estimate_particle_properties(self.events,settings)
        except Exception as exc:self.error(str(exc));return
        self.mass_sensitivity.setValue(self.property_calibration['sensitivity_pg_hz'])
        saved=False
        if settings.calibration_method=='reference':
            try:
                profile=save_sensitivity_profile(self.calibration_store_path,self.property_calibration,self.path)
                self.saved_sensitivity_profiles[str(settings.mode_number)]=profile
                self.mass_method.setCurrentIndex(self.mass_method.findData('sensitivity'))
                saved=True
            except Exception as exc:
                self.error(f'계산은 완료했지만 감도 보정값을 저장하지 못했습니다: {exc}')
        self.refresh_table(); self.update_meta()
        cal=self.property_calibration; individual=cal['individual_stats']; pairs=cal['pair_stats']
        saved_text=' · 감도 영구 저장 완료' if saved else ''
        automatic_text='자동 적용 · ' if automatic else ''
        source=signal_label or (self.events[0].get('depth_signal_source','raw') if self.events else 'raw')
        result=(f"{automatic_text}입력 {source} · 왕복쌍 {cal['complete_pair_count']}개 · 대표 신호 {cal['representative_depth_hz']:.4f} Hz · "
                f"대표 부유질량 {cal['representative_buoyant_mass_pg']:.4g} pg · "
                f"쌍 표준편차 {pairs['std_hz']:.4f} Hz / 표준오차 {pairs['sem_hz']:.4f} Hz / CV {pairs['cv_percent']:.2f}% · "
                f"개별 CV {individual['cv_percent']:.2f}% · 방향 평균 {cal['direction_mean_forward_hz']:.4f}/{cal['direction_mean_reverse_hz']:.4f} Hz "
                f"(차이 {cal['direction_difference_percent']:.1f}%){saved_text}")
        if cal['inference_method']=='density_from_diameter':
            result+=f" · 지름 외부값 기반 밀도 {cal['representative_density_kg_m3']:,.1f} kg/m³ ({cal['representative_density_low_kg_m3']:,.1f}–{cal['representative_density_high_kg_m3']:,.1f})"
        elif cal['inference_method']=='diameter_from_density':
            result+=f" · 밀도 외부값 기반 등가지름 {cal['representative_equivalent_diameter_um']:.2f} µm"
        if cal['excluded_event_count']:
            result+=f" · 불완전쌍 통과 {cal['excluded_event_count']}개 제외"
        if cal['self_calibration_warning']:
            result+=' · 주의: 같은 데이터의 지름·밀도로 자체 보정한 값은 독립 측정값이 아닙니다.'
        self.mass_info.setText(result)
        self.update_property_summary('자동 역산 완료' if automatic else ('보정·역산 완료' if saved else '역산 완료'))
        self.statusBar().showMessage('왕복쌍 대표 신호와 대표 부유질량 계산을 완료했습니다.')

    def calculate_sg_particle_properties(self):
        if self.data is None or not self.events:self.error('SG 물성 계산을 수행할 후보가 없습니다.');return
        try:
            filtered=self.savgol_panel.filtered_data();sg=self.savgol_panel.get_settings()
            label=f"SG(p={sg['order']}, w={sg['window_length']})"
            self.events=recompute_event_depths(filtered,self.events,label)
            failed=[event for event in self.events if not np.isfinite(event.get('depth_hz',np.nan))]
            if failed:raise ValueError(f'SG Δf를 다시 계산하지 못한 후보가 {len(failed)}개 있습니다. 이벤트 구간과 양쪽 baseline을 확인하세요.')
            self.invalidate_particle_properties()
            self.calculate_particle_properties(signal_label=label)
            self.statusBar().showMessage('SG 신호에서 Δf를 다시 계산해 왕복쌍 대표값과 물성을 갱신했습니다. raw_depth_hz는 보존했습니다.')
        except Exception as exc:self.error(str(exc))

    def export_results(self):
        if self.data is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'검토 결과 저장',str(self.path.with_name(self.path.stem+'_review.csv')),'CSV (*.csv)')
        if not path:return
        target=Path(path)
        if target.resolve()==self.path:self.error('원본 CSV에는 덮어쓸 수 없습니다. 다른 이름을 선택하세요.');return
        if target.with_suffix('.json').exists() and QtWidgets.QMessageBox.question(self,'설정 파일 덮어쓰기','같은 이름의 JSON 설정 파일도 교체할까요?')!=QtWidgets.QMessageBox.Yes:return
        try:
            pd.DataFrame(self.events).to_csv(target,index=False,encoding='utf-8-sig')
            info=dict(source=str(self.path),settings=settings_dict(self.last_settings) if self.last_settings else None,metadata=self.metadata,thresholds=self.thresholds,pressure_changes=self.changes.to_dict('records'),sample_count=len(self.data),time_start=float(self.data.Time_s.iloc[0]),time_end=float(self.data.Time_s.iloc[-1]),note='탐색 후보이며 입자 확정/검출한계가 아닙니다.')
            info['noise_settings']=self.get_noise_settings()
            info['mit_noise_settings']=self.get_mit_noise_settings()
            info['savgol_settings']=self.savgol_panel.get_settings()
            info['mit_signal_settings']=self.get_mit_signal_settings()
            info['particle_property_settings']=settings_dict(self.get_property_settings())
            info['particle_property_calibration']=self.property_calibration
            if self.property_calibration:
                pair_target=target.with_name(target.stem+'_round_trip_pairs.csv')
                pd.DataFrame(self.property_calibration['pair_rows']).to_csv(pair_target,index=False,encoding='utf-8-sig')
                info['round_trip_pairs_file']=str(pair_target)
            target.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            self.statusBar().showMessage(f'저장 완료: {target}')
        except Exception as exc:self.error(str(exc))

    def calculate_context_noise(self):
        if self.data is None or not self.events:self.error('전후 noise를 계산할 후보가 없습니다.');return
        try:
            result=analyze_noise(self.data,self.events,**self.get_noise_settings(),signal_source='raw')
            self.noise_panel.set_result(result,self.data,self.path,deepcopy(self.metadata))
            self.right_tabs.setCurrentWidget(self.noise_panel)
            self.statusBar().showMessage('노이즈 / SNR 계산 완료. 표의 행을 선택해 제외 구간을 확인하고 결과를 저장하세요.')
        except Exception as exc:self.error(str(exc))

    def calculate_sg_context_noise(self):
        if self.data is None or not self.events:self.error('SG 노이즈/SNR을 계산할 후보가 없습니다.');return
        try:
            filtered=self.savgol_panel.filtered_data(); sg=self.savgol_panel.get_settings()
            fs=1/float(np.median(np.diff(filtered.Time_s.to_numpy())))
            label=f"SG(p={sg['order']}, w={sg['window_length']}, {sg['window_length']/fs * 1000:.3f} ms)"
            result=analyze_noise(filtered,self.events,**self.get_noise_settings(),signal_source=label,recompute_depth=True)
            metadata=deepcopy(self.metadata);metadata['Analysis_signal']=label
            self.noise_panel.set_result(result,filtered,self.path,metadata)
            self.right_tabs.setCurrentWidget(self.noise_panel)
            self.statusBar().showMessage('SG 신호에서 Δf와 전후 노이즈를 다시 계산해 SNR을 갱신했습니다.')
        except Exception as exc:self.error(str(exc))

    def calculate_mit_noise(self):
        if self.data is None or len(self.events)<2:
            self.error('MIT 비교에는 피크 사이 baseline을 만들기 위해 후보가 최소 2개 필요합니다.')
            return
        try:
            result=analyze_mit_noise(
                self.data, self.events, guard_s=self.mit_guard.value(),
                min_duration_s=self.mit_min_duration.value())
            self.mit_noise_panel.set_result(result,self.data,self.path,deepcopy(self.metadata))
            self.right_tabs.setCurrentWidget(self.mit_noise_panel)
            self.statusBar().showMessage('MIT 논문 방식 비교 계산 완료. 유효 구간과 필터 edge 경고를 확인하세요.')
        except Exception as exc:
            self.error(str(exc))

    def calculate_mit_signal(self):
        if self.data is None or not self.events:
            self.error('MIT 피크 분석에는 검토 후보가 필요합니다.');return
        try:
            settings=self.get_mit_signal_settings()
            result=analyze_mit_signal(self.data,self.events,**settings,
                noise_guard_s=self.mit_guard.value(),noise_min_duration_s=self.mit_min_duration.value())
            self.mit_signal_panel.set_result(result,self.path,deepcopy(self.metadata))
            self.right_tabs.setCurrentWidget(self.mit_signal_panel)
            valid=int((result['summary'].status=='ok').sum())
            self.statusBar().showMessage(f'MIT 방식 피크 분석 완료: 유효 {valid}/{len(result["summary"])}개. 추정 파라미터와 antinode를 확인하세요.')
        except Exception as exc:self.error(str(exc))

    def get_noise_settings(self):
        return dict(window_s=self.noise_window.value(),guard_s=self.noise_guard.value(),degree=self.noise_degree.currentIndex()+1)

    def get_mit_noise_settings(self):
        return dict(guard_s=self.mit_guard.value(), min_duration_s=self.mit_min_duration.value(),
                    cutoff_hz=1.0, order=2)

    def get_mit_signal_settings(self):
        sg=self.savgol_panel.get_settings()
        return dict(sg_order=sg['order'],sg_window=sg['window_length'],
                    prominence_sigma=self.mit_prominence.value(),baseline_multiplier=3.0,
                    local_fit_fraction=self.mit_local_fit.value(),antinode_count=7,
                    outer_tolerance_percent=20.0)

    def show_noise_summary(self,result):
        rows=[]
        for context,g in result.groupby('context'):
            rows.append(f'{context}: N={len(g)} | raw std median={g.raw_std_hz.median():.4f} Hz | detrended σ median={g.detrended_std_hz.median():.4f} Hz | 10σ median={g.threshold_10sigma_hz.median():.4f} Hz')
        QtWidgets.QMessageBox.information(self,'전후 noise 결과','\n'.join(rows))

    def import_review(self):
        if self.data is None:self.error('먼저 해당 원본 CSV를 여세요.');return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'저장한 검토 CSV 선택','','CSV (*.csv)')
        if not path:return
        try:
            info=json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
            saved_source=Path(info['source']).resolve()
            if saved_source!=self.path:
                relocated=(not saved_source.exists() and saved_source.name==self.path.name
                    and info.get('sample_count')==len(self.data)
                    and np.isclose(info.get('time_start',np.nan),self.data.Time_s.iloc[0],rtol=0,atol=1e-6)
                    and np.isclose(info.get('time_end',np.nan),self.data.Time_s.iloc[-1],rtol=0,atol=1e-6))
                if not relocated:
                    raise ValueError('검토 파일의 원본 경로가 현재 파일과 다릅니다. 기록된 원본 CSV를 먼저 여세요.')
            if info.get('sample_count',len(self.data))!=len(self.data):raise ValueError('원본의 표본 수가 저장 당시와 다릅니다.')
            try:table=pd.read_csv(path,encoding='utf-8-sig')
            except pd.errors.EmptyDataError:table=pd.DataFrame()
            if len(table) and not {'time_s','depth_hz','width_s','contrast_hz','pressure_delay_s','family','status','note'}<=set(table):raise ValueError('검토 CSV 열 형식이 맞지 않습니다.')
            loaded=table.to_dict('records')
            for e in loaded:
                e['note']='' if pd.isna(e['note']) else str(e['note'])
                e['status']='미검토' if pd.isna(e['status']) else str(e['status'])
                for key,default in [('peak_time_s',e['time_s']),('event_start_s',e['time_s']),('event_end_s',e['time_s']),('duration_s',0.)]:
                    if key not in e or pd.isna(e[key]):e[key]=default
                if 'selection_source' not in e or pd.isna(e['selection_source']):e['selection_source']='legacy'
            if self.events and QtWidgets.QMessageBox.question(self,'검토 교체','현재 목록을 저장된 검토 목록으로 교체할까요?')!=QtWidgets.QMessageBox.Yes:return
            self.events=loaded; self.thresholds=info.get('thresholds',[])
            self.removed_stack.clear(); self.undo_btn.setEnabled(False)
            self.last_settings=Settings(**info['settings']) if info.get('settings') else None
            ns=info.get('noise_settings')
            if ns:
                self.noise_window.setValue(ns.get('window_s',1.0)); self.noise_guard.setValue(ns.get('guard_s',.25)); self.noise_degree.setCurrentIndex(ns.get('degree',1)-1)
            ms=info.get('mit_noise_settings')
            if ms:
                self.mit_guard.setValue(ms.get('guard_s',.25)); self.mit_min_duration.setValue(ms.get('min_duration_s',1.0))
            self.savgol_panel.set_settings(info.get('savgol_settings'))
            mss=info.get('mit_signal_settings')
            if mss:
                self.mit_prominence.setValue(mss.get('prominence_sigma',2.0));self.mit_local_fit.setValue(mss.get('local_fit_fraction',.20))
            ps=info.get('particle_property_settings')
            if ps:
                self.mode_number.setValue(ps.get('mode_number',6)); self.diameter_min.setValue(ps.get('diameter_min_um',190)); self.diameter_max.setValue(ps.get('diameter_max_um',220))
                self.fluid_density.setValue(ps.get('fluid_density_kg_m3',997)); self.reference_density.setValue(ps.get('reference_density_kg_m3',7930)); self.mass_sensitivity.setValue(ps.get('sensitivity_pg_hz',1))
                method=self.mass_method.findData(ps.get('calibration_method','reference')); self.mass_method.setCurrentIndex(max(0,method))
                representative=self.representative_method.findData(ps.get('representative_method','mean')); self.representative_method.setCurrentIndex(max(0,representative))
                inference=self.inference_method.findData(ps.get('inference_method','mass_only')); self.inference_method.setCurrentIndex(max(0,inference))
            self.property_calibration=info.get('particle_property_calibration')
            self.update_property_summary('저장 결과 불러옴' if self.property_calibration else '역산 대기')
            self.refresh_table(); self.update_meta()
            if loaded:self.table.selectRow(0)
            self.statusBar().showMessage('저장된 검토 목록을 불러왔습니다.' if saved_source==self.path else '이동된 원본: 파일명·표본 수·시간 범위가 일치하여 검토 목록을 불러왔습니다.')
        except Exception as exc:self.error(str(exc))

    def export_plot(self):
        if self.data is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'현재 상세 그래프 저장','particle_detail.png','PNG (*.png)')
        if path:
            try:pg.exporters.ImageExporter(self.graph.scene()).export(path)
            except Exception as exc:self.error(str(exc))

    def export_all_plots(self):
        if self.data is None or not self.events:self.error('저장할 후보가 없습니다.');return
        if self.worker is not None and self.worker.isRunning():self.error('자동 분석이 끝난 뒤 저장하세요.');return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,f'남은 {len(self.events)}행의 그래프 저장 위치 (확대 폭 {self.span.value():g}초)',str(self.path.parent))
        if not parent:return
        self.figure_worker=FigureExportWorker(self.data,self.events,parent,self.path,self.span.value(),self.residual.isChecked(),settings_dict(self.last_settings) if self.last_settings else None,dict(metadata=self.metadata,thresholds=self.thresholds,noise_settings=self.get_noise_settings(),mit_noise_settings=self.get_mit_noise_settings(),particle_property_settings=settings_dict(self.get_property_settings()),particle_property_calibration=self.property_calibration))
        self.export_progress=QtWidgets.QProgressDialog('후보별 주파수·위상·진폭·압력 그래프 저장 중…','중단',0,len(self.events),self)
        self.export_progress.setWindowTitle('남은 후보 그래프 일괄 저장')
        self.export_progress.setWindowModality(QtCore.Qt.ApplicationModal)
        self.export_progress.setMinimumDuration(0); self.export_progress.setAutoClose(False); self.export_progress.setAutoReset(False)
        self.export_progress.canceled.connect(self.figure_worker.requestInterruption)
        self.figure_worker.progress.connect(lambda count,total:self.export_progress.setValue(count))
        self.figure_worker.done.connect(self.batch_done)
        self.figure_worker.failed.connect(self.batch_failed)
        self.batch_btn.setEnabled(False)
        self.figure_worker.finished.connect(lambda:self.batch_btn.setEnabled(True))
        self.export_progress.show(); self.figure_worker.start()

    def batch_done(self,folder,count):
        self.export_progress.close()
        self.statusBar().showMessage(f'그래프 {count}개 저장: {folder}')
        QtWidgets.QMessageBox.information(self,'일괄 저장 결과',f'그래프 {count}개와 검토 CSV/JSON을 저장했습니다.\n{folder}')

    def batch_failed(self,message):
        self.export_progress.close(); self.error(message)

    def closeEvent(self,event):
        if self.noise_panel.save_worker is not None and self.noise_panel.save_worker.isRunning():
            self.statusBar().showMessage('노이즈 / SNR 저장이 끝난 뒤 닫아주세요.'); event.ignore()
        elif self.figure_worker is not None and self.figure_worker.isRunning():
            self.statusBar().showMessage('그래프 저장을 마치거나 중단한 뒤 닫아주세요.'); event.ignore()
        elif self.worker is not None and self.worker.isRunning():
            self.statusBar().showMessage('분석이 끝난 뒤 닫아주세요.'); event.ignore()
        else:event.accept()


def configure_application(app):
    app.setStyle('Fusion')
    # Explicit Korean-capable font also permits headless Windows rendering.
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        QtGui.QFontDatabase.addApplicationFont(str(font))
        app.setFont(QtGui.QFont('Malgun Gothic',9))


def main():
    app=QtWidgets.QApplication(sys.argv)
    configure_application(app)
    window=ReviewWindow(); window.show()
    if len(sys.argv)>1:window.load_file(sys.argv[1])
    sys.exit(app.exec_())


if __name__=='__main__':main()
