"""GUI ownership of the bounded closed-loop delay optimization."""
from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel, QPushButton, QMessageBox
from fpga_pll_refinement import ClosedLoopDelayWorker
from fpga_initial_resonance import InitialResonanceWorker,verification_bounds


class RefinementMixin:
    def _build_refinement_controls(self, group):
        self.fpga_peak_worker = None
        self.fpga_initial_worker=None
        self._initial_outcome=None
        self._pll_start_validated=False
        self._pll_validation_failed=False
        self._initial_locked=[]
        self._peak_outcome = None
        self._peak_locked = []
        self._last_peak_refinement = None
        layout = group.layout()
        row = layout.rowCount()
        self.fpga_peak_auto_check = QCheckBox("error 영점 보정 후 폐루프 진폭 최대점 확인")
        self.fpga_peak_auto_check.setChecked(True)
        self.fpga_peak_auto_check.setToolTip("PLL 시작 시 주변 delay 탐색. AGC 일시 해제, 완료 후 원래 상태 복구.")
        self.fpga_auto_delay_check.setToolTip("PLL 시작 시 초기 공진 스윕 후 반드시 수행합니다. 수동 error 보정은 별도로 실행할 수 있습니다.")
        layout.addWidget(self.fpga_peak_auto_check,row,0,1,2)
        layout.addWidget(QLabel("폐루프 delay 탐색 반폭 (turn):"),row+1,0)
        self.fpga_peak_width_spin = QDoubleSpinBox()
        self.fpga_peak_width_spin.setDecimals(6)
        self.fpga_peak_width_spin.setRange(1/2048,1/32)
        self.fpga_peak_width_spin.setValue(1/128)
        layout.addWidget(self.fpga_peak_width_spin,row+1,1)
        self.fpga_peak_btn = QPushButton("폐루프 delay · 진폭 최대점 지금 확인")
        self.fpga_peak_btn.clicked.connect(self.start_fpga_peak_refinement)
        layout.addWidget(self.fpga_peak_btn,row+2,0,1,2)
        self.fpga_peak_label = QLabel("폐루프 진폭 최대점: 미확인")
        self.fpga_peak_label.setWordWrap(True)
        layout.addWidget(self.fpga_peak_label,row+3,0,1,2)
        startup_note=QLabel("PLL 시작: 공진 스윕 → error 영점 → 폐루프 진폭 검증이 모두 통과해야 기록합니다.")
        startup_note.setWordWrap(True)
        layout.addWidget(startup_note,row+4,0,1,2)

    def _auto_busy(self):
        worker = getattr(self,"fpga_peak_worker",None)
        initial=getattr(self,"fpga_initial_worker",None)
        return bool(worker and worker.isRunning()) or bool(initial and initial.isRunning()) or super()._auto_busy()

    def start_pll_preflight(self):
        if self._auto_busy() or not self.fpga:
            return False
        for name in ("sweep_worker","fpga_delay_worker","fpga_error_worker","fpga_response_worker"):
            worker=getattr(self,name,None)
            if worker and worker.isRunning(): return False
        if self.pingpong_active or self.baseline_active:
            return False
        if self.plotter_worker and self.plotter_worker.isRunning(): self.stop_plotter()
        self._pll_start_validated=False
        self._pll_validation_failed=True
        self._initial_parameters=None
        self._initial_resonance_result=None
        self._last_peak_refinement=None
        p=self._fpga_parameters()
        low,high=self.f_start_spin.value(),self.f_stop_spin.value()
        self._initial_capture_span=self.pll_range_spin.value()
        # Sweeper coverage supplies acquisition bounds independently of PLL
        # capture limits. Known swept peaks bound the chosen mode. Never expand into a known
        # neighboring mode merely to recover a mistyped starting frequency.
        peaks=sorted(float(item[0]) for item in self._peak_candidates)
        reference=p.frequency_hz
        fit=self.last_resonance_fit
        if fit is not None and low<fit.f0_hz<high: reference=float(fit.f0_hz)
        low,high=verification_bounds(reference,low,high,peaks)
        # These stages are mandatory for a validated particle acquisition.
        self.fpga_auto_delay_check.setChecked(True)
        self.fpga_peak_auto_check.setChecked(True)
        self._initial_outcome=None
        worker=InitialResonanceWorker(self.fpga,p,low=low,high=high,reference=reference)
        self.fpga_initial_worker=worker
        self._initial_locked=[]
        for widget in list(self._settings_widgets().values())+self.findChildren(QPushButton):
            if widget is self.pll_enable_btn or widget.objectName()=="emergency_stop": continue
            self._initial_locked.append((widget,widget.isEnabled()))
        for widget,_ in self._initial_locked: widget.setEnabled(False)
        worker.completed.connect(lambda result,source=worker:self._initial_completed(result)
            if self.fpga_initial_worker is source else None)
        worker.failed.connect(lambda message,source=worker:self._initial_failed(message)
            if self.fpga_initial_worker is source else None)
        worker.progress.connect(lambda message,source=worker:self.pll_design_label.setText(message)
            if self.fpga_initial_worker is source else None)
        worker.finished.connect(lambda source=worker:self._initial_finished()
            if self.fpga_initial_worker is source else None)
        self.pll_enable_btn.setText("초기 공진 검증 중…")
        worker.start()
        return True

    def _initial_completed(self,result): self._initial_outcome=(True,result)
    def _initial_failed(self,message): self._initial_outcome=(False,message)

    def _initial_finished(self):
        for widget,enabled in self._initial_locked: widget.setEnabled(enabled)
        self._initial_locked=[]
        self.fpga_initial_worker=None
        if not self.pll_enable_btn.isChecked(): return
        success,value=self._initial_outcome or (False,"초기 공진 결과 없음")
        if not success:
            self._abort_pll_start(value)
            return
        self.pll_center_spin.setValue(value["frequency_hz"])
        self.pll_q_spin.setValue(value["q"])
        # Recentring must not move the capture range into a neighboring mode.
        span=min(getattr(self,"_initial_capture_span",self.pll_range_spin.value()),
                 value["frequency_hz"]-value["low_hz"],value["high_hz"]-value["frequency_hz"])
        if span<=0:
            self._abort_pll_start("검증된 피크의 capture range 확보 실패")
            return
        self.pll_range_spin.setValue(span)
        self._initial_resonance_result=value
        self._initial_parameters=self._fpga_parameters()
        self.pll_design_label.setText(f"초기 공진 확인: {value['original_frequency_hz']:.3f} → {value['frequency_hz']:.3f} Hz")
        if not self._start_fpga_delay_calibration(start_tracking=True):
            self._abort_pll_start("error 영점 보정을 시작하지 못했습니다.")

    def _abort_pll_start(self,message):
        self._pll_start_validated=False
        self._pll_validation_failed=True
        self._start_pll_after_delay=False
        try:
            if self.fpga:
                self.fpga.stop_output()
                self.fpga.stop_stream()
        except Exception as exc:
            message+=f" · 출력 정지 확인 실패: {exc}"
        self.pll_enable_btn.blockSignals(True)
        self.pll_enable_btn.setChecked(False)
        self.pll_enable_btn.blockSignals(False)
        self.pll_enable_btn.setText("PLL 활성화")
        self.pll_design_label.setText("PLL 시작 중단: "+message)
        self.pll_locked_label.setText("공진 검증 실패 · 스윕/피크 선택 확인 필요")
        self.plotter_start_btn.setEnabled(False)

    def start_fpga_peak_refinement(self):
        if not self.fpga or not self.pll_enable_btn.isChecked():
            QMessageBox.warning(self,"폐루프 delay","먼저 PLL tracking을 활성화하세요.")
            return
        if self._auto_busy() or self.baseline_active or self.pingpong_active:
            return
        for name in ("sweep_worker","fpga_delay_worker","fpga_error_worker","fpga_response_worker"):
            worker=getattr(self,name,None)
            if worker and worker.isRunning():
                return
        if self.plotter_worker and self.plotter_worker.isRunning():
            self.stop_plotter()
        self._peak_outcome=None
        self._pll_start_validated=False
        self._last_peak_refinement=None
        self._peak_restore_failed=False
        worker=ClosedLoopDelayWorker(self.fpga,half_width=self.fpga_peak_width_spin.value())
        self.fpga_peak_worker=worker
        self._peak_locked=[]
        for widget in list(self._settings_widgets().values())+self.findChildren(QPushButton):
            if widget is self.pll_enable_btn or widget.objectName()=="emergency_stop":
                continue
            self._peak_locked.append((widget,widget.isEnabled()))
        for widget,_ in self._peak_locked:
            widget.setEnabled(False)
        worker.progress.connect(self.fpga_peak_label.setText)
        worker.completed.connect(self._peak_completed)
        worker.failed.connect(self._peak_failed)
        worker.finished.connect(self._peak_finished)
        self.fpga_peak_label.setText("폐루프 delay 탐색 중 · 입자 기록 일시 중단")
        worker.start()

    def _peak_completed(self,result):
        self._peak_outcome=(True,result)

    def _peak_failed(self,message):
        self._peak_outcome=(False,message)
        self._peak_restore_failed=message.startswith("PLL delay/mode 복구 실패")

    def _peak_finished(self):
        for widget,enabled in self._peak_locked:
            widget.setEnabled(enabled)
        self._peak_locked=[]
        self.fpga_peak_worker=None
        success,value=self._peak_outcome or (False,"탐색 결과 없음")
        if success:
            self._pll_start_validated=True
            self._pll_validation_failed=False
            self.fpga_pll_delay_spin.setValue(value["delay_turns"])
            self._last_peak_refinement=value
            self.fpga_peak_label.setText(f"진폭 최대 구간 확인: delay={value['delay_turns']:.6f} turn · "
                                         f"f={value['peak']['frequency_hz']:.3f} Hz · AGC 원래 상태 복구")
            self.fpga_peak_label.setToolTip(value["path"])
            self._last_step_response=None
            self._last_step_response_context=None
        else:
            self.fpga_peak_label.setText(f"진폭 최대점 미확인: {value}")
            self._abort_pll_start(str(value))
            return
        if getattr(self,"_peak_restore_failed",False):
            self.pll_enable_btn.blockSignals(True)
            self.pll_enable_btn.setChecked(False)
            self.pll_enable_btn.blockSignals(False)
            self.pll_enable_btn.setText("PLL 활성화")
            self.pll_locked_label.setText("FPGA 복구 실패 · 출력 상태 확인 필요")
            return
        p=self.fpga.last_parameters if self.fpga else None
        if self.pll_enable_btn.isChecked() and p and p.mode.run:
            self.start_plotter()

    def _stop_peak_refinement(self):
        initial=getattr(self,"fpga_initial_worker",None)
        if initial and initial.isRunning():
            initial.stop()
            if not initial.wait(3000): return False
        worker=getattr(self,"fpga_peak_worker",None)
        if worker and worker.isRunning():
            worker.stop()
            return worker.wait(3000)
        return True
