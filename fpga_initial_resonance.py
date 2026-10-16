"""Open-loop validation before enabling a resonance tracking PLL."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from fpga_resonance_gui import SweepWorker
from fpga_auto_sweep import peak_candidates
from resonance_fit import fit_complex_resonance, assess_fit_quality


def verification_bounds(reference,sweep_low,sweep_high,peaks):
    """Acquisition coverage is independent of the PLL's tracking limits."""
    low,high=float(sweep_low),float(sweep_high)
    if not 0<low<reference<high<6250000:
        raise ValueError("초기 주파수를 포함하도록 Sweeper 시작/끝 범위를 설정하세요.")
    measured=sorted(set(float(f) for f in peaks if low<=f<=high))
    if measured:
        chosen=min(measured,key=lambda f:abs(f-reference))
        below=[f for f in measured if f<chosen]
        above=[f for f in measured if f>chosen]
        if below: low=max(low,(below[-1]+chosen)/2)
        if above: high=min(high,(above[0]+chosen)/2)
    if not low<reference<high:
        raise ValueError("초기 주파수가 선택 모드 경계를 벗어났습니다. 피크를 다시 선택하세요.")
    return low,high


def isolated_peak(data, reference, low, high):
    candidates,_=peak_candidates(data,reference,low,high)
    candidates=sorted(candidates,key=lambda p:abs(p["frequency_hz"]-reference))
    if not candidates:
        raise ValueError("공진 피크가 없습니다.")
    peak=candidates[0]
    if len(candidates)>1 and candidates[1]["prominence"]>.5*peak["prominence"]:
        raise ValueError("여러 공진 후보가 있어 자동 모드 선택을 중단합니다. 스윕에서 피크를 다시 선택하세요.")
    if peak["edge"] or peak["bounds"] is None:
        r=np.asarray(data["R"],float)
        top=float(np.max(r))
        raise ValueError(f"피크 경계 또는 양쪽 −3 dB 선폭 미확보: {low:.1f}–{high:.1f} Hz, "
                         f"양 끝 진폭/최대={r[0]/top:.3f}/{r[-1]/top:.3f} (−3 dB=0.707). "
                         "Sweeper 범위를 양쪽으로 넓혀 다시 측정하세요.")
    return peak


class InitialResonanceWorker(QThread):
    completed=pyqtSignal(object)
    failed=pyqtSignal(str)
    progress=pyqtSignal(str)

    def __init__(self,controller,parameters,*,low,high,reference,directory=None):
        super().__init__()
        self.controller,self.parameters=controller,parameters
        self.low,self.high,self.reference=low,high,reference
        self.directory=Path(directory or Path(__file__).parent/"diagnostics")
        self.cancelled=False
        self.segment=None
        self.records=[]

    def stop(self):
        self.cancelled=True
        if self.segment: self.segment.stop()

    def _scan(self,low,high,points,name):
        if self.cancelled: raise RuntimeError("초기 공진 검증 중지")
        self.progress.emit(f"초기 공진 검증: {name} ({low:.1f}–{high:.1f} Hz)")
        p=replace(self.parameters,frequency_hz=low,minimum_frequency_hz=low,maximum_frequency_hz=high)
        segment=SweepWorker(self.controller,p,np.linspace(low,high,points),.1,256)
        self.segment=segment
        try:
            segment.measure()
            if self.cancelled: raise RuntimeError("초기 공진 검증 중지")
            return segment.last_result
        finally:
            if segment.last_result:
                self.records.append((name,segment.last_result))
            self.segment=None

    def run(self):
        result=error=None
        try:
            if not 0<self.low<self.high<6250000:
                raise ValueError("유효한 초기 공진 탐색 범위가 필요합니다.")
            # First inspect a local region; expand at most once inside the
            # Sweeper coverage and known neighboring-mode boundaries.
            half=min((self.high-self.low)/4,max(10.,self.reference/self.parameters.resonator_q))
            low,high=max(self.low,self.reference-half),min(self.high,self.reference+half)
            data=self._scan(low,high,81,"local")
            try:
                peak=isolated_peak(data,self.reference,low,high)
            except ValueError as exc:
                if "여러" in str(exc): raise
                data=self._scan(self.low,self.high,161,"bounded_retry")
                peak=isolated_peak(data,self.reference,self.low,self.high)
            width=peak["bounds"][1]-peak["bounds"][0]
            low=max(self.low,peak["frequency_hz"]-1.5*width)
            high=min(self.high,peak["frequency_hz"]+1.5*width)
            fine=self._scan(low,high,161,"fine")
            peak=isolated_peak(fine,peak["frequency_hz"],low,high)
            fit=fit_complex_resonance(fine["freqs"],fine["X"],fine["Y"],fine["X_sem"],fine["Y_sem"],
                f0_guess=peak["frequency_hz"],f0_bounds=(low,high),cancel_check=lambda:self.cancelled,max_nfev=500)
            quality=assess_fit_quality(fine["freqs"],fine["X"],fine["Y"],fit)
            if self.cancelled: raise RuntimeError("초기 공진 검증 중지")
            if not fit.success or not quality.accepted:
                raise ValueError("공진 fitting 품질 미달: "+"; ".join(quality.reasons))
            if abs(fit.f0_hz-peak["frequency_hz"])>.25*fit.linewidth_hz:
                raise ValueError("진폭 최대점과 복소 공진 중심의 차이가 큽니다. 배경/모드 선택을 검토하세요.")
            result=dict(frequency_hz=peak["frequency_hz"],q=fit.q,fit_f0_hz=fit.f0_hz,
                        linewidth_hz=fit.linewidth_hz,low_hz=self.low,high_hz=self.high,
                        original_frequency_hz=self.parameters.frequency_hz)
        except Exception as exc:
            error=str(exc)
        finally:
            try:
                self.controller.stop_output()
                self.controller.stop_stream()
            except Exception as exc:
                error=f"초기 공진 검증 출력 정지 실패: {exc}"
            try:
                self.directory.mkdir(parents=True,exist_ok=True)
                stem=self.directory/f"fpga_initial_resonance_{datetime.now():%Y%m%d_%H%M%S_%f}"
                for name,data in self.records:
                    np.savez_compressed(str(stem)+f"_{name}.npz",**data)
                stem.with_suffix(".json").write_text(json.dumps(dict(accepted=error is None,error=error,result=result,
                    original_frequency_hz=self.parameters.frequency_hz,reference_hz=self.reference,
                    allowed_low_hz=self.low,allowed_high_hz=self.high),
                    ensure_ascii=False,indent=2),encoding="utf-8")
                if result: result["path"]=str(stem.with_suffix(".json"))
            except Exception as exc:
                error=error or f"초기 공진 기록 저장 실패: {exc}"
        if error: self.failed.emit(error)
        else: self.completed.emit(result)
