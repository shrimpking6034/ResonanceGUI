"""Bounded closed-loop phase search; never optimizes while recording particles."""
from dataclasses import replace
from pathlib import Path
from datetime import datetime
import csv
import json
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


def select_amplitude_peak(rows, center, half_width):
    """Require an interior, resolved peak, rather than a noisy/edge maximum."""
    data = sorted(rows, key=lambda r: r["offset_turns"])
    amp = np.array([r["amplitude_counts"] for r in data])
    noise = np.array([r["amplitude_uncertainty_counts"] for r in data])
    if len(data) < 5 or not np.all(np.isfinite(amp)) or np.min(amp) <= 0:
        raise ValueError("진폭 데이터가 부족하거나 유효하지 않습니다.")
    # The deployed magnitude stream has quantized plateaus. An argmax picks
    # the first equal sample, often an artificial boundary point.
    differences=np.diff(np.unique(amp))
    quantum=float(np.min(differences)) if len(differences) else 0.
    tolerance=max(.5*quantum,2*float(np.max(noise)),np.finfo(float).eps)
    top=np.flatnonzero(amp>=np.max(amp)-tolerance)
    if 0 in top or len(data)-1 in top:
        raise ValueError("진폭 최대 구간이 탐색 경계에 닿습니다. 더 넓은 범위 확인이 필요합니다.")
    if len(top)==0 or np.any(np.diff(top)>1):
        raise ValueError("진폭 최대 구간이 여러 개로 분리되어 있습니다.")
    best=int(min(top,key=lambda i:abs(data[i]["offset_turns"])))
    # Both edges must be measurably lower, allowing the one-bin rounding
    # uncertainty of magnitude counts instead of imposing a 0.1% floor.
    contrast=float(np.max(amp)-max(amp[0],amp[-1]))
    required=max(1.5*quantum,3*float(np.max(noise)),np.finfo(float).eps)
    if contrast <= required:
        raise ValueError("진폭 최대 구간이 잡음/양자화와 구별되지 않습니다.")
    if abs(data[best]["offset_turns"]) > half_width:
        raise ValueError("선택 delay가 허용 범위를 벗어났습니다.")
    return dict(delay_turns=(center+data[best]["offset_turns"]) % 1,
                contrast_counts=float(contrast), peak=data[best],
                selection="nearest original delay within resolved maximum plateau",
                plateau_low_turns=float(data[int(top[0])]["offset_turns"]),
                plateau_high_turns=float(data[int(top[-1])]["offset_turns"]),
                magnitude_quantum_estimate_counts=quantum,tolerance_counts=tolerance)


class ClosedLoopDelayWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, controller, *, half_width=1/128, points=17, settle_s=.4,
                 directory=None):
        super().__init__()
        self.controller = controller
        self.half_width, self.points, self.settle_s = half_width, points, settle_s
        self.directory = Path(directory or Path(__file__).parent/"diagnostics")
        self.cancelled = False

    def stop(self):
        self.cancelled = True

    def _check(self):
        if self.cancelled:
            raise RuntimeError("폐루프 delay 탐색 중지")
        receiver = self.controller.receiver
        if receiver is not None and receiver.error is not None:
            raise RuntimeError(f"UDP 수신 실패: {receiver.error}")
        if self.controller.dropped_local_batches:
            raise RuntimeError("UDP 로컬 큐 누락 발생: 탐색 결과를 적용하지 않습니다.")

    def _wait(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self._check()
            self.msleep(10)

    def _signal(self, parameters, signal):
        self._selected_signal = signal
        # Switch only the selected UDP signal. Do not reconfigure/restart PLL.
        self.controller.stop_stream()
        self.controller.start_stream(signal="unsigned" if signal == 2 else "frequency")
        self.controller.write_named("mode", replace(parameters.mode, signal_of_interest=signal,
                                   enable_agc=False, run=True, send_data_to_pc=True).encode())
        self._wait(self.settle_s)
        self.controller.drain_stream(1000)

    def _samples(self, count=512):
        values, blocks = [], []
        deadline = time.monotonic()+3
        while len(values) < count and time.monotonic() < deadline:
            self._check()
            for batch in self.controller.drain_stream(100):
                if batch.channel != self.controller.channel:
                    continue
                x = np.asarray(batch.values, float)
                if batch.missing_packets or batch.sequence_classification in ("gap", "restart"):
                    raise RuntimeError("UDP 데이터 연속성 오류")
                if len(x):
                    values.extend(x.tolist())
                    blocks.append(float(np.median(x)))
            self.msleep(5)
        if len(values) < count:
            raise TimeoutError("FPGA 데이터 수신 부족")
        v = np.asarray(values[:count])
        if not np.all(np.isfinite(v)):
            raise RuntimeError("유효하지 않은 수신값")
        uncertainty = max(float(np.std(blocks, ddof=1)) if len(blocks)>1 else 0,
                          float(1.4826*np.median(abs(v-np.median(v))))/
                          (1 if self._selected_signal == 0 else np.sqrt(len(v))))
        return float(np.median(v)), uncertainty

    def _point(self, p, offset):
        self._check()
        delay = (p.pll_delay_turns+offset) % 1
        self.controller.write_named("delay", round(delay*(2**32-1)))
        self._signal(p, 0)
        freq, jitter = self._samples()
        margin = .02*(p.maximum_frequency_hz-p.minimum_frequency_hz)
        # Stay near the selected resonance, never jump to a neighboring mode.
        allowed = min(.25*p.frequency_hz/p.resonator_q,
                      .45*(p.maximum_frequency_hz-p.minimum_frequency_hz))
        if not p.minimum_frequency_hz+margin < freq < p.maximum_frequency_hz-margin:
            raise RuntimeError("PLL 주파수가 capture range 경계에 도달했습니다.")
        if abs(freq-p.frequency_hz) > allowed or jitter > allowed/4:
            raise RuntimeError("선택 공진 부근의 안정적인 추적을 확인하지 못했습니다.")
        self._signal(p, 2)
        amplitude, uncertainty = self._samples()
        if amplitude <= 0 or amplitude >= .98*(2**32-1):
            raise RuntimeError("진폭 신호가 0 또는 포화 상태입니다.")
        return dict(offset_turns=float(offset), delay_turns=delay, frequency_hz=freq,
                    frequency_jitter_hz=jitter, amplitude_counts=amplitude,
                    amplitude_uncertainty_counts=uncertainty)

    def run(self):
        p = self.controller.last_parameters
        rows, result, error = [], None, None
        attempts=[]
        selected = None
        try:
            if p is None or not p.mode.run or p.mode.run_nco_at_fixed_frequency:
                raise RuntimeError("PLL tracking이 먼저 실행되어야 합니다.")
            if not 0 < self.half_width <= 1/32 or self.points < 5 or self.points % 2 != 1:
                raise ValueError("탐색 반폭은 0..1/32 turn, 점 수는 5 이상의 홀수여야 합니다.")
            self.settle_s=max(self.settle_s,5*p.resonator_q/(np.pi*p.frequency_hz),
                              5/(2*np.pi*p.loop_bandwidth_hz))
            # AGC off during comparison, drive/NCO gain unchanged.
            width=self.half_width
            for attempt in range(3):
                rows=[]
                reference = self._point(p, 0)
                offsets=np.linspace(-width,width,self.points)
                # Alternate around center to limit time-drift bias.
                for i,offset in enumerate(sorted(offsets,key=lambda x:(abs(x),x))):
                    self.progress.emit(f"폐루프 delay 탐색 {attempt+1}회 {i+1}/{self.points} · ±{width:.6f} turn")
                    rows.append(self._point(p,float(offset)))
                end_reference=self._point(p,0)
                attempts.append(dict(half_width_turns=width,rows=rows,
                                     reference=reference,end_reference=end_reference))
                if abs(end_reference["amplitude_counts"]-reference["amplitude_counts"]) > max(
                        .01*reference["amplitude_counts"],
                        6*max(reference["amplitude_uncertainty_counts"],end_reference["amplitude_uncertainty_counts"])):
                    raise RuntimeError("탐색 전후 진폭 drift가 커서 최대점을 신뢰할 수 없습니다.")
                try:
                    result=select_amplitude_peak(rows,p.pll_delay_turns,width)
                    break
                except ValueError as exc:
                    attempts[-1]["selection_error"]=str(exc)
                    if attempt==2 or width>=1/32: raise
                    width=min(2*width,1/32)
            result["search_half_width_turns"]=width
            offset = (result["delay_turns"]-p.pll_delay_turns+.5) % 1-.5
            verify = self._point(p, offset)
            if verify["amplitude_counts"] < result["peak"]["amplitude_counts"]-max(
                    .005*result["peak"]["amplitude_counts"], 4*verify["amplitude_uncertainty_counts"]):
                raise RuntimeError("선택 진폭 최대점 재측정 불일치")
            selected = result["delay_turns"]
            result.update(reference=reference, end_reference=end_reference, verification=verify)
            result["context"]=dict(center_hz=p.frequency_hz,cic_rate=p.cic_rate,
                                   drive_fraction=p.drive_amplitude_fraction,agc=p.mode.enable_agc)
        except Exception as exc:
            error = str(exc)
        finally:
            if p is not None:
                try:
                    self.controller.stop_stream()
                    self.controller.write_named("delay", round((selected if selected is not None else p.pll_delay_turns)*(2**32-1)))
                    restored_mode=replace(p.mode,run=False,send_data_to_pc=False) if self.cancelled else p.mode
                    self.controller.write_named("mode", restored_mode.encode())
                    self.controller.last_parameters = replace(p,mode=restored_mode,
                        pll_delay_turns=selected if selected is not None else p.pll_delay_turns)
                    if self.cancelled:
                        self.controller.stop_output()
                    else:
                        self.controller.start_stream(signal="frequency")
                except Exception as exc:
                    error = f"PLL delay/mode 복구 실패: {exc}"
                    try:
                        self.controller.stop_output()
                    except Exception as stop_exc:
                        error += f"; 출력 정지 실패: {stop_exc}"
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                stem = self.directory/f"fpga_delay_peak_{datetime.now():%Y%m%d_%H%M%S_%f}"
                metadata = dict(accepted=error is None, error=error, result=result, rows=rows,attempts=attempts,
                                agc_during_search=False, original_agc=p.mode.enable_agc if p else None,
                                original_delay=p.pll_delay_turns if p else None,
                                center_frequency_hz=p.frequency_hz if p else None,
                                cic_rate=p.cic_rate if p else None,
                                drive_fraction=p.drive_amplitude_fraction if p else None,
                                resonator_q=p.resonator_q if p else None,
                                target_bandwidth_hz=p.loop_bandwidth_hz if p else None,
                                half_width_turns=self.half_width,points=self.points,
                                settle_s=self.settle_s,
                                sampling="sequential frequency and unsigned magnitude; no simultaneous sampling")
                stem.with_suffix(".json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
                if rows:
                    with stem.with_suffix(".csv").open("w",newline="",encoding="utf-8-sig") as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                if result:
                    result["path"] = str(stem.with_suffix(".json"))
            except Exception as exc:
                error = error or f"탐색 기록 저장 실패: {exc}"
        if error:
            self.failed.emit(error)
        else:
            self.completed.emit(result)
