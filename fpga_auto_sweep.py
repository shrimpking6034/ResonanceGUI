"""Sequential theory-guided FPGA sweeps, independent of GUI widget state."""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from scipy.signal import find_peaks, peak_widths, savgol_filter

from fpga_resonance_gui import SweepWorker
from resonance_fit import three_db_bounds, fit_complex_resonance, assess_fit_quality, model_response


@dataclass(frozen=True)
class AutoSweepSettings:
    first_mode: int = 1
    last_mode: int = 8
    coarse_percent: float = 10.0
    coarse_half_hz: float = 0.0  # zero selects percentage
    coarse_step_hz: float = 10.0
    fine_step_hz: float = 1.0
    fallback_half_hz: float = 100.0
    linewidth_factor: float = 3.0
    coarse_settle_s: float = 0.1
    fine_settle_s: float = 0.1
    coarse_samples: int = 512
    fine_samples: int = 512
    max_points: int = 100000

    def validate(self):
        if not 1 <= self.first_mode <= self.last_mode <= 30:
            raise ValueError("모드 범위는 1 ≤ 시작 ≤ 종료 ≤ 30이어야 합니다.")
        for key in ("coarse_percent", "coarse_step_hz", "fine_step_hz",
                    "fallback_half_hz", "linewidth_factor", "coarse_settle_s", "fine_settle_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key}: 양의 유한값이 필요합니다.")
        if not math.isfinite(self.coarse_half_hz) or self.coarse_half_hz < 0:
            raise ValueError("Coarse 반폭은 0 이상이어야 합니다.")
        if min(self.coarse_settle_s, self.fine_settle_s) < 0.05:
            raise ValueError("FPGA 정착시간은 최소 0.05 s입니다.")
        if min(self.coarse_samples, self.fine_samples) < 1 or self.max_points < 9:
            raise ValueError("평균 샘플 수와 최대 포인트 수를 확인하세요.")
        if min(self.coarse_step_hz, self.fine_step_hz) < 12500000 / 2**32:
            raise ValueError("주파수 간격이 NCO 분해능보다 작습니다.")


def frequency_grid(low, high, step, max_points=100000):
    """Exact requested spacing; do not append an irregular last interval."""
    if not all(math.isfinite(x) for x in (low, high, step)) or not 0 < low < high < 6250000 or step <= 0:
        raise ValueError("유효한 주파수 범위와 간격이 필요합니다.")
    count = int(math.floor((high - low) / step + 1e-9)) + 1
    if count < 5 or count > max_points:
        raise ValueError(f"스윕 포인트 {count:,}개: 간격/범위를 조정하세요 (5~{max_points:,}).")
    return low + np.arange(count) * step


def build_mode_plans(modes, settings):
    settings.validate()
    ordered = sorted(modes, key=lambda m: m.index)
    frequencies = np.array([m.frequency_hz for m in ordered])
    if not len(frequencies) or not np.all(np.isfinite(frequencies)) or not np.all(np.diff(frequencies) > 0):
        raise ValueError("이론 모드 주파수가 유효하고 오름차순이어야 합니다.")
    plans = []
    for i, mode in enumerate(ordered):
        if not settings.first_mode <= mode.index <= settings.last_mode:
            continue
        f = float(mode.frequency_hz)
        # Missing end neighbors use half the adjacent spacing, not infinity.
        left_gap = f - frequencies[i-1] if i else f
        right_gap = frequencies[i+1] - f if i+1 < len(ordered) else left_gap
        allowed_low = max(1.0, f - left_gap / 2)
        allowed_high = min(6249999.0, f + right_gap / 2)
        half = settings.coarse_half_hz or f * settings.coarse_percent / 100
        low, high = max(allowed_low, f-half), min(allowed_high, f+half)
        grid = frequency_grid(low, high, settings.coarse_step_hz, settings.max_points)
        plans.append(dict(mode=mode.index, theory_hz=f, low=low, high=high,
                          allowed_low=allowed_low, allowed_high=allowed_high, points=len(grid)))
    if [p["mode"] for p in plans] != list(range(settings.first_mode, settings.last_mode+1)):
        raise ValueError("선택한 차수까지 이론 모드를 계산해야 합니다.")
    return plans


def peak_candidates(data, target, low, high):
    """Short smoothing preserves narrow peaks in large coarse scans.

    Candidates are evidence, not proof of a mechanical mode assignment.
    """
    f, r = np.asarray(data["freqs"]), np.asarray(data["R"])
    if len(f) < 5 or not np.all(np.isfinite(r)):
        return [], r
    smooth = savgol_filter(r, 5, 2)
    residual = r - smooth
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    threshold = max(5*noise, float(np.ptp(smooth))*0.03, np.finfo(float).eps)
    indices, properties = find_peaks(smooth, prominence=threshold, distance=2)
    candidates = []
    if len(indices):
        half_prominence_widths = peak_widths(smooth, indices, rel_height=0.5)[0]
        spacing = float(np.median(np.diff(f)))
    else:
        half_prominence_widths = []
        spacing = 0.0
    for j, index in enumerate(indices):
        if not low <= f[index] <= high:
            continue
        prominence = float(properties["prominences"][j])
        bounds = three_db_bounds(f, smooth, index)
        # Absolute -3 dB is only meaningful if the baseline is below that level.
        baseline = max(float(smooth[properties["left_bases"][j]]),
                       float(smooth[properties["right_bases"][j]]))
        reliable = bounds is not None and baseline < 0.85 * smooth[index]
        width = float(bounds[1]-bounds[0]) if reliable else None
        candidates.append(dict(index=int(index), frequency_hz=float(f[index]),
                               peak_height=float(smooth[index]), prominence=prominence,
                               prominence_width_hz=float(half_prominence_widths[j]*spacing),
                               noise_sigma=noise,
                               linewidth_hz=width, bounds=list(bounds) if bounds else None,
                               edge=bool(index <= 2 or index >= len(f)-3)))
    rank_peak_candidates(candidates, target, low, high)
    return candidates, smooth


def rank_peak_candidates(candidates, target, low, high):
    """Rank measured resonances; theory proximity is a modest tie-breaker.

    Scores are meaningful only among candidates from the same broad scan.
    A short local retry must not replace the broad scan's evidence.
    """
    if not candidates:
        return candidates
    height_scale = max(c["peak_height"] for c in candidates)
    prominence_scale = max(c["prominence"] for c in candidates)
    distance_scale = max(target-low, high-target, 1.0)
    for candidate in candidates:
        height = max(candidate["peak_height"], 0) / max(height_scale, np.finfo(float).eps)
        prominence = candidate["prominence"] / max(prominence_scale, np.finfo(float).eps)
        distance = abs(candidate["frequency_hz"]-target) / distance_scale
        candidate["score"] = float(np.sqrt(height*prominence) / (1 + 0.4*distance))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def reconcile_coarse_candidates(original, retry, target, low, high,
                                retry_low, retry_high, coarse_step, expanded):
    """Keep broad-scan prominence when retry only covers a short window."""
    if expanded:
        return rank_peak_candidates(retry, target, retry_low, retry_high)
    if not original:
        return rank_peak_candidates(retry, target, retry_low, retry_high)
    best = original[0]
    nearby = [c for c in retry if abs(c["frequency_hz"]-best["frequency_hz"]) <= 3*coarse_step]
    if nearby:
        refined = min(nearby, key=lambda c: abs(c["frequency_hz"]-best["frequency_hz"]))
        best.update(frequency_hz=refined["frequency_hz"],
                    linewidth_hz=refined["linewidth_hz"] or best["linewidth_hz"],
                    bounds=refined["bounds"] or best["bounds"], edge=refined["edge"])
    return rank_peak_candidates(original, target, low, high)


def fine_window(candidate, candidates, plan, settings, coarse_step):
    center = candidate["frequency_hz"]
    width = candidate["linewidth_hz"]
    estimated_width = width or candidate.get("prominence_width_hz")
    half = max(settings.linewidth_factor * estimated_width if estimated_width else settings.fallback_half_hz,
               settings.fallback_half_hz if width is None else 0,
               4*coarse_step, 10*settings.fine_step_hz)
    low, high = plan["allowed_low"], plan["allowed_high"]
    for neighbor in candidates:
        f = neighbor["frequency_hz"]
        if f < center:
            low = max(low, (f+center)/2)
        elif f > center:
            high = min(high, (f+center)/2)
    a, b = max(low, center-half), min(high, center+half)
    return a, b, low, high, (a > center-half or b < center+half)


def write_json(path, value):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


SUMMARY_FIELDS = ("mode", "theory_hz", "status", "peak_hz", "f0_hz", "q", "phase_deg",
                  "deviation_percent", "fine_low_hz", "fine_high_hz", "fit_accepted", "notes")


class AutoModeSweepWorker(QThread):
    stage_started = pyqtSignal(object)
    point = pyqtSignal(object)
    stage_done = pyqtSignal(object)
    mode_done = pyqtSignal(object)
    completed = pyqtSignal(object)

    def __init__(self, controller, parameters, modes, settings, directory, context=None):
        super().__init__()
        self.controller, self.parameters = controller, parameters
        self.settings = settings
        self.plans = build_mode_plans(modes, settings)
        self.directory = Path(directory)
        self.context = context or {}
        self.stop_requested = False
        self.active_segment = None
        self.records = []
        self.outcome = None

    def stop(self):
        self.stop_requested = True
        segment = self.active_segment
        if segment is not None:
            segment.stop()

    def _save_summary(self, state, error=None):
        write_json(self.directory / "session.json", dict(status=state, error=error, records=self.records))
        temporary = self.directory / "summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow({**record, "notes": "; ".join(record["notes"])})
        temporary.replace(self.directory / "summary.csv")

    def _measure(self, record, stage, low, high, step, fine=False):
        grid = frequency_grid(low, high, step, self.settings.max_points)
        settle = self.settings.fine_settle_s if fine else self.settings.coarse_settle_s
        samples = self.settings.fine_samples if fine else self.settings.coarse_samples
        params = replace(self.parameters, frequency_hz=float(grid[0]),
                         minimum_frequency_hz=float(grid[0]), maximum_frequency_hz=float(grid[-1]))
        segment = SweepWorker(self.controller, params, grid, settle, samples)
        self.active_segment = segment
        if self.stop_requested:
            segment.stop()
        meta = dict(mode=record["mode"], stage=stage, total=len(grid),
                    low_hz=low, high_hz=float(grid[-1]), step_hz=step,
                    settle_s=settle, samples=samples)
        self.stage_started.emit(meta)
        count = 0

        def on_point(f, r, theta, x, y):
            nonlocal count
            count += 1
            self.point.emit(dict(mode=record["mode"], stage=stage, count=count,
                                 total=len(grid), values=(f, r, theta, x, y)))

        segment.point.connect(on_point, Qt.DirectConnection)
        error = None
        try:
            segment.measure()
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self.active_segment = None
            data = segment.last_result
            stem = f"mode_{record['mode']:02d}_{stage}"
            meta.update(status="failed" if error else "stopped" if self.stop_requested else "complete", error=error)
            if data:
                np.savez_compressed(self.directory / (stem + ".npz"), **data)
                with (self.directory / (stem + ".csv")).open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.writer(handle)
                    for key, value in meta.items():
                        writer.writerow(["# Metadata", key, value])
                    writer.writerow(["# Metadata", "Drive_fraction", self.parameters.drive_amplitude_fraction])
                    writer.writerow(["# Metadata", "Theory_Hz", record["theory_hz"]])
                    keys = ("freqs", "R", "theta", "X", "Y", "X_sem", "Y_sem", "XY_cov", "n_samples", "requested_freqs")
                    writer.writerow(["Frequency_Hz", "Amplitude_R_counts", "Phase_deg", "X_counts", "Y_counts",
                                     "X_SEM_counts", "Y_SEM_counts", "XY_cov_counts2", "N_samples", "Requested_frequency_Hz"])
                    writer.writerows(zip(*(data[k] for k in keys)))
            record["stages"].append(dict(**meta, file=stem))
            self._save_summary("running")
            self.stage_done.emit(dict(**meta, file=stem, data=data))
        return data

    def _mode(self, record, plan):
        s = self.settings
        data = self._measure(record, "coarse", plan["low"], plan["high"], s.coarse_step_hz)
        if self.stop_requested:
            return
        step = s.coarse_step_hz
        candidates, _ = peak_candidates(data, plan["theory_hz"], plan["allowed_low"], plan["allowed_high"])
        # One bounded coarse retry: expand missing/edge peaks, or resolve a
        # narrow/unmeasurable peak locally at a smaller interval.
        best = candidates[0] if candidates else None
        previous = candidates
        if best is None or best["edge"]:
            low, high = plan["allowed_low"], plan["allowed_high"]
        elif best["linewidth_hz"] is None or best["linewidth_hz"] < 4*step:
            half = max(10*step, s.fallback_half_hz)
            low, high = max(plan["allowed_low"], best["frequency_hz"]-half), min(plan["allowed_high"], best["frequency_hz"]+half)
        else:
            low = high = None
        if low is not None:
            step = max(s.fine_step_hz, step/2)
            data = self._measure(record, "coarse_retry", low, high, step)
            if self.stop_requested:
                return
            retry_candidates, _ = peak_candidates(data, plan["theory_hz"], low, high)
            candidates = reconcile_coarse_candidates(
                previous, retry_candidates, plan["theory_hz"], plan["low"], plan["high"],
                low, high, s.coarse_step_hz, best is None or best["edge"])
        record["candidates"] = candidates
        if not candidates:
            record["status"] = "not_found"
            record["notes"].append("재측정 후 유효 피크 없음")
            return
        best = candidates[0]
        if len(candidates) > 1 and candidates[1]["score"] >= 0.85*best["score"]:
            record["status"] = "ambiguous"
            record["notes"].append("비슷한 피크 후보가 여러 개: 자동 모드 배정 보류")
            return
        if best["edge"]:
            record["status"] = "edge_peak"
            record["notes"].append("허용 탐색 영역 경계의 피크")
            return
        record["peak_hz"] = best["frequency_hz"]
        low, high, limit_low, limit_high, clipped = fine_window(best, candidates, plan, s, step)
        if not best["linewidth_hz"]:
            record["notes"].append("선폭 추정 불가: 예비 반폭 사용")
        if clipped:
            record["notes"].append("인접 피크/모드 경계로 범위 제한")
        fine = self._measure(record, "fine", low, high, s.fine_step_hz, True)
        if self.stop_requested:
            return
        fine_candidates, _ = peak_candidates(fine, best["frequency_hz"], low, high)
        selected = min(fine_candidates, key=lambda c: abs(c["frequency_hz"]-best["frequency_hz"]), default=None)
        if selected is None or selected["edge"] or selected["bounds"] is None:
            center = selected["frequency_hz"] if selected else best["frequency_hz"]
            half = max(high-low, 10*s.fine_step_hz)
            new_low, new_high = max(limit_low, center-half), min(limit_high, center+half)
            if new_low < low-s.fine_step_hz or new_high > high+s.fine_step_hz:
                low, high = new_low, new_high
                fine = self._measure(record, "fine_retry", low, high, s.fine_step_hz, True)
                if self.stop_requested:
                    return
                fine_candidates, _ = peak_candidates(fine, best["frequency_hz"], low, high)
                selected = min(fine_candidates, key=lambda c: abs(c["frequency_hz"]-best["frequency_hz"]), default=None)
        record.update(fine_low_hz=float(fine["freqs"][0]), fine_high_hz=float(fine["freqs"][-1]))
        if selected is None:
            record["status"] = "fine_not_found"
            return
        if selected["edge"] or selected["bounds"] is None:
            record["notes"].append("Fine에서도 양쪽 공진 폭 미확보")
        record["peak_hz"] = selected["frequency_hz"]
        try:
            fit = fit_complex_resonance(fine["freqs"], fine["X"], fine["Y"], fine["X_sem"], fine["Y_sem"],
                                        f0_guess=selected["frequency_hz"], f0_bounds=(low, high),
                                        cancel_check=lambda: self.stop_requested, max_nfev=500)
            quality = assess_fit_quality(fine["freqs"], fine["X"], fine["Y"], fit)
            accepted = bool(fit.success and quality.accepted and not selected["edge"] and selected["bounds"] is not None)
            record.update(f0_hz=fit.f0_hz, q=fit.q, phase_deg=fit.phase_deg,
                          deviation_percent=100*(fit.f0_hz/plan["theory_hz"]-1), fit_accepted=accepted,
                          status="complete" if accepted else "fit_review")
            record["notes"].extend(quality.reasons)
            record["fit"] = asdict(fit)
            record["fit_quality"] = asdict(quality)
            z = model_response(fine["freqs"], fit.parameters, float(np.mean(fine["freqs"])), fit.resonance_sign)
            np.savez_compressed(self.directory / f"mode_{plan['mode']:02d}_fit.npz", freqs=fine["freqs"], R=abs(z), theta=np.angle(z, deg=True))
        except Exception as exc:
            record["status"] = "fit_failed"
            record["notes"].append(str(exc))

    def run(self):
        state, error = "complete", None
        started = time.monotonic()
        created = False
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            created = True
            write_json(self.directory / "settings.json", dict(peak_selection_version=2,
                       settings=asdict(self.settings), plans=self.plans,
                       parameters=asdict(self.parameters), context=self.context,
                       backend=type(self.controller).__name__, created=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            self.records = [dict(mode=p["mode"], theory_hz=p["theory_hz"], status="pending", notes=[], stages=[]) for p in self.plans]
            self._save_summary("running")
            for record, plan in zip(self.records, self.plans):
                if self.stop_requested:
                    break
                record["status"] = "running"
                try:
                    self._mode(record, plan)
                except Exception as exc:
                    record["status"] = "failed"
                    record["notes"].append(str(exc))
                    raise
                if self.stop_requested:
                    record["status"] = "stopped"
                self._save_summary("running")
                self.mode_done.emit(dict(record))
            if self.stop_requested:
                state = "stopped"
        except Exception as exc:
            state, error = "failed", str(exc)
        finally:
            # Segment cleanup already ran, but a final explicit stop also covers
            # errors before acquisition and is reported rather than suppressed.
            try:
                self.controller.stop_output()
            except Exception as exc:
                state, error = "failed", f"{error or ''} 출력 정지 확인 실패: {exc}"
            try:
                stop_stream = getattr(self.controller, "stop_stream", None)
                if stop_stream is not None:
                    stop_stream()
            except Exception as exc:
                state, error = "failed", f"{error or ''} 스트림 정지 실패: {exc}"
            if created:
                try:
                    self._save_summary(state, error)
                except Exception as exc:
                    state, error = "failed", f"{error or ''} 저장 실패: {exc}"
            self.outcome = dict(status=state, error=error, directory=str(self.directory),
                                records=self.records, elapsed_s=time.monotonic()-started)
            self.completed.emit(self.outcome)


def load_auto_session(directory):
    directory = Path(directory)
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    stages = []
    for record in session["records"]:
        for stage in record["stages"]:
            # Session metadata is data, never a path outside its session.
            stem = stage["file"]
            if Path(stem).name != stem or "/" in stem or "\\" in stem:
                raise ValueError("잘못된 스윕 파일 이름")
            with np.load(directory / (stem + ".npz"), allow_pickle=False) as arrays:
                stages.append(dict(stage, data={key: arrays[key].copy() for key in arrays.files}))
    return session, stages
