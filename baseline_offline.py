"""Hardware-independent loading, analysis and export for baseline noise CSVs."""
from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from baseline_analysis import (
    EVENT_WINDOWS_S, correlations, linear_baseline, timing_quality,
)
from allan_export import save_allan_export


def offline_rolling(t, f, window_s):
    """Linear-time equivalent of rolling_metrics for large, ordered recordings.

    Remove the global linear trend before forming moments to keep small noise
    variances accurate even when the carrier frequency or drift is large.
    """
    n = len(t)
    base = linear_baseline(t, f)
    residual = base["residual_hz"]
    slope = base["drift_hz_per_min"] / 60
    left = np.searchsorted(t, t - window_s, side="left")
    count = np.arange(1, n + 1) - left

    def sums(values):
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        return prefix[1:] - prefix[left]

    mt, mr = sums(t) / count, sums(residual) / count
    vt = np.maximum(sums(t * t) / count - mt * mt, 0)
    vr = np.maximum(sums(residual * residual) / count - mr * mr, 0)
    cov = sums(t * residual) / count - mt * mr
    valid = (count >= 3) & (vt > 0)
    local_slope = np.divide(cov, vt, out=np.zeros(n), where=valid)
    detrended_var = np.maximum(vr - cov * local_slope, 0)
    raw_var = np.maximum(vr + slope * slope * vt + 2 * slope * cov, 0)
    # Variable-duration windows require monotonic queues, not sample windows.
    low, high, peak_to_peak = deque(), deque(), np.empty(n)
    for i, value in enumerate(f):
        while low and low[0] < left[i]:
            low.popleft()
        while high and high[0] < left[i]:
            high.popleft()
        while low and f[low[-1]] >= value:
            low.pop()
        while high and f[high[-1]] <= value:
            high.pop()
        low.append(i)
        high.append(i)
        peak_to_peak[i] = f[high[0]] - f[low[0]]
    metrics = {
        "count": count.astype(float),
        "mean_hz": base["trend_hz"][0] + slope * (mt - t[0]) + mr,
        "std_hz": np.sqrt(raw_var), "drift_hz_per_min": (slope + local_slope) * 60,
        "detrended_std_hz": np.sqrt(detrended_var), "peak_to_peak_hz": peak_to_peak,
    }
    summary = {}
    for key, values in metrics.items():
        if key != "count":
            values[~valid] = np.nan
        finite = values[np.isfinite(values)]
        if len(finite):
            summary[key] = dict(zip(("median", "p05", "p95", "max"),
                                   map(float, (np.median(finite), np.percentile(finite, 5),
                                               np.percentile(finite, 95), np.max(finite)))))
    return {"time_s": t, "window_s": float(window_s), "metrics": metrics,
            "edge_policy": "trailing causal window; insufficient samples=NaN", "summary": summary}


def offline_allan(t, f):
    """Same timestamp-bin estimator as baseline_analysis, using grouped reductions."""
    rows = []
    if len(t) < 4:
        return rows
    for tau in EVENT_WINDOWS_S + (2., 5., 10., 30., 60., 300.):
        if tau <= 0:
            continue
        bins = np.floor((t - t[0]) / tau)
        starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
        counts = np.diff(np.r_[starts, len(t)])
        # Subtracting the carrier avoids loss of precision in bin sums.
        means = (np.add.reduceat(f - f[0], starts) / counts)[counts >= 2]
        if len(means) >= 2:
            rows.append({"tau_s": float(tau),
                         "allan_deviation_hz": float(np.sqrt(.5 * np.mean(np.diff(means) ** 2))),
                         "cluster_count": len(means),
                         "method": "non-overlapping timestamp bins; no resampling"})
    return rows


def load_noise_csv(path):
    """Accept Plotter/FPGA raw CSVs; amplitude and phase are optional."""
    path = Path(path).resolve()
    metadata, columns = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = None
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].startswith("#"):
                if row[0].strip() == "# Metadata" and len(row) >= 3:
                    metadata[row[1]] = row[2]
                continue
            if header is None:
                header = [cell.strip() for cell in row]
                missing = {"Time_s", "Frequency_Hz"} - set(header)
                if missing:
                    raise ValueError("필수 CSV 열 누락: " + ", ".join(sorted(missing)))
                columns = {key: [] for key in (
                    "Time_s", "Frequency_Hz", "Amplitude_R_V", "Phase_deg", "PLL_PhaseError_deg"
                ) if key in header}
                indexes = {key: header.index(key) for key in columns}
                continue
            for key, values in columns.items():
                index = indexes[key]
                cell = row[index].strip() if index < len(row) else ""
                try:
                    values.append(float(cell) if cell else np.nan)
                except ValueError as exc:
                    raise ValueError(f"CSV {line}행 {key}: 숫자가 아닙니다 ({cell!r}).") from exc
    if not columns or len(columns["Time_s"]) < 3:
        raise ValueError("분석하려면 최소 3개 샘플이 필요합니다.")
    arrays = {key: np.asarray(value, dtype=float) for key, value in columns.items()}
    t, f = arrays["Time_s"], arrays["Frequency_Hz"]
    if not np.all(np.isfinite(t) & np.isfinite(f)):
        raise ValueError("Time_s / Frequency_Hz에 빈 값 또는 비유한 값이 있습니다.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time_s는 중복 없이 증가해야 합니다. 원본 시간 열을 확인하세요.")
    return {"path": str(path), "metadata": metadata, "columns": arrays}


def analyze_noise(source, start_s=None, end_s=None, rolling_s=10.0):
    if not np.isfinite(rolling_s) or rolling_s <= 0:
        raise ValueError("Rolling window는 0보다 커야 합니다.")
    raw = source["columns"]
    time = raw["Time_s"]
    start = float(time[0] if start_s is None else start_s)
    end = float(time[-1] if end_s is None else end_s)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("분석 끝 시간은 시작 시간보다 커야 합니다.")
    mask = (time >= start) & (time <= end)
    columns = {key: values[mask].copy() for key, values in raw.items()}
    t, f = columns["Time_s"], columns["Frequency_Hz"]
    if len(t) < 3:
        raise ValueError("선택 구간에 최소 3개 샘플이 필요합니다.")
    # Relative fitting coordinates avoid loss of precision with absolute timestamps.
    relative = t - t[0]
    baseline = linear_baseline(relative, f)
    baseline["time_s"] = t.copy()
    rolling = offline_rolling(relative, f, rolling_s)
    rolling["time_s"] = t.copy()
    absent = np.full(len(t), np.nan)
    try:
        nominal = float(source["metadata"].get("Plotter_requested_interval_ms", "nan")) / 1000
    except (TypeError, ValueError):
        nominal = np.nan
    quality = timing_quality(t, nominal if np.isfinite(nominal) and nominal > 0 else None)
    # Reuse the online GUI's timestamp-bin estimator and expose its limitations.
    allan = offline_allan(relative, f)
    return {
        "source_path": source["path"], "metadata": dict(source["metadata"]),
        "settings": {"start_s": start, "end_s": end, "rolling_s": float(rolling_s)},
        "columns": columns, "global": baseline, "timing": quality, "rolling": rolling,
        "allan": allan,
        "correlation": correlations(f, columns.get("Amplitude_R_V", absent),
                                    columns.get("Phase_deg", absent), baseline["residual_hz"]),
        "processing": "Raw frequency; linear detrending; no smoothing or resampling. "
                      "Allan uses raw frequency and the existing non-overlapping timestamp-bin estimator; "
                      "bins with fewer than two samples are omitted. Gaps can affect Allan estimates.",
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_noise_result(result, parent):
    """Create a unique result directory so source files cannot be overwritten."""
    folder = Path(parent) / ("baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                             + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in result.items()
               if key not in ("columns", "global", "rolling")}
    summary["global"] = {key: value for key, value in result["global"].items()
                         if not isinstance(value, np.ndarray)}
    summary["rolling"] = {"window_s": result["rolling"]["window_s"],
                          "edge_policy": result["rolling"]["edge_policy"],
                          "summary": result["rolling"]["summary"]}
    with (folder / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    columns = dict(result["columns"])
    columns["LinearTrend_Hz"] = result["global"]["trend_hz"]
    columns["DetrendedFrequency_Hz"] = result["global"]["residual_hz"]
    for key, values in result["rolling"]["metrics"].items():
        columns["Rolling_" + key] = values
    with (folder / "derived.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    if result["allan"]:
        save_allan_export(folder / "allan.csv", result["allan"])
    return folder
