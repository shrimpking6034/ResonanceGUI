"""Timestamp-aware, raw-preserving analysis for resonator baseline/event data.

All functions return derived arrays/new dictionaries only.  They never alter a
caller supplied raw series, resample only when explicitly requested, and use
the actual time axis for every window calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import csv
import numpy as np
from scipy.stats import chi2


EVENT_WINDOWS_S = (0.0, 0.05, 0.10, 0.15, 0.20, 0.50, 1.00)
DETECTION_DURATIONS_S = (0.30, 0.50, 1.00)
ROLLING_WINDOWS_S = (10.0, 30.0, 60.0, 300.0)


def _finite_time_series(t, *series):
    t = np.asarray(t, dtype=float)
    values = [np.asarray(v, dtype=float) for v in series]
    mask = np.isfinite(t)
    for value in values:
        mask &= np.isfinite(value)
    return t[mask], [value[mask] for value in values], mask


def timing_quality(t: Iterable[float], nominal_interval_s: float | None = None):
    raw = np.asarray(t, dtype=float)
    finite = raw[np.isfinite(raw)]
    dt = np.diff(finite)
    positive = dt[dt > 0]
    result = {
        "sample_count": int(len(raw)), "finite_timestamp_count": int(len(finite)),
        "measurement_duration_s": float(finite[-1] - finite[0]) if len(finite) >= 2 else np.nan,
        "nominal_interval_s": nominal_interval_s if nominal_interval_s else np.nan,
        "nominal_rate_hz": (1.0 / nominal_interval_s) if nominal_interval_s and nominal_interval_s > 0 else np.nan,
        "duplicate_timestamps": int(np.count_nonzero(dt == 0)),
        "non_monotonic_timestamps": int(np.count_nonzero(dt < 0)),
    }
    if not len(positive):
        return result | {k: np.nan for k in ("dt_mean_s", "dt_median_s", "dt_std_s", "dt_min_s", "dt_max_s", "dt_p95_s", "dt_p99_s", "dt_cv", "effective_rate_hz", "mean_rate_hz", "long_gap_threshold_s", "long_gap_count")}
    median = float(np.median(positive))
    duration = result["measurement_duration_s"]
    result |= {
        "dt_mean_s": float(np.mean(positive)), "dt_median_s": median,
        "dt_std_s": float(np.std(positive)), "dt_min_s": float(np.min(positive)),
        "dt_max_s": float(np.max(positive)), "dt_p95_s": float(np.percentile(positive, 95)),
        "dt_p99_s": float(np.percentile(positive, 99)),
        "dt_cv": float(np.std(positive) / np.mean(positive)),
        "effective_rate_hz": 1.0 / median,
        "mean_rate_hz": ((len(finite) - 1) / duration) if duration > 0 else np.nan,
        "long_gap_threshold_s": max(3.0 * median, 0.2),
    }
    result["long_gap_count"] = int(np.count_nonzero(positive > result["long_gap_threshold_s"]))
    return result


def linear_baseline(t, frequency):
    t, (f,), _ = _finite_time_series(t, frequency)
    if len(t) < 2 or np.ptp(t) <= 0:
        return {"valid": False, "reason": "insufficient samples", "trend_hz": np.full(len(t), np.nan), "residual_hz": np.full(len(t), np.nan)}
    slope, intercept = np.polyfit(t, f, 1)
    trend = slope * t + intercept
    residual = f - trend
    return {"valid": True, "time_s": t, "trend_hz": trend, "residual_hz": residual,
            "mean_hz": float(np.mean(f)), "raw_std_hz": float(np.std(f)),
            "peak_to_peak_hz": float(np.ptp(f)), "drift_hz_per_min": float(slope * 60.0),
            "detrended_std_hz": float(np.std(residual)), "threshold_3sigma_hz": float(3*np.std(residual)),
            "threshold_5sigma_hz": float(5*np.std(residual))}


def rolling_metrics(t, frequency, window_s, min_samples=3):
    t, (f,), _ = _finite_time_series(t, frequency)
    n = len(t)
    output = {name: np.full(n, np.nan) for name in ("count", "mean_hz", "std_hz", "drift_hz_per_min", "detrended_std_hz", "peak_to_peak_hz")}
    for i, now in enumerate(t):
        left = np.searchsorted(t, now - window_s, side="left")
        tt, ff = t[left:i+1], f[left:i+1]
        output["count"][i] = len(tt)
        if len(tt) < min_samples or np.ptp(tt) <= 0:
            continue
        slope, intercept = np.polyfit(tt, ff, 1)
        residual = ff - (slope * tt + intercept)
        output["mean_hz"][i] = np.mean(ff); output["std_hz"][i] = np.std(ff)
        output["drift_hz_per_min"][i] = slope * 60.0
        output["detrended_std_hz"][i] = np.std(residual); output["peak_to_peak_hz"][i] = np.ptp(ff)
    summary = {}
    for name, values in output.items():
        valid = values[np.isfinite(values)]
        if len(valid): summary[name] = {"median": float(np.median(valid)), "p05": float(np.percentile(valid, 5)), "p95": float(np.percentile(valid, 95)), "max": float(np.max(valid))}
    return {"time_s": t, "window_s": float(window_s), "edge_policy": "trailing causal window; insufficient samples=NaN", "metrics": output, "summary": summary}


def event_timescale_noise(t, frequency, duration_s, min_samples=2, stride_s=None):
    """Causal window minima and local detrended noise; no interpolation."""
    t, (f,), _ = _finite_time_series(t, frequency)
    rows = []
    if len(t) < min_samples: return {"valid": False, "reason": "insufficient samples", "rows": []}
    stride_s = duration_s if stride_s is None else max(float(stride_s), 1e-12)
    starts = np.arange(t[0], t[-1] + 1e-12, stride_s)
    for start in starts:
        mask = (t >= start) & (t < start + duration_s)
        tt, ff = t[mask], f[mask]
        if len(tt) < min_samples: continue
        baseline = linear_baseline(tt, ff)
        rows.append({"start_s": float(start), "effective_duration_s": float(tt[-1]-tt[0]), "sample_count": int(len(tt)),
                     "min_frequency_hz": float(np.min(ff)), "mean_frequency_hz": float(np.mean(ff)),
                     "std_hz": float(np.std(ff)), "detrended_std_hz": baseline.get("detrended_std_hz", np.nan)})
    return {"valid": bool(rows), "duration_s": float(duration_s), "processing": "raw causal sliding minima; actual timestamps; no interpolation", "stride_s": stride_s, "rows": rows}


def local_event_metrics(t, frequency, events, baseline_s=0.4, guard_s=0.05,
                        direction="negative", baseline_method="median", min_samples=2):
    """Calculate explicit event intervals against the immediately preceding baseline."""
    t, (f,), _ = _finite_time_series(t, frequency); output=[]
    for event_id, (start, end) in enumerate(events, 1):
        event_mask=(t>=start)&(t<=end); base_mask=(t>=start-guard_s-baseline_s)&(t<start-guard_s)
        tt, ff=t[event_mask],f[event_mask]; bt,bf=t[base_mask],f[base_mask]
        row={"EventID":event_id,"EventStart_s":float(start),"EventEnd_s":float(end),"EventDuration_s":float(end-start),
             "EventSampleCount":int(len(ff)),"LocalBaselineStart_s":float(start-guard_s-baseline_s),"LocalBaselineEnd_s":float(start-guard_s),"BaselineSampleCount":int(len(bf)),"Status":"ok"}
        if len(ff)<min_samples or len(bf)<min_samples:
            row["Status"]="insufficient samples"; output.append(row); continue
        base=float(np.median(bf) if baseline_method=="median" else np.mean(bf)); baseline=linear_baseline(bt,bf); sigma=baseline.get("detrended_std_hz",np.nan)
        index=int(np.argmin(ff) if direction=="negative" else np.argmax(ff)); value=float(ff[index]); depth=(base-value) if direction=="negative" else (value-base)
        row |= {"EventMinimum_s":float(tt[index]),"LocalBaselineFrequency_Hz":base,"LocalBaselineRawStd_Hz":float(np.std(bf)),
                "LocalBaselineDetrendedStd_Hz":sigma,"EventMinimumFrequency_Hz":value,"DipDepth_Hz":depth,
                "LocalSNR":float(depth/sigma) if np.isfinite(sigma) and sigma>0 else np.nan,
                "BaselineMethod":baseline_method,"Status":"ok"}
        output.append(row)
    return output


def false_event_distribution(t, frequency, duration_s, baseline_s=0.4, guard_s=0.05,
                             stride_s=None, direction="negative", min_samples=2):
    stride_s = duration_s if stride_s is None else stride_s
    t, (f,), _ = _finite_time_series(t, frequency); rows=[]
    for start in np.arange(t[0]+baseline_s+guard_s, t[-1]-duration_s+1e-12, stride_s):
        bi0=np.searchsorted(t,start-guard_s-baseline_s,side="left"); bi1=np.searchsorted(t,start-guard_s,side="left")
        ei0=np.searchsorted(t,start,side="left"); ei1=np.searchsorted(t,start+duration_s,side="right")
        bt,bf=t[bi0:bi1],f[bi0:bi1]; et,ef=t[ei0:ei1],f[ei0:ei1]
        if len(bf)<min_samples or len(ef)<min_samples: continue
        base=float(np.median(bf)); baseline=linear_baseline(bt,bf); sigma=baseline.get("detrended_std_hz",np.nan)
        index=int(np.argmin(ef) if direction=="negative" else np.argmax(ef)); value=float(ef[index]); depth=base-value if direction=="negative" else value-base
        rows.append({"EventStart_s":float(start),"EventEnd_s":float(start+duration_s),"EventDuration_s":float(duration_s),"EventSampleCount":int(len(ef)),"BaselineSampleCount":int(len(bf)),"LocalBaselineFrequency_Hz":base,"LocalBaselineDetrendedStd_Hz":sigma,"EventMinimum_s":float(et[index]),"EventMinimumFrequency_Hz":value,"DipDepth_Hz":float(depth),"LocalSNR":float(depth/sigma) if np.isfinite(sigma) and sigma>0 else np.nan,"Status":"ok"})
    depths=np.asarray([r["DipDepth_Hz"] for r in rows],float)
    summary={"count":int(len(depths)),"mean_hz":np.nan,"median_hz":np.nan,"std_hz":np.nan,"max_hz":np.nan,"p95_hz":np.nan,"p99_hz":np.nan,"p999_hz":np.nan}
    if len(depths):
        summary.update({"mean_hz":float(np.mean(depths)),"median_hz":float(np.median(depths)),"std_hz":float(np.std(depths)),"max_hz":float(np.max(depths)),"p95_hz":float(np.percentile(depths,95)),"p99_hz":float(np.percentile(depths,99)),"p999_hz":float(np.percentile(depths,99.9))})
    return {"duration_s":duration_s,"stride_s":stride_s,"overlap":bool(stride_s<duration_s),"rows":rows,"summary":summary,"processing":"raw local-baseline minimum; no interpolation"}


def detect_particle_candidates(t, frequency, duration_s=0.15, baseline_s=0.4, guard_s=0.05,
                               threshold_hz=None, sigma_multiplier=5.0, stride_s=None,
                               threshold_table=None, amplitude=None, phase=None):
    """Compatibility wrapper around the waveform detector.

    Automatic detection deliberately requires a threshold table produced from
    an independent calibration blank.  ``threshold_hz`` remains available for
    an explicitly manual, single-duration threshold.
    """
    if threshold_table is None:
        if threshold_hz is None:
            return {
                "duration_s": float(duration_s), "baseline_s": float(baseline_s),
                "guard_s": float(guard_s), "stride_s": stride_s,
                "candidates": [], "all_candidates": [],
                "calibration_required": True,
                "processing": "waveform detection disabled: independent calibration threshold required",
            }
        threshold_table = [{
            "Duration_s": float(duration_s), "PeakThreshold_Hz": float(threshold_hz),
            "IntegralThreshold_Hz_s": 0.0, "TemplateThreshold_Hz": float(threshold_hz),
            "ThresholdMethod": "manual fixed threshold",
        }]
    result = detect_waveform_events(
        t, frequency, threshold_table, baseline_s=baseline_s, guard_s=guard_s,
        stride_fraction=(float(stride_s) / float(duration_s)) if stride_s else 0.25,
        amplitude=amplitude, phase=phase,
    )
    result["duration_s"] = float(duration_s)
    return result


def _window_waveform_metrics(t, f, start_s, duration_s, baseline_s, guard_s,
                             amplitude=None, phase=None, min_samples=4):
    """Return timestamp-aware dip metrics for one candidate window."""
    bi0 = np.searchsorted(t, start_s - guard_s - baseline_s, side="left")
    bi1 = np.searchsorted(t, start_s - guard_s, side="left")
    ei0 = np.searchsorted(t, start_s, side="left")
    ei1 = np.searchsorted(t, start_s + duration_s, side="right")
    bt, bf = t[bi0:bi1], f[bi0:bi1]
    et, ef = t[ei0:ei1], f[ei0:ei1]
    row = {
        "EventStart_s": float(start_s), "EventEnd_s": float(start_s + duration_s),
        "WindowDuration_s": float(duration_s), "EventSampleCount": int(len(ef)),
        "BaselineSampleCount": int(len(bf)), "Status": "ok",
    }
    if len(bt) < min_samples or len(et) < min_samples or np.ptp(bt) <= 0:
        row["Status"] = "insufficient samples"
        return row
    slope, intercept = np.polyfit(bt, bf, 1)
    predicted = slope * et + intercept
    dip = predicted - ef
    positive_dip = np.maximum(dip, 0.0)
    peak_index = int(np.argmax(dip))
    peak_depth = float(dip[peak_index])
    integral = float(np.trapezoid(positive_dip, et))

    # A half-sine is a simple symmetric transit template.  Its least-squares
    # amplitude keeps the score in Hz, while correlation reports shape match.
    u = np.clip((et - start_s) / duration_s, 0.0, 1.0)
    template = np.sin(np.pi * u)
    denom = float(np.dot(template, template))
    template_amp = float(np.dot(positive_dip, template) / denom) if denom > 0 else np.nan
    if np.std(positive_dip) > 0 and np.std(template) > 0:
        template_corr = float(np.corrcoef(positive_dip, template)[0, 1])
    else:
        template_corr = np.nan

    half = max(peak_depth * 0.5, 0.0)
    above = dip >= half
    left = right = peak_index
    while left > 0 and above[left - 1]:
        left -= 1
    while right + 1 < len(et) and above[right + 1]:
        right += 1
    actual_duration = float(et[right] - et[left]) if right > left else 0.0
    noise = linear_baseline(bt, bf).get("detrended_std_hz", np.nan)
    row |= {
        "PeakTime_s": float(et[peak_index]), "PeakDepth_Hz": peak_depth,
        "ActualTransitStart_s": float(et[left]), "ActualTransitEnd_s": float(et[right]),
        "ActualTransitTime_s": actual_duration, "IntegratedDip_Hz_s": integral,
        "TemplateAmplitude_Hz": template_amp, "TemplateCorrelation": template_corr,
        "LocalBaselineFrequency_Hz": float(np.median(bf)),
        "LocalBaselineDetrendedStd_Hz": float(noise),
        "PeakSNR": float(peak_depth / noise) if np.isfinite(noise) and noise > 0 else np.nan,
        "IntegralSNR": float(integral / (noise * np.sqrt(max(duration_s, 1e-12))))
        if np.isfinite(noise) and noise > 0 else np.nan,
    }
    for name, values, key in (
        ("Amplitude", amplitude, "AmplitudeChange_V"),
        ("Phase", phase, "PhaseChange_deg"),
    ):
        if values is None:
            row[key] = np.nan
            continue
        values = np.asarray(values, float)
        event_values, base_values = values[ei0:ei1], values[bi0:bi1]
        row[key] = (float(np.nanmedian(event_values) - np.nanmedian(base_values))
                    if len(event_values) and len(base_values) else np.nan)
    return row


def waveform_window_distribution(t, frequency, duration_s, baseline_s=0.4,
                                 guard_s=0.05, stride_s=None, amplitude=None,
                                 phase=None):
    """Measure peak, integral and template scores for fixed windows."""
    raw_t = np.asarray(t, float)
    raw_f = np.asarray(frequency, float)
    mask = np.isfinite(raw_t) & np.isfinite(raw_f)
    t, f = raw_t[mask], raw_f[mask]
    amplitude = np.asarray(amplitude, float)[mask] if amplitude is not None else None
    phase = np.asarray(phase, float)[mask] if phase is not None else None
    stride_s = float(duration_s if stride_s is None else stride_s)
    rows = []
    if len(t) >= 2:
        first = t[0] + baseline_s + guard_s
        last = t[-1] - duration_s
        for start in np.arange(first, last + 1e-12, max(stride_s, 1e-12)):
            row = _window_waveform_metrics(t, f, start, duration_s, baseline_s,
                                           guard_s, amplitude, phase)
            if row["Status"] == "ok":
                rows.append(row)
    return {
        "duration_s": float(duration_s), "stride_s": stride_s, "rows": rows,
        "processing": "local linear baseline; positive dip integral; half-sine template; actual timestamps; no resampling",
    }


def calibrate_waveform_thresholds(t, frequency, durations_s=DETECTION_DURATIONS_S,
                                  baseline_s=0.4, guard_s=0.05,
                                  familywise_false_alarm=0.01):
    """Freeze duration-specific thresholds from one calibration blank.

    Bonferroni correction controls the requested family-wise tail probability
    when several event durations are searched together.
    """
    durations = tuple(sorted(set(float(v) for v in durations_s if float(v) > 0)))
    if not durations:
        raise ValueError("at least one positive event duration is required")
    alpha = float(familywise_false_alarm)
    if not 0 < alpha < 1:
        raise ValueError("familywise_false_alarm must be between 0 and 1")
    per_duration_alpha = alpha / len(durations)
    percentile = 100.0 * (1.0 - per_duration_alpha)
    table = []
    distributions = {}
    for duration in durations:
        distribution = waveform_window_distribution(
            t, frequency, duration, baseline_s, guard_s, stride_s=duration)
        distributions[str(duration)] = distribution
        rows = distribution["rows"]
        if not rows:
            raise ValueError(f"calibration blank has no valid {duration:g} s windows")
        def q(key):
            values = np.asarray([row[key] for row in rows], float)
            values = values[np.isfinite(values)]
            return float(np.percentile(values, percentile)) if len(values) else np.nan
        table.append({
            "Duration_s": duration, "PeakThreshold_Hz": q("PeakDepth_Hz"),
            "IntegralThreshold_Hz_s": q("IntegratedDip_Hz_s"),
            "TemplateThreshold_Hz": q("TemplateAmplitude_Hz"),
            "CalibrationWindowCount": len(rows), "FamilywiseFalseAlarm": alpha,
            "PerDurationAlpha": per_duration_alpha, "Percentile": percentile,
            "ThresholdMethod": f"independent calibration blank; Bonferroni {len(durations)} durations",
        })
    return {
        "thresholds": table, "durations_s": list(durations),
        "familywise_false_alarm": alpha, "per_duration_alpha": per_duration_alpha,
        "distributions": distributions,
        "processing": "thresholds frozen from independent calibration blank",
    }


def _overlaps_flags(t, start, end, flags):
    if flags is None:
        return False
    left = np.searchsorted(t, start, side="left")
    right = np.searchsorted(t, end, side="right")
    return bool(np.any(np.asarray(flags, bool)[left:right]))


def detect_waveform_events(t, frequency, threshold_table, baseline_s=0.4,
                           guard_s=0.05, method="integral", stride_fraction=0.25,
                           amplitude=None, phase=None, transient=None,
                           exclude_transients=True, distance_mm=49.0):
    """Detect and de-duplicate calibrated events across several durations."""
    t = np.asarray(t, float)
    f = np.asarray(frequency, float)
    table = [dict(row) for row in threshold_table]
    if method not in ("integral", "template"):
        raise ValueError("method must be 'integral' or 'template'")
    hits = []
    for threshold in table:
        duration = float(threshold["Duration_s"])
        distribution = waveform_window_distribution(
            t, f, duration, baseline_s, guard_s,
            stride_s=max(duration * float(stride_fraction), 1e-12),
            amplitude=amplitude, phase=phase,
        )
        score_key = "IntegratedDip_Hz_s" if method == "integral" else "TemplateAmplitude_Hz"
        limit_key = "IntegralThreshold_Hz_s" if method == "integral" else "TemplateThreshold_Hz"
        score_limit = float(threshold[limit_key])
        peak_limit = float(threshold.get("PeakThreshold_Hz", 0.0))
        for row in distribution["rows"]:
            score = float(row.get(score_key, np.nan))
            peak = float(row.get("PeakDepth_Hz", np.nan))
            if not np.isfinite(score) or score <= score_limit or peak <= peak_limit:
                continue
            item = dict(row)
            item |= {
                "DetectionMethod": method, "DetectionScore": score,
                "ScoreThreshold": score_limit, "PeakThreshold_Hz": peak_limit,
                "ThresholdMethod": threshold.get("ThresholdMethod", "frozen calibration"),
                "NormalizedScore": score / score_limit if score_limit > 0 else np.inf,
                "Detected": True,
            }
            item["EventTransientOverlap"] = _overlaps_flags(
                t, item["EventStart_s"], item["EventEnd_s"], transient)
            item["BaselineTransientOverlap"] = _overlaps_flags(
                t, item["EventStart_s"] - baseline_s - guard_s,
                item["EventStart_s"], transient)
            item["TransientOverlap"] = bool(
                item["EventTransientOverlap"] or item["BaselineTransientOverlap"])
            transit = item.get("ActualTransitTime_s", np.nan)
            item["Velocity_mm_s"] = (float(distance_mm) / transit
                                      if np.isfinite(transit) and transit > 0 else np.nan)
            item["Decision"] = ("excluded_transient" if exclude_transients and item["TransientOverlap"]
                                else "accepted")
            hits.append(item)

    # Search windows overlap heavily and durations describe the same physical
    # event.  Keep the strongest normalized score within each overlapping group.
    hits.sort(key=lambda row: (row["PeakTime_s"], -row["NormalizedScore"]))
    groups = []
    for hit in hits:
        overlapping = [g for g in groups if hit["EventStart_s"] <= g["end"] and hit["EventEnd_s"] >= g["start"]]
        if not overlapping:
            groups.append({"start": hit["EventStart_s"], "end": hit["EventEnd_s"], "rows": [hit]})
        else:
            group = overlapping[0]
            group["start"] = min(group["start"], hit["EventStart_s"])
            group["end"] = max(group["end"], hit["EventEnd_s"])
            group["rows"].append(hit)
    all_candidates = []
    for event_id, group in enumerate(groups, 1):
        best = max(group["rows"], key=lambda row: row["NormalizedScore"])
        best = dict(best)
        best["EventID"] = event_id
        best["MatchedDurationCount"] = len({row["WindowDuration_s"] for row in group["rows"]})
        all_candidates.append(best)
    accepted = [row for row in all_candidates if row["Decision"] == "accepted"]
    excluded = [row for row in all_candidates if row["Decision"] != "accepted"]
    return {
        "candidates": accepted, "all_candidates": all_candidates,
        "excluded_transient": excluded, "thresholds": table,
        "baseline_s": float(baseline_s), "guard_s": float(guard_s),
        "distance_mm": float(distance_mm),
        "processing": f"calibrated {method} waveform detector; multi-duration de-duplication; no resampling",
    }


def poisson_rate_summary(event_count, observation_s, confidence=0.95):
    """Observed false-event rate and exact one-sided Poisson upper limit."""
    count = int(event_count)
    hours = float(observation_s) / 3600.0
    rate = count / hours if hours > 0 else np.nan
    upper_count = 0.5 * chi2.ppf(float(confidence), 2 * (count + 1))
    return {
        "event_count": count, "observation_s": float(observation_s),
        "observation_h": hours, "false_events_per_h": float(rate),
        "poisson_confidence": float(confidence),
        "poisson_upper_events_per_h": float(upper_count / hours) if hours > 0 else np.nan,
    }


def evaluate_blank(t, frequency, calibration, baseline_s=0.4, guard_s=0.05,
                   method="integral", amplitude=None, phase=None,
                   transient=None, distance_mm=49.0):
    """Evaluate a blank without changing any calibration threshold."""
    detection = detect_waveform_events(
        t, frequency, calibration["thresholds"], baseline_s, guard_s, method,
        amplitude=amplitude, phase=phase, transient=transient,
        distance_mm=distance_mm,
    )
    finite_t = np.asarray(t, float)
    finite_t = finite_t[np.isfinite(finite_t)]
    duration = float(finite_t[-1] - finite_t[0]) if len(finite_t) >= 2 else 0.0
    rate = poisson_rate_summary(len(detection["candidates"]), duration)
    return {
        "thresholds_frozen": True, "detection": detection, "false_event_rate": rate,
        "processing": "evaluation blank scored with unchanged independent-calibration thresholds",
    }


def load_plotter_csv(path):
    """Load legacy/current Plotter CSV with optional # Metadata preamble."""
    metadata={}; rows=[]; header=None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row: continue
            if row[0]=="# Metadata" and len(row)>=3: metadata[row[1]]=row[2]; continue
            if header is None: header=row; continue
            if len(row)<len(header): row += [""]*(len(header)-len(row))
            rows.append(dict(zip(header,row)))
    if not header: raise ValueError("CSV header를 찾지 못했습니다.")
    required=("Time_s","Frequency_Hz","Amplitude_R_V","Phase_deg")
    missing=[column for column in required if column not in header]
    if missing: raise ValueError("필수 column 누락: "+", ".join(missing))
    def numeric(column):
        return np.asarray([float(r.get(column,"nan")) if r.get(column,"") not in ("",None) else np.nan for r in rows],float)
    result = {"metadata":metadata,"header":header,"rows":rows,"time_s":numeric("Time_s"),"frequency_hz":numeric("Frequency_Hz"),"amplitude_v":numeric("Amplitude_R_V"),"phase_deg":numeric("Phase_deg")}
    # Legacy exports do not have this field; retain Phase_deg as a display-only
    # fallback in that case rather than inventing a phase-error signal.
    if "PLL_PhaseError_deg" in header:
        result["phase_error_deg"] = numeric("PLL_PhaseError_deg")
    return result


def transient_flags(t, frequency, amplitude=None, phase=None, window_s=10.0, multiplier=5.0,
                    include_frequency=True):
    """Derivative/step flag using trailing MAD scale; raw samples are retained."""
    t = np.asarray(t, float); channels = {}
    if include_frequency: channels["Frequency"] = np.asarray(frequency, float)
    if amplitude is not None: channels["Amplitude"] = np.asarray(amplitude, float)
    if phase is not None: channels["Phase"] = np.asarray(phase, float)
    flags = np.zeros(len(t), dtype=bool); types = np.full(len(t), "", dtype=object); magnitude = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        hits = []
        for name, values in channels.items():
            left = np.searchsorted(t, t[i]-window_s, side="left")
            diff = np.diff(values[left:i+1]); diff = diff[np.isfinite(diff)]
            if len(diff) < 3 or not np.isfinite(values[i]-values[i-1]): continue
            mad = np.median(np.abs(diff - np.median(diff))) * 1.4826
            if mad > 0 and abs(values[i]-values[i-1]) > multiplier*mad: hits.append((name, abs(values[i]-values[i-1])))
        if hits:
            flags[i] = True; types[i] = "multiple" if len(hits)>1 else hits[0][0]
            magnitude[i] = max(v for _, v in hits)
    return {"flag": flags, "type": types, "magnitude": magnitude, "definition": f"step > {multiplier} × trailing MAD scale, window={window_s}s; channels={','.join(channels)}"}


def correlations(frequency, amplitude, phase, detrended_frequency=None):
    def corr(a, b):
        a, b = np.asarray(a,float), np.asarray(b,float); mask=np.isfinite(a)&np.isfinite(b)
        if np.count_nonzero(mask)<3 or np.std(a[mask])==0 or np.std(b[mask])==0: return {"r":np.nan,"n":int(np.count_nonzero(mask)),"reason":"insufficient or constant"}
        return {"r":float(np.corrcoef(a[mask],b[mask])[0,1]),"n":int(np.count_nonzero(mask)),"reason":""}
    result={"frequency_amplitude":corr(frequency,amplitude),"frequency_phase":corr(frequency,phase),"amplitude_phase":corr(amplitude,phase)}
    if detrended_frequency is not None: result["detrended_frequency_amplitude"] = corr(detrended_frequency,amplitude)
    return result


def allan_deviation(t, frequency, taus=EVENT_WINDOWS_S + (2.,5.,10.,30.,60.,300.)):
    """Non-overlapping, timestamp-binned Allan deviation; no resampling."""
    t, (f,), _ = _finite_time_series(t, frequency); results=[]
    for tau in taus:
        if tau <= 0 or len(t)<4: continue
        bins=np.floor((t-t[0])/tau).astype(int); means=[]
        for b in range(int(bins.min()), int(bins.max())+1):
            v=f[bins==b]
            if len(v)>=2: means.append(float(np.mean(v)))
        if len(means)<2: continue
        means=np.asarray(means); adev=np.sqrt(0.5*np.mean(np.diff(means)**2))
        results.append({"tau_s":float(tau),"allan_deviation_hz":float(adev),"cluster_count":int(len(means)),"method":"non-overlapping timestamp bins; no resampling"})
    return results
