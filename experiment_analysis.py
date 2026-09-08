"""Cross-session comparison and acquisition integration diagnostics."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from baseline_analysis import allan_deviation, linear_baseline, load_plotter_csv, timing_quality


CONDITION_KEYS = (
    "Mounting_condition", "Clamp_torque_contact", "Fluid", "Triton_concentration",
    "Triton_basis", "Particle_material", "Particle_size_min_um",
    "Particle_size_max_um", "Effective_travel_distance_mm",
)


def _numeric_metadata(metadata, key):
    try:
        return float(metadata.get(key, "nan"))
    except (TypeError, ValueError):
        return np.nan


def _condition_signature(metadata):
    return {key: str(metadata.get(key, "")).strip() for key in CONDITION_KEYS}


def summarize_baseline_file(path):
    loaded = load_plotter_csv(path)
    t, f = loaded["time_s"], loaded["frequency_hz"]
    amplitude = loaded["amplitude_v"]
    baseline = linear_baseline(t, f)
    timing = timing_quality(t)
    allan = allan_deviation(t, f)
    finite_adev = [row for row in allan if np.isfinite(row["allan_deviation_hz"])]
    closest = min(finite_adev, key=lambda row: abs(row["tau_s"] - 0.5)) if finite_adev else {}
    minimum = min(finite_adev, key=lambda row: row["allan_deviation_hz"]) if finite_adev else {}
    metadata = loaded["metadata"]
    summary = {
        "Source": os.path.abspath(path),
        "SessionID": metadata.get("SessionID", ""),
        "MeanFrequency_Hz": baseline.get("mean_hz", np.nan),
        "DetrendedSigma_Hz": baseline.get("detrended_std_hz", np.nan),
        "AllanTauClosestTo0p5_s": closest.get("tau_s", np.nan),
        "AllanDeviationClosestTo0p5_Hz": closest.get("allan_deviation_hz", np.nan),
        "AllanMinimumTau_s": minimum.get("tau_s", np.nan),
        "AllanMinimum_Hz": minimum.get("allan_deviation_hz", np.nan),
        "MeanAmplitude_V": float(np.nanmean(amplitude)) if np.any(np.isfinite(amplitude)) else np.nan,
        "SamplingRate_Hz": timing.get("effective_rate_hz", np.nan),
        "SampleCount": timing.get("sample_count", 0),
        "Q": np.nan,
        "Conditions": _condition_signature(metadata),
    }
    summary.update(summary["Conditions"])
    return summary


def summarize_sweep_file(path):
    metadata = {}
    rows = []
    header = None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for values in csv.reader(handle):
            if not values:
                continue
            if values[0] == "# Metadata" and len(values) >= 3:
                metadata[values[1]] = values[2]
                continue
            if header is None:
                header = values
                continue
            rows.append(dict(zip(header, values)))
    q_values = []
    for row in rows:
        for key in ("Q", "Q_factor", "QualityFactor"):
            try:
                q = float(row.get(key, "nan"))
            except (TypeError, ValueError):
                q = np.nan
            if np.isfinite(q):
                q_values.append(q)
                break
    summary = {
        "Source": os.path.abspath(path), "SessionID": metadata.get("SessionID", ""),
        "Q": float(np.mean(q_values)) if q_values else np.nan,
        "Conditions": _condition_signature(metadata),
    }
    summary.update(summary["Conditions"])
    return summary


def compare_baselines(baseline_paths, sweep_paths=()):
    """Summarize baseline files and associate sweep Q by SessionID when possible."""
    baseline_rows = [summarize_baseline_file(path) for path in baseline_paths]
    sweeps = [summarize_sweep_file(path) for path in sweep_paths]
    sweep_by_session = {}
    for row in sweeps:
        session_id = row["SessionID"]
        if not session_id:
            continue
        current = sweep_by_session.get(session_id)
        if current is None or (not np.isfinite(current["Q"]) and np.isfinite(row["Q"])):
            sweep_by_session[session_id] = row
    for row in baseline_rows:
        sweep = sweep_by_session.get(row["SessionID"])
        if sweep:
            row["Q"] = sweep["Q"]
            row["SweepSource"] = sweep["Source"]
        else:
            row["SweepSource"] = ""
    signatures = [json.dumps(row["Conditions"], ensure_ascii=False, sort_keys=True)
                  for row in baseline_rows]
    warnings = []
    if len(set(signatures)) > 1:
        differing = [key for key in CONDITION_KEYS
                     if len({row["Conditions"].get(key, "") for row in baseline_rows}) > 1]
        warnings.append("실험조건 불일치: " + ", ".join(differing))
    missing = [os.path.basename(row["Source"]) for row in baseline_rows
               if any(not row["Conditions"].get(key) for key in CONDITION_KEYS)]
    if missing:
        warnings.append("조건 metadata 누락 파일: " + ", ".join(missing))
    return {"rows": baseline_rows, "sweeps": sweeps, "warnings": warnings,
            "condition_keys": list(CONDITION_KEYS)}


def write_comparison(result, csv_path):
    csv_path = str(csv_path)
    json_path = str(Path(csv_path).with_suffix(".json"))
    fields = [
        "Source", "SessionID", "MeanFrequency_Hz", "DetrendedSigma_Hz",
        "AllanTauClosestTo0p5_s", "AllanDeviationClosestTo0p5_Hz",
        "AllanMinimumTau_s", "AllanMinimum_Hz", "MeanAmplitude_V", "Q",
        "SamplingRate_Hz", "SampleCount", "SweepSource", *CONDITION_KEYS, "ConditionMatch",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        condition_match = not result["warnings"]
        for source in result["rows"]:
            row = {key: source.get(key, "") for key in fields}
            row["ConditionMatch"] = condition_match
            writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return csv_path, json_path


def integration_diagnostics(time_s, received_sample_count, pll_status_age_s=None,
                            demod_latency_s=None, pressure_command_a=None,
                            pressure_command_b=None, measured_pressure_a=None,
                            measured_pressure_b=None, expected_pll_update_s=0.25,
                            ui_refresh_latency_s=None, ui_refresh_period_s=0.05):
    """Audit a recorded acquisition without requiring live hardware access."""
    t = np.asarray(time_s, float)
    timing = timing_quality(t)
    logged = int(len(t))
    received = int(received_sample_count)
    pll_age = np.asarray(pll_status_age_s if pll_status_age_s is not None else [], float)
    latency = np.asarray(demod_latency_s if demod_latency_s is not None else [], float)
    finite_age = pll_age[np.isfinite(pll_age)]
    finite_latency = latency[np.isfinite(latency)]
    ui_latency = np.asarray(ui_refresh_latency_s if ui_refresh_latency_s is not None else [], float)
    finite_ui = ui_latency[np.isfinite(ui_latency)]
    missing = max(received - logged, 0)
    checks = {
        "received_vs_logged": {
            "received": received, "logged": logged, "missing": missing,
            "pass": received == logged,
        },
        "timestamps": {
            "duplicates": timing.get("duplicate_timestamps", 0),
            "non_monotonic": timing.get("non_monotonic_timestamps", 0),
            "long_gaps": timing.get("long_gap_count", 0),
            "pass": (timing.get("duplicate_timestamps", 0) == 0 and
                     timing.get("non_monotonic_timestamps", 0) == 0 and
                     timing.get("long_gap_count", 0) == 0),
        },
        "pll_status_update": {
            "max_age_s": float(np.max(finite_age)) if len(finite_age) else np.nan,
            "expected_max_s": float(expected_pll_update_s * 2.0),
            "pass": bool(len(finite_age) and np.max(finite_age) <= expected_pll_update_s * 2.0),
        },
        "demod_read_latency": {
            "p99_s": float(np.percentile(finite_latency, 99)) if len(finite_latency) else np.nan,
            "max_s": float(np.max(finite_latency)) if len(finite_latency) else np.nan,
        },
        "memory_estimate": {
            "logged_rows": logged,
            "estimated_bytes": int(logged * 24 * 8),
        },
        "ui_refresh_latency": {
            "count": int(len(finite_ui)),
            "p99_s": float(np.percentile(finite_ui, 99)) if len(finite_ui) else np.nan,
            "max_s": float(np.max(finite_ui)) if len(finite_ui) else np.nan,
            "target_period_s": float(ui_refresh_period_s),
            "pass": bool(len(finite_ui) and np.percentile(finite_ui, 99) <= ui_refresh_period_s),
        },
    }
    command_a = np.asarray(pressure_command_a if pressure_command_a is not None else [], float)
    command_b = np.asarray(pressure_command_b if pressure_command_b is not None else [], float)
    measured_a = np.asarray(measured_pressure_a if measured_pressure_a is not None else [], float)
    measured_b = np.asarray(measured_pressure_b if measured_pressure_b is not None else [], float)
    finite_measured = np.isfinite(measured_a) & np.isfinite(measured_b)
    checks["pressure_recording"] = {
        "command_rows": int(min(len(command_a), len(command_b))),
        "measured_rows": int(np.count_nonzero(finite_measured)),
        "pass": bool(len(command_a) == logged and len(command_b) == logged and
                     np.count_nonzero(finite_measured) == logged),
        "note": "Measured values may be manually entered; inspect Pressure_source metadata.",
    }
    checks["overall_pass"] = bool(all(
        item.get("pass", True) for item in checks.values() if isinstance(item, dict)
    ))
    return {"timing": timing, "checks": checks}
