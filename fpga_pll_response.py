"""Closed-loop step-response analysis for the SMR FPGA PLL."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class PLLStepResponse:
    baseline_hz: float
    final_hz: float
    step_hz: float
    delay_step_turns: float
    dc_gain_hz_per_turn: float
    tau_s: float
    bandwidth_hz: float
    dead_time_s: float
    overshoot_percent: float
    settling_2pct_s: float
    noise_sigma_hz: float
    r_squared: float
    accepted: bool
    reasons: tuple[str, ...]


def _mad_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def analyze_step_response(time_s, frequency_hz, *, step_time_s: float,
                          restore_time_s: float, delay_step_turns: float) -> PLLStepResponse:
    """Fit a delayed first-order response to a bounded PLL delay step."""
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(frequency_hz, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 30 or not np.all(np.diff(t) > 0):
        raise ValueError("step 응답에는 증가하는 시간축과 유효 샘플이 최소 30개 필요합니다.")
    if not t[0] < step_time_s < restore_time_s < t[-1]:
        raise ValueError("step 및 복귀 시간이 기록 구간 안에 있어야 합니다.")
    if not np.isfinite(delay_step_turns) or abs(delay_step_turns) < 1e-12:
        raise ValueError("delay step은 0이 아니어야 합니다.")

    pre = y[t < step_time_s]
    step_mask = (t >= step_time_s) & (t < restore_time_s)
    ts = t[step_mask] - step_time_s
    ys = y[step_mask]
    if len(pre) < 10 or len(ys) < 20:
        raise ValueError("step 전후 측정 샘플이 부족합니다.")
    pre_tail = pre[max(0, len(pre) // 2):]
    steady = ys[max(0, int(0.75 * len(ys))):]
    baseline = float(np.median(pre_tail))
    final = float(np.median(steady))
    initial_step = final - baseline
    noise_sigma = _mad_sigma(pre_tail)
    dt = float(np.median(np.diff(t)))
    duration = float(ts[-1] - ts[0])
    tau0 = max(5.0 * dt, min(duration / 5.0, 0.1))

    def model(p):
        base, amplitude, log_tau, dead = p
        tau = np.exp(log_tau)
        elapsed = np.maximum(0.0, ts - dead)
        return base + amplitude * (1.0 - np.exp(-elapsed / tau))

    scale = max(noise_sigma, abs(initial_step) * 0.02, np.finfo(float).eps)

    def residuals(p):
        return (model(p) - ys) / scale

    lower_tau = max(dt / 4.0, 1e-6)
    upper_tau = max(duration * 10.0, lower_tau * 10.0)
    result = least_squares(
        residuals,
        np.array([baseline, initial_step, np.log(tau0), 0.0]),
        bounds=(
            [-np.inf, -np.inf, np.log(lower_tau), 0.0],
            [np.inf, np.inf, np.log(upper_tau), max(dt, 0.25 * duration)],
        ),
        loss="soft_l1", max_nfev=5000,
    )
    fitted = model(result.x)
    base_fit, amplitude, log_tau, dead = map(float, result.x)
    tau = float(np.exp(log_tau))
    residual_power = float(np.sum((ys - fitted) ** 2))
    signal_power = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = float(1.0 - residual_power / max(signal_power, np.finfo(float).tiny))
    direction = 1.0 if amplitude >= 0 else -1.0
    normalized = direction * (ys - base_fit) / max(abs(amplitude), np.finfo(float).tiny)
    overshoot = max(0.0, float((np.max(normalized) - 1.0) * 100.0))
    bandwidth = float(1.0 / (2.0 * np.pi * tau))
    reasons = []
    steady_uncertainty = noise_sigma / np.sqrt(max(1, len(steady)))
    if abs(amplitude) <= max(0.01, 6.0 * steady_uncertainty):
        reasons.append("delay step에 대한 주파수 변화가 baseline noise와 구별되지 않습니다.")
    if not np.isfinite(r_squared) or r_squared < 0.80:
        reasons.append(f"1차 응답 적합도 R²={r_squared:.3f}가 0.80보다 낮습니다.")
    if tau >= 0.95 * upper_tau or tau <= 1.05 * lower_tau:
        reasons.append("응답 시상수가 fitting 경계에 있어 측정 구간 또는 샘플 속도가 부족합니다.")
    if overshoot > 20.0:
        reasons.append(f"overshoot {overshoot:.1f}%가 커서 1차 저역통과 응답으로 보기 어렵습니다.")

    return PLLStepResponse(
        baseline_hz=base_fit, final_hz=base_fit + amplitude,
        step_hz=amplitude, delay_step_turns=float(delay_step_turns),
        dc_gain_hz_per_turn=float(amplitude / delay_step_turns),
        tau_s=tau, bandwidth_hz=bandwidth, dead_time_s=dead,
        overshoot_percent=overshoot, settling_2pct_s=float(dead + 3.912 * tau),
        noise_sigma_hz=noise_sigma, r_squared=r_squared,
        accepted=not reasons, reasons=tuple(reasons),
    )


def save_step_response(directory, time_s, frequency_hz, segment, result,
                       metadata=None) -> tuple[Path, Path, Path]:
    """Save raw response, fitted summary, and a review plot."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"fpga_pll_step_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = directory / f"{stem}.csv"
    json_path = directory / f"{stem}.json"
    png_path = directory / f"{stem}.png"
    payload = {**(metadata or {}), **asdict(result)}
    payload["reasons"] = list(result.reasons)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        for key, value in payload.items():
            writer.writerow(["# Metadata", key, value])
        writer.writerow(["Time_s", "Frequency_Hz", "Segment"])
        writer.writerows(zip(time_s, frequency_hz, segment))
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(time_s, float)
    y = np.asarray(frequency_hz, float)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    stride = max(1, len(t) // 6000)
    ax.plot(t[::stride], y[::stride], color="#1769d2", lw=1.0, label="Measured frequency")
    step_time = float((metadata or {}).get("step_time_s", 0.0))
    restore_time = float((metadata or {}).get("restore_time_s", t[-1]))
    fit_mask = (t >= step_time) & (t < restore_time)
    tf = t[fit_mask]
    if len(tf):
        elapsed = np.maximum(0.0, tf - step_time - result.dead_time_s)
        fit_y = result.baseline_hz + result.step_hz * (1.0 - np.exp(-elapsed / result.tau_s))
        ax.plot(tf, fit_y, color="#d13b73", lw=2.0, label="First-order fit")
    ax.axvline(step_time, color="#d47b00", ls="--", lw=1, label="Delay step")
    ax.axvline(restore_time, color="#555", ls="--", lw=1, label="Restore")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PLL frequency (Hz)")
    ax.set_title(
        f"FPGA PLL step response: BW={result.bandwidth_hz:.3g} Hz, "
        f"tau={result.tau_s:.4g} s, R²={result.r_squared:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, json_path, png_path
