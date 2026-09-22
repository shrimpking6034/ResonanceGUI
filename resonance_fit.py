"""Complex second-order resonator fitting for HF2 demodulator data."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclass(frozen=True)
class ResonanceFit:
    f0_hz: float
    q: float
    linewidth_hz: float
    phase_deg: float
    phase_slope_deg_per_hz: float
    f0_std_hz: float
    q_std: float
    reduced_chi2: float
    success: bool
    message: str
    parameters: np.ndarray
    resonance_sign: int = 1


@dataclass(frozen=True)
class ResonanceFitQuality:
    accepted: bool
    linewidth_bins: float
    normalized_rmse: float
    complex_r_squared: float
    reasons: tuple


def detect_resonance_peaks(frequency_hz, amplitude):
    """Detect resonance envelopes without treating point noise as resonances.

    Peak finding on the raw trace is unstable when a sweep contains only one
    sample per frequency.  A short Savitzky-Golay trend is used only for peak
    detection and the raw complex samples remain untouched for fitting.
    """
    f = np.asarray(frequency_hz, dtype=float)
    r = np.asarray(amplitude, dtype=float)
    if len(r) < 5 or len(f) != len(r):
        return np.array([], dtype=int), np.array([], dtype=float), r.copy(), np.nan

    # Roughly 2 % of the trace suppresses isolated samples while preserving
    # the much wider mechanical resonance envelope.  Keep the window odd.
    window = max(5, int(round(len(r) * 0.02)))
    if window % 2 == 0:
        window += 1
    window = min(window, len(r) if len(r) % 2 else len(r) - 1)
    smooth = savgol_filter(r, window_length=window, polyorder=min(2, window - 1), mode="interp")
    residual = r - smooth
    median = float(np.median(residual))
    noise_sigma = float(1.4826 * np.median(np.abs(residual - median)))
    smooth_range = float(np.ptp(smooth))
    prominence = max(0.03 * smooth_range, 5.0 * noise_sigma, np.finfo(float).eps)
    indices, _ = find_peaks(smooth, prominence=prominence, distance=max(2, window // 2))
    if len(indices) == 0:
        return indices, np.array([], dtype=float), smooth, noise_sigma

    widths, _, _, _ = peak_widths(smooth, indices, rel_height=0.5)
    spacing = float(np.median(np.diff(np.sort(f)))) if len(f) > 1 else 1.0
    return indices, widths * abs(spacing), smooth, noise_sigma


def three_db_bounds(frequency_hz, smoothed_amplitude, peak_index):
    """Interpolate absolute -3 dB crossings; return None if either is missing.

    A prominence half-width differs from -3 dB when a background is present.
    """
    f = np.asarray(frequency_hz, dtype=float)
    r = np.asarray(smoothed_amplitude, dtype=float)
    p = int(peak_index)
    if (len(f) != len(r) or p <= 0 or p >= len(f) - 1 or
            not np.all(np.diff(f) > 0) or not np.all(np.isfinite(r)) or r[p] <= 0):
        return None
    level = r[p] / np.sqrt(2.0)
    left = np.flatnonzero(r[:p] <= level)
    right = np.flatnonzero(r[p + 1:] <= level)
    if not len(left) or not len(right):
        return None
    li = int(left[-1])
    ri = int(p + 1 + right[0])
    left_hz = float(np.interp(level, r[li:li + 2], f[li:li + 2]))
    right_hz = float(np.interp(level, r[ri - 1:ri][::-1], f[ri - 1:ri][::-1]))
    return left_hz, right_hz


def assess_fit_quality(frequency_hz, x, y, fit, min_linewidth_bins=3.0,
                       min_complex_r_squared=0.80):
    """Return conservative diagnostics for using a fit in PLL design."""
    f = np.asarray(frequency_hz, dtype=float)
    z = np.asarray(x, dtype=float) + 1j * np.asarray(y, dtype=float)
    valid = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag)
    f, z = f[valid], z[valid]
    if len(f) < 2:
        return ResonanceFitQuality(False, 0.0, np.inf, -np.inf,
                                   ("유효 측정점이 부족합니다.",))

    spacing = float(np.median(np.diff(np.unique(np.sort(f)))))
    linewidth_bins = float(fit.linewidth_hz / spacing) if spacing > 0 else 0.0
    model = model_response(f, fit.parameters, float(np.mean(f)), fit.resonance_sign)
    residual_power = float(np.sum(np.abs(model - z) ** 2))
    signal_power = float(np.sum(np.abs(z - np.mean(z)) ** 2))
    normalized_rmse = float(np.sqrt(residual_power / max(signal_power, np.finfo(float).tiny)))
    complex_r_squared = float(1.0 - residual_power / max(signal_power, np.finfo(float).tiny))

    reasons = []
    observed_span = float(np.ptp(f))
    if not np.isfinite(fit.linewidth_hz) or fit.linewidth_hz <= 0 or fit.linewidth_hz > observed_span:
        reasons.append(
            f"피팅 선폭 {fit.linewidth_hz:.3g} Hz가 측정 구간 {observed_span:.3g} Hz보다 넓어 "
            "공진 곡률을 확인할 수 없습니다. 공진 주변을 더 넓게 스윕하세요."
        )
    if linewidth_bins < min_linewidth_bins:
        target_step = fit.linewidth_hz / 5.0
        reasons.append(
            f"선폭 {fit.linewidth_hz:.3g} Hz가 스윕 간격 {spacing:.3g} Hz로 해상되지 않습니다 "
            f"({linewidth_bins:.2f} points/FWHM; 권장 간격 ≤{target_step:.3g} Hz)."
        )
    if not np.isfinite(complex_r_squared) or complex_r_squared < min_complex_r_squared:
        reasons.append(
            f"복소 적합도 R²={complex_r_squared:.3f}가 기준 {min_complex_r_squared:.2f}보다 낮습니다."
        )
    if not np.isfinite(fit.q_std) or fit.q_std > 0.5 * fit.q:
        reasons.append("Q 불확도가 50%보다 크거나 계산되지 않았습니다.")
    if not np.isfinite(fit.f0_std_hz) or fit.f0_std_hz > 0.5 * fit.linewidth_hz:
        reasons.append("중심주파수 불확도가 선폭에 비해 너무 큽니다.")

    return ResonanceFitQuality(not reasons, linewidth_bins, normalized_rmse,
                               complex_r_squared, tuple(reasons))


def model_response(frequency_hz, parameters, center_hz=None, resonance_sign=1):
    """Complex background plus a resonance in either I/Q convention.

    The denominator sign is +1 for decreasing resonance phase and -1 for
    increasing resonance phase. The complex background can alter the total
    response's local phase slope. The default preserves the legacy convention.
    """
    if resonance_sign not in (-1, 1):
        raise ValueError("resonance_sign must be +1 or -1")
    f = np.asarray(frequency_hz, dtype=float)
    f0, log_q, ar, ai, b0r, b0i, b1r, b1i = parameters
    q = np.exp(log_q)
    center_hz = f0 if center_hz is None else center_hz
    resonance = (ar + 1j * ai) / (1.0 - (f / f0) ** 2 + resonance_sign * 1j * f / (q * f0))
    background = (b0r + 1j * b0i) + (b1r + 1j * b1i) * (f - center_hz)
    return resonance + background


def fit_complex_resonance(frequency_hz, x, y, x_sem=None, y_sem=None,
                          f0_guess=None, f0_bounds=None):
    """Fit one complex resonance, automatically selecting its I/Q convention.

    Both denominator signs are compared using weighted least-squares cost
    before robust refinement. Data, fitted phase, and phase slope remain in
    the original measurement coordinates.

    ``f0_guess`` and ``f0_bounds`` make a user-selected peak an explicit
    fitting constraint.  Without them a broad multi-mode sweep naturally
    converges to its largest feature, which is not necessarily the selected
    resonance.
    """
    f = np.asarray(frequency_hz, dtype=float)
    z = np.asarray(x, dtype=float) + 1j * np.asarray(y, dtype=float)
    valid = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag)
    f, z = f[valid], z[valid]
    if len(f) < 9 or np.ptp(f) <= 0:
        raise ValueError("복소 공진 fitting에는 서로 다른 유효 포인트가 최소 9개 필요합니다.")
    order = np.argsort(f)
    f, z = f[order], z[order]
    center = float(np.mean(f))
    peak = int(np.argmax(np.abs(z)))
    if f0_guess is None:
        f0_guess = float(f[peak])
    else:
        f0_guess = float(f0_guess)
        if not f[0] <= f0_guess <= f[-1]:
            raise ValueError("f0 초기값이 선택된 fitting 구간 밖에 있습니다.")
        peak = int(np.argmin(np.abs(f - f0_guess)))
    span = float(np.ptp(f))
    q_guess = max(5.0, min(1e7, f0_guess / max(span / 8.0, np.finfo(float).eps)))
    edge_background = 0.5 * (z[0] + z[-1])
    amplitude = z[peak] - edge_background
    p0 = np.array([f0_guess, np.log(q_guess), amplitude.real, amplitude.imag,
                   edge_background.real, edge_background.imag, 0.0, 0.0])
    amplitude_scale = max(np.ptp(z.real), np.ptp(z.imag), abs(amplitude), 1e-9)
    if f0_bounds is None:
        f0_lower, f0_upper = float(f[0]), float(f[-1])
    else:
        f0_lower, f0_upper = map(float, f0_bounds)
        f0_lower = max(float(f[0]), f0_lower)
        f0_upper = min(float(f[-1]), f0_upper)
        if not f0_lower < f0_upper or not f0_lower <= f0_guess <= f0_upper:
            raise ValueError("유효하지 않은 선택 피크 주파수 제한입니다.")
    lower = [f0_lower, np.log(0.5), -np.inf, -np.inf, -np.inf, -np.inf,
             -100.0 * amplitude_scale / span, -100.0 * amplitude_scale / span]
    upper = [f0_upper, np.log(1e9), np.inf, np.inf, np.inf, np.inf,
             100.0 * amplitude_scale / span, 100.0 * amplitude_scale / span]

    sx = np.ones(len(f))
    sy = np.ones(len(f))
    if x_sem is not None:
        sx_raw = np.asarray(x_sem, dtype=float)[valid][order]
        sx = np.where(np.isfinite(sx_raw) & (sx_raw > 0), sx_raw, 1.0)
    if y_sem is not None:
        sy_raw = np.asarray(y_sem, dtype=float)[valid][order]
        sy = np.where(np.isfinite(sy_raw) & (sy_raw > 0), sy_raw, 1.0)
    # Preserve relative uncertainty but avoid unrealistically tiny weights.
    sx = np.maximum(sx, amplitude_scale * 1e-9)
    sy = np.maximum(sy, amplitude_scale * 1e-9)

    def residuals(p, resonance_sign):
        model = model_response(f, p, center, resonance_sign)
        return np.concatenate(((model.real - z.real) / sx, (model.imag - z.imag) / sy))

    # Q and the complex background are correlated.  Starting a robust loss
    # directly from a coarse seed can classify the resonance itself as an
    # outlier.  In that failure mode Q collapses toward its lower bound while
    # huge resonance/background terms cancel each other.  First locate the
    # least-squares basin from several Q seeds, then robustly refine only the
    # best physical basin.
    q_seeds = [q_guess, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    x_scale = np.array([max(span, 1.0), 1.0,
                        amplitude_scale, amplitude_scale, amplitude_scale, amplitude_scale,
                        amplitude_scale / max(span, 1.0), amplitude_scale / max(span, 1.0)])
    preliminary = None
    selected_sign = 1
    for resonance_sign in (1, -1):
        for q_seed in q_seeds:
            start = p0.copy()
            start[1] = np.clip(np.log(q_seed), lower[1] + 1e-12, upper[1] - 1e-12)
            candidate = least_squares(
                residuals, start, args=(resonance_sign,), bounds=(lower, upper), loss="linear",
                max_nfev=5000, x_scale=x_scale,
            )
            if preliminary is None or candidate.cost < preliminary.cost:
                preliminary = candidate
                selected_sign = resonance_sign

    preliminary_residuals = residuals(preliminary.x, selected_sign)
    residual_center = float(np.median(preliminary_residuals))
    residual_sigma = float(
        1.4826 * np.median(np.abs(preliminary_residuals - residual_center))
    )
    residual_rms = float(np.sqrt(np.mean(preliminary_residuals ** 2)))
    robust_scale = max(residual_sigma, 0.05 * residual_rms, np.finfo(float).eps)
    result = least_squares(
        residuals, preliminary.x, args=(selected_sign,), bounds=(lower, upper), loss="soft_l1",
        f_scale=robust_scale, max_nfev=5000, x_scale=x_scale,
    )
    p = result.x
    q = float(np.exp(p[1]))
    dof = max(1, 2 * len(f) - len(p))
    chi2 = float(np.sum(residuals(p, selected_sign) ** 2) / dof)
    covariance = np.full((len(p), len(p)), np.nan)
    try:
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * chi2
    except np.linalg.LinAlgError:
        pass
    f0_std = float(np.sqrt(max(covariance[0, 0], 0))) if np.isfinite(covariance[0, 0]) else float("nan")
    q_std = float(q * np.sqrt(max(covariance[1, 1], 0))) if np.isfinite(covariance[1, 1]) else float("nan")
    df = max(abs(p[0]) * 1e-7, span * 1e-6)
    z_minus = model_response(np.array([p[0] - df]), p, center, selected_sign)[0]
    z_plus = model_response(np.array([p[0] + df]), p, center, selected_sign)[0]
    phase_slope = np.angle(z_plus / z_minus) / (2.0 * df) * 180.0 / np.pi
    at_resonance = model_response(np.array([p[0]]), p, center, selected_sign)[0]
    return ResonanceFit(float(p[0]), q, float(p[0] / q),
                        float(np.angle(at_resonance, deg=True)), float(phase_slope),
                        f0_std, q_std, chi2, bool(result.success), result.message, p,
                        selected_sign)
