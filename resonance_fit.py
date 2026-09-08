"""Complex second-order resonator fitting for HF2 demodulator data."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


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


def model_response(frequency_hz, parameters, center_hz=None):
    """Complex background plus an exact second-order driven resonance."""
    f = np.asarray(frequency_hz, dtype=float)
    f0, log_q, ar, ai, b0r, b0i, b1r, b1i = parameters
    q = np.exp(log_q)
    center_hz = f0 if center_hz is None else center_hz
    resonance = (ar + 1j * ai) / (1.0 - (f / f0) ** 2 + 1j * f / (q * f0))
    background = (b0r + 1j * b0i) + (b1r + 1j * b1i) * (f - center_hz)
    return resonance + background


def fit_complex_resonance(frequency_hz, x, y, x_sem=None, y_sem=None,
                          f0_guess=None, f0_bounds=None):
    """Fit one complex resonance.

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

    def residuals(p):
        model = model_response(f, p, center)
        return np.concatenate(((model.real - z.real) / sx, (model.imag - z.imag) / sy))

    # Q and the complex background are correlated.  Use several physically
    # spaced Q seeds, then retain the lowest robust cost rather than trusting
    # a single coarse FWHM estimate.
    q_seeds = [q_guess, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    x_scale = np.array([max(span, 1.0), 1.0,
                        amplitude_scale, amplitude_scale, amplitude_scale, amplitude_scale,
                        amplitude_scale / max(span, 1.0), amplitude_scale / max(span, 1.0)])
    result = None
    for q_seed in q_seeds:
        start = p0.copy()
        start[1] = np.clip(np.log(q_seed), lower[1] + 1e-12, upper[1] - 1e-12)
        candidate = least_squares(
            residuals, start, bounds=(lower, upper), loss="soft_l1",
            f_scale=max(amplitude_scale * 0.05, 1e-12), max_nfev=5000, x_scale=x_scale,
        )
        if result is None or candidate.cost < result.cost:
            result = candidate
    p = result.x
    q = float(np.exp(p[1]))
    model = model_response(f, p, center)
    dof = max(1, 2 * len(f) - len(p))
    chi2 = float(np.sum(residuals(p) ** 2) / dof)
    covariance = np.full((len(p), len(p)), np.nan)
    try:
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * chi2
    except np.linalg.LinAlgError:
        pass
    f0_std = float(np.sqrt(max(covariance[0, 0], 0))) if np.isfinite(covariance[0, 0]) else float("nan")
    q_std = float(q * np.sqrt(max(covariance[1, 1], 0))) if np.isfinite(covariance[1, 1]) else float("nan")
    df = max(abs(p[0]) * 1e-7, span * 1e-6)
    z_minus = model_response(np.array([p[0] - df]), p, center)[0]
    z_plus = model_response(np.array([p[0] + df]), p, center)[0]
    phase_slope = np.angle(z_plus / z_minus) / (2.0 * df) * 180.0 / np.pi
    at_resonance = model_response(np.array([p[0]]), p, center)[0]
    return ResonanceFit(float(p[0]), q, float(p[0] / q),
                        float(np.angle(at_resonance, deg=True)), float(phase_slope),
                        f0_std, q_std, chi2, bool(result.success), result.message, p)
