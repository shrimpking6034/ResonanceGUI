"""Hardware-independent calculations used by the resonance sweep engine."""
from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincc


@dataclass(frozen=True)
class FilterPlan:
    order: int
    time_constant_s: float
    bandwidth_3db_hz: float
    settling_s: float
    measurement_s: float


def bandwidth_3db_from_tc(time_constant_s, order):
    if time_constant_s <= 0 or not 1 <= order <= 8:
        raise ValueError("유효한 TC와 필터 차수가 필요합니다.")
    return float(np.sqrt(2.0 ** (1.0 / order) - 1.0) / (2.0 * pi * time_constant_s))


def tc_from_bandwidth_3db(bandwidth_hz, order):
    if bandwidth_hz <= 0 or not 1 <= order <= 8:
        raise ValueError("유효한 bandwidth와 필터 차수가 필요합니다.")
    return float(np.sqrt(2.0 ** (1.0 / order) - 1.0) / (2.0 * pi * bandwidth_hz))


def settling_multiplier(order, residual):
    """TC multiples for an n-pole critically damped step response residual."""
    if not 1 <= order <= 8:
        raise ValueError("필터 차수는 1~8이어야 합니다.")
    if not 1e-13 <= residual <= 0.1:
        raise ValueError("정착 오차는 1e-13~0.1 범위여야 합니다.")
    # gammaincc(n, u) = exp(-u) sum_{k=0}^{n-1} u^k/k!.
    return float(brentq(lambda u: gammaincc(order, u) - residual, 0.0, 100.0))


def auto_time_constant(frequency_hz, step_hz, order, max_bandwidth_hz,
                       omega_suppression_db, bandwidth_overlap):
    """Fastest TC satisfying 2-omega rejection and bandwidth constraints."""
    if frequency_hz <= 0 or max_bandwidth_hz <= 0:
        raise ValueError("주파수와 최대 bandwidth는 0보다 커야 합니다.")
    max_allowed_bw = max_bandwidth_hz
    if not bandwidth_overlap and step_hz > 0:
        max_allowed_bw = min(max_allowed_bw, abs(step_hz) * 0.5)
    # Require the n-pole filter attenuation at 2*f to meet the requested dB.
    ratio = 10.0 ** (omega_suppression_db / (10.0 * order)) - 1.0
    tc_for_suppression = np.sqrt(max(ratio, 0.0)) / (2.0 * pi * 2.0 * frequency_hz)
    tc_for_max_bw = tc_from_bandwidth_3db(max_allowed_bw, order)
    return float(max(tc_for_suppression, tc_for_max_bw))


def make_filter_plan(*, frequency_hz, step_hz, order, mode, manual_tc_s,
                     fixed_bandwidth_hz, max_bandwidth_hz,
                     omega_suppression_db, bandwidth_overlap,
                     min_settling_s, settling_residual,
                     sample_count, sample_rate_hz, averaging_time_s,
                     averaging_tc):
    """Build the effective point-by-point plan, matching LabOne max rules."""
    mode = mode.lower()
    if mode == "manual":
        tc = manual_tc_s
    elif mode == "fixed":
        tc = tc_from_bandwidth_3db(fixed_bandwidth_hz, order)
    elif mode == "auto":
        tc = auto_time_constant(frequency_hz, step_hz, order, max_bandwidth_hz,
                                omega_suppression_db, bandwidth_overlap)
    else:
        raise ValueError("필터 모드는 Manual, Fixed, Auto 중 하나여야 합니다.")
    settling = max(min_settling_s, settling_multiplier(order, settling_residual) * tc)
    sample_rate_hz = max(float(sample_rate_hz), 1e-6)
    measurement = max(sample_count / sample_rate_hz, averaging_time_s, averaging_tc * tc)
    return FilterPlan(order, float(tc), bandwidth_3db_from_tc(tc, order),
                      float(settling), float(measurement))


def average_complex_samples(x, y):
    """Average quadratures first; return mean, SEMs and covariance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        raise ValueError("평균할 유효 복소 샘플이 없습니다.")
    z = x + 1j * y
    mean = complex(np.mean(z))
    if len(z) < 2:
        return mean, float("nan"), float("nan"), float("nan"), 1
    covariance = np.cov(np.vstack((x, y)), ddof=1) / len(z)
    return mean, float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1])), float(covariance[0, 1]), len(z)
