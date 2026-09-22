"""Select the stable raw-error zero for the deployed SMR channel-A PLL.

The channel-A experiments at 29.38 and 39.32 kHz established that the
positive error-versus-delay crossing is the stable closed-loop branch.  The
stream is raw FPGA counts; no degree conversion or lock claim is made here.
"""
from __future__ import annotations

import numpy as np


def positive_error_zero(delays, errors, noise_counts=0.0):
    """Return (delay turns, span counts) from a uniform cyclic 0..1 scan.

    Raises ValueError when the selected error stream has no credible positive
    crossing; starting a closed loop with an arbitrary delay would rail it.
    """
    x = np.asarray(delays, dtype=float)
    y = np.asarray(errors, dtype=float)
    if len(x) < 8 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("PLL delay 보정 데이터가 부족하거나 유효하지 않습니다.")
    step = 1.0 / len(x)
    if not np.allclose(x, np.arange(len(x)) * step, atol=1e-9):
        raise ValueError("PLL delay 보정은 균일한 0..1 주기 스캔이 필요합니다.")
    span = float(np.max(y) - np.min(y))
    if span <= max(1000.0, 8.0 * float(noise_counts)):
        raise ValueError("원시 error의 delay 응답이 잡음과 구별되지 않습니다.")
    crossings = []
    for index in range(len(x)):
        next_index = (index + 1) % len(x)
        low, high = y[index], y[next_index]
        if low <= 0 < high:
            fraction = -low / (high - low)
            delay = (x[index] + fraction * step) % 1.0
            crossings.append((high - low, float(delay)))
    if not crossings:
        raise ValueError("양의 기울기 원시 error 영점을 찾지 못했습니다.")
    return max(crossings)[1], span


def refine_positive_error_zero(coarse_delay, delays, errors, noise_counts=0.0):
    """Interpolate the positive crossing inside one unwrapped coarse interval.

    ``delays`` may extend beyond one turn at the cyclic boundary. Refuse a
    weak or missing crossing instead of falling back to the coarse estimate.
    """
    x = np.asarray(delays, dtype=float)
    y = np.asarray(errors, dtype=float)
    if (len(x) < 3 or x.shape != y.shape or not np.all(np.isfinite(x)) or
            not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0)):
        raise ValueError("PLL delay 세밀 스캔 데이터가 부족하거나 유효하지 않습니다.")
    noise = max(0.0, float(noise_counts))
    span = float(np.ptp(y))
    if span <= max(1000.0, 8.0 * noise):
        raise ValueError("PLL delay 세밀 스캔 응답이 잡음과 구별되지 않습니다.")
    crossings = []
    for index in range(len(x) - 1):
        low, high = y[index:index + 2]
        if low <= 0 < high and high - low > max(1000.0, 4.0 * noise):
            zero = x[index] - low * (x[index + 1] - x[index]) / (high - low)
            distance = abs((zero - coarse_delay + 0.5) % 1.0 - 0.5)
            crossings.append((distance, zero % 1.0))
    if not crossings:
        raise ValueError("PLL delay 세밀 스캔에서 양의 기울기 영점을 재확인하지 못했습니다.")
    return min(crossings)[1], span
