"""Bounded Savitzky-Golay smoothing for live detector snapshots."""
import numpy as np


def filter_frequency(time_s, frequency_hz, enabled=False, window_length=27, polyorder=4):
    """Return centered SG smoothing with offline-style sample-count settings.

    A centered window uses future samples in a bounded snapshot. The live
    detector already waits 0.6 s before examining a point, so its analysis
    region does not depend on incomplete samples at the trailing edge.
    """
    values = np.asarray(frequency_hz, dtype=float)
    if not enabled:
        return values.copy(), 0
    times = np.asarray(time_s, dtype=float)
    if len(times) != len(values) or len(times) < 3:
        raise ValueError("Savitzky-Golay 입력의 시간축과 주파수 길이가 맞지 않습니다.")
    order = int(polyorder)
    window = int(window_length)
    if window % 2 == 0:
        window += 1
    if order < 1 or order > 8 or window < 3 or window > len(values) or window <= order:
        raise ValueError("SG window는 데이터 길이 이하의 홀수이고 polynomial order보다 커야 합니다.")
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        pass
    else:
        # Use the same implementation and boundary mode as the offline panel.
        return savgol_filter(values, window_length=window,
                             polyorder=order, mode="interp"), window
    radius = window // 2
    offsets = np.arange(-radius, radius + 1, dtype=float)
    design = np.vander(offsets, N=order + 1, increasing=True)
    weights = np.linalg.pinv(design)[0]
    filtered = np.empty_like(values)
    filtered[radius:len(values) - radius] = np.convolve(values, weights[::-1], mode="valid")
    # Match scipy.signal.savgol_filter(..., mode='interp') at the boundaries.
    edge_x = np.arange(window, dtype=float)
    first_fit = np.polyfit(edge_x, values[:window], order)
    last_fit = np.polyfit(edge_x, values[-window:], order)
    filtered[:radius] = np.polyval(first_fit, edge_x[:radius])
    filtered[-radius:] = np.polyval(last_fit, edge_x[-radius:])
    return filtered, window
