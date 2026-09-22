"""MIT-paper-compatible inter-peak frequency noise analysis.

This module intentionally does not change the existing local before/after
noise/SNR calculation.  It implements the reported method used for Fig. 4:
inter-peak baseline -> 1 Hz, second-order Butterworth high-pass -> remove the
segment mean -> time-domain sample standard deviation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


PAPER_VISUAL_REFERENCE_HZ = 0.0225


def _event_bounds(event, guard_s):
    """Return the excluded event interval, including a user-selected guard."""
    center = float(event['time_s'])
    start = float(event.get('event_start_s', center))
    end = float(event.get('event_end_s', center))
    if not np.isfinite(start):
        start = center
    if not np.isfinite(end):
        end = center
    return min(start, end) - guard_s, max(start, end) + guard_s


def _sampling_rate(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return np.nan, np.nan
    median_dt = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - median_dt)) / median_dt)
    return 1.0 / median_dt, jitter


def _empty_segment(index, start, end, status, warning):
    return dict(segment_index=index, start_s=float(start), end_s=float(end),
                duration_s=float(max(0., end-start)), sample_count=0,
                sampling_rate_hz=np.nan, raw_std_hz=np.nan, sigma_mit_hz=np.nan,
                status=status, warning=warning)


def analyze_mit_noise(data, events, guard_s=0.25, cutoff_hz=1.0, order=2,
                      min_duration_s=1.0, reference_hz=PAPER_VISUAL_REFERENCE_HZ):
    """Calculate Fig.-4-style baseline noise from consecutive event gaps.

    A segment is the interval from the end of one event's exclusion region to
    the start of the next.  Segments are filtered independently, so an event
    cannot bleed through a filter state into its neighbour.
    """
    if len(events) < 2:
        raise ValueError('MIT 비교에는 피크 사이 baseline을 만들기 위해 후보가 최소 2개 필요합니다.')
    if guard_s < 0 or cutoff_hz <= 0 or order != 2 or min_duration_s <= 0:
        raise ValueError('guard는 0 이상, cutoff/최소 길이는 양수, MIT filter order는 2여야 합니다.')
    required = {'Time_s', 'Frequency_Hz'}
    if not required <= set(data.columns):
        raise ValueError('원본에 Time_s, Frequency_Hz 열이 필요합니다.')

    d = data.loc[:, ['Time_s', 'Frequency_Hz']].dropna().sort_values('Time_s')
    t_all = d.Time_s.to_numpy(dtype=float)
    f_all = d.Frequency_Hz.to_numpy(dtype=float)
    if len(t_all) < 8 or np.any(np.diff(t_all) <= 0):
        raise ValueError('시간 열은 중복 없이 증가하는 유효 표본을 포함해야 합니다.')

    ordered = sorted(events, key=lambda e: float(e['time_s']))
    bounds = [_event_bounds(event, float(guard_s)) for event in ordered]
    rows, residual_rows = [], []
    for segment_index, ((_, left_end), (right_start, _)) in enumerate(zip(bounds, bounds[1:]), 1):
        start, end = left_end, right_start
        if end <= start:
            rows.append(_empty_segment(segment_index, start, end, 'overlap', '인접 후보의 제외 구간이 겹칩니다.'))
            continue
        mask = (t_all >= start) & (t_all <= end)
        t, f = t_all[mask], f_all[mask]
        if end - start < min_duration_s:
            rows.append(_empty_segment(segment_index, start, end, 'too_short', f'구간이 최소 길이 {min_duration_s:g} s보다 짧습니다.'))
            continue
        if len(t) < 8:
            rows.append(_empty_segment(segment_index, start, end, 'insufficient_samples', '필터링에 필요한 표본이 부족합니다.'))
            continue
        fs, jitter = _sampling_rate(t)
        if not np.isfinite(fs) or fs <= 2 * cutoff_hz:
            rows.append(_empty_segment(segment_index, start, end, 'invalid_sampling', 'sampling rate가 1 Hz high-pass에 충분하지 않습니다.'))
            continue
        if jitter > .05:
            rows.append(_empty_segment(segment_index, start, end, 'irregular_sampling', f'표본 간격 변동 {jitter * 100:.1f}%: 보간 없이 공정 비교를 할 수 없습니다.'))
            continue
        try:
            sos = butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
            filtered = sosfiltfilt(sos, f)
        except ValueError as exc:
            rows.append(_empty_segment(segment_index, start, end, 'filter_failed', str(exc)))
            continue
        residual = filtered - np.mean(filtered)
        sigma = float(np.std(residual, ddof=1))
        warning = '짧은 구간: high-pass edge 영향 가능' if end-start < 3/cutoff_hz else ''
        rows.append(dict(segment_index=segment_index, start_s=float(start), end_s=float(end),
                         duration_s=float(end-start), sample_count=int(len(t)), sampling_rate_hz=float(fs),
                         raw_std_hz=float(np.std(f, ddof=1)), sigma_mit_hz=sigma,
                         status='ok', warning=warning))
        residual_rows.extend(dict(segment_index=segment_index, time_s=float(tt), frequency_hz=float(ff),
                                  filtered_hz=float(hh), residual_hz=float(rr))
                             for tt, ff, hh, rr in zip(t, f, filtered, residual))

    segments = pd.DataFrame(rows)
    residuals = pd.DataFrame(residual_rows, columns=['segment_index', 'time_s', 'frequency_hz', 'filtered_hz', 'residual_hz'])
    valid = segments[segments.status == 'ok']
    if len(residuals):
        # Each segment mean is zero. ddof=1 matches the conventional MATLAB
        # std definition and is explicit in saved provenance.
        pooled_sigma = float(np.std(residuals.residual_hz.to_numpy(), ddof=1))
    else:
        pooled_sigma = np.nan
    return dict(
        segments=segments, residuals=residuals,
        summary=dict(pooled_sigma_hz=pooled_sigma,
                     median_segment_sigma_hz=float(valid.sigma_mit_hz.median()) if len(valid) else np.nan,
                     valid_segment_count=int(len(valid)), total_segment_count=int(len(segments)),
                     total_baseline_duration_s=float(valid.duration_s.sum()) if len(valid) else 0.,
                     paper_reference_hz=float(reference_hz),
                     paper_ratio=float(pooled_sigma/reference_hz) if np.isfinite(pooled_sigma) and reference_hz > 0 else np.nan),
        settings=dict(guard_s=float(guard_s), cutoff_hz=float(cutoff_hz), order=int(order),
                      min_duration_s=float(min_duration_s), std_ddof=1,
                      filtering='per-segment scipy.signal.sosfiltfilt (zero phase); no polynomial detrending',
                      method='inter-peak baseline; high-pass; subtract segment mean; time-domain SD',
                      paper_reference_hz=float(reference_hz),
                     paper_reference_note='Supplementary Fig. S5 all-data mean; not an acceptance threshold'),
    )


def save_mit_noise_analysis(folder, analysis, source, metadata=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    analysis['segments'].to_csv(folder/'mit_noise_segments.csv', index=False, encoding='utf-8-sig')
    analysis['residuals'].to_csv(folder/'mit_noise_residuals.csv', index=False, encoding='utf-8-sig')
    payload = dict(source=str(source), metadata=metadata or {}, summary=analysis['summary'], settings=analysis['settings'])
    (folder/'mit_noise_analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')
    summary = analysis['summary']
    report = [
        '# MIT 논문 방식 노이즈 비교', '', f'원본: `{source}`', '',
        '피크 사이 baseline에 1 Hz, 2차 Butterworth high-pass를 적용하고 각 구간 평균을 제거한 뒤 시간영역 표준편차(ddof=1)를 계산했습니다.', '',
        f"통합 σ: {summary['pooled_sigma_hz']:.6g} Hz", f"유효 구간: {summary['valid_segment_count']} / {summary['total_segment_count']}",
        f"Supplement Fig. S5 전체 평균 ≈ {summary['paper_reference_hz']:.4g} Hz 대비: {summary['paper_ratio']:.3g}×", '',
        '논문 평균값은 합격/불합격 기준이 아닙니다.', '',
    ]
    (folder/'mit_noise_report.md').write_text('\n'.join(report), encoding='utf-8')
    return folder
