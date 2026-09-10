"""Offline raw-event review. No instrument dependencies or hardware access."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, find_peaks


@dataclass
class Settings:
    widths: tuple = (.03, .05, .10, .20, .40, .80, 1.)
    reference_start: float = 1.
    reference_end: float = 6.
    analyze_start: float = 0.
    analyze_end: float = 1e9
    sigma_multiplier: float = 6.
    floor_hz: float = .15
    merge_s: float = .30


def robust_sigma(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def sample_window_stats(t, start_s, end_s):
    """Count raw timestamps in the closed interval; no resampling or N=fs*T approximation."""
    t=np.asarray(t,dtype=float)
    lo,hi=sorted((float(start_s),float(end_s)))
    if not np.isfinite([lo,hi]).all():raise ValueError('구간 경계는 유한한 숫자여야 합니다.')
    left=int(np.searchsorted(t,lo,side='left')); right=int(np.searchsorted(t,hi,side='right'))
    samples=t[left:right]; dt=np.diff(samples)
    median=float(np.median(dt)) if len(dt) else np.nan
    return dict(start_s=lo,end_s=hi,window_duration_s=hi-lo,sample_count=len(samples),
                first_sample_s=float(samples[0]) if len(samples) else np.nan,
                last_sample_s=float(samples[-1]) if len(samples) else np.nan,
                sample_span_s=float(samples[-1]-samples[0]) if len(samples)>1 else np.nan,
                median_interval_s=median,min_interval_s=float(np.min(dt)) if len(dt) else np.nan,
                max_interval_s=float(np.max(dt)) if len(dt) else np.nan,
                sampling_rate_hz=1/median if median>0 else np.nan,
                large_gap_count=int(np.sum(dt>1.5*median)) if median>0 else 0)


def load_raw(path):
    metadata = {}
    with open(path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() == '# Metadata' and len(row) >= 3:
                metadata[row[1]] = ','.join(row[2:])
    d = pd.read_csv(path, comment='#', encoding='utf-8-sig')
    d.columns = d.columns.str.strip()
    if not {'Time_s', 'Frequency_Hz'} <= set(d.columns):
        raise ValueError('Time_s, Frequency_Hz 열이 필요합니다.')
    for col in d.columns:
        if col not in ('SessionID', 'FlowDirection', 'SwitchState'):
            d[col] = pd.to_numeric(d[col], errors='coerce')
    t, f = d.Time_s.to_numpy(), d.Frequency_Hz.to_numpy()
    if len(t) < 20 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        raise ValueError('시간·주파수에 결측값이 있거나 표본이 20개 미만입니다.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('시간이 중복되거나 역전되었습니다. 원본 시간축을 먼저 확인하세요.')
    return d, metadata


def pressure_changes(d):
    cols = ['PressureCommand_A_V', 'PressureCommand_B_V']
    if not all(c in d for c in cols):
        return pd.DataFrame(columns=['Time_s'] + cols)
    x = d[cols].to_numpy()
    ok = np.isfinite(x).all(axis=1)
    indices = np.flatnonzero(ok[1:] & ok[:-1] & np.any(abs(np.diff(x, axis=0)) > 1e-8, axis=1)) + 1
    return d.iloc[indices][['Time_s'] + cols].reset_index(drop=True)


def local_baseline(d, center):
    x = d.Time_s.to_numpy() - center
    mask = (abs(x) >= .25) & (abs(x) <= .55)
    if sum(mask) < 6 or not np.any(x[mask] < 0) or not np.any(x[mask] > 0):
        raise ValueError('최저점 양쪽에 배경 피팅용 표본이 부족합니다 (각 0.55초 필요).')
    return np.polyval(np.polyfit(x[mask], d.Frequency_Hz.to_numpy()[mask], 2), x)


def manual_event(d, center):
    t = d.Time_s.to_numpy()
    base = local_baseline(d, center)
    i = int(np.argmin(abs(t-center)))
    changes = pressure_changes(d).Time_s.to_numpy()
    prev = changes[changes <= t[i]]
    return dict(time_s=float(t[i]), width_s=0., contrast_hz=np.nan,
                depth_hz=float(base[i]-d.Frequency_Hz.iloc[i]),
                reference_ratio=np.nan, family='manual',
                pressure_delay_s=float(t[i]-prev[-1]) if len(prev) else np.nan,
                status='미검토', note='수동 추가')


def detect(d, settings):
    t, f = d.Time_s.to_numpy(), d.Frequency_Hz.to_numpy()
    dt = float(np.median(np.diff(t)))
    # Never convolve over dropped samples or silently resample raw data.
    if np.max(abs(np.diff(t)-dt)) > .05*dt:
        raise ValueError('표본 간격이 5% 이상 불규칙하거나 누락 구간이 있습니다. 그래프 열람은 가능하지만 자동 분석은 지원하지 않습니다.')
    ref = (t >= settings.reference_start) & (t <= settings.reference_end)
    if sum(ref) < 30:
        raise ValueError('잡음 기준 구간에 최소 30표본이 필요합니다.')
    if settings.analyze_end <= settings.analyze_start:
        raise ValueError('분석 종료 시간은 시작 시간보다 커야 합니다.')
    hits, thresholds = [], []
    for width in settings.widths:
        if not np.isfinite(width) or not 0 < width <= 30:
            raise ValueError('탐색 폭은 0초 초과, 30초 이하로 입력하세요.')
        m = max(3, round(width/dt)); m += 1-m%2
        g = max(2, round(.03/dt))
        kernel = np.r_[np.ones(m)/(2*m), np.zeros(g), -np.ones(m)/m,
                       np.zeros(g), np.ones(m)/(2*m)]
        edge = len(kernel)//2
        if 2*edge >= len(t):
            raise ValueError(f'{width:g}초 탐색 폭에 비해 기록이 너무 짧습니다.')
        valid = np.arange(len(t))
        valid = (valid >= edge) & (valid < len(t)-edge)
        ref_valid = ref & valid
        if sum(ref_valid) < 30:
            raise ValueError(f'{width:g}초 폭의 기준 구간이 기록 경계와 겹칩니다. 기준 구간을 넓히거나 옮기세요.')
        score = fftconvolve(f, kernel, mode='same')
        sigma = robust_sigma(score[ref_valid])
        threshold = max(settings.floor_hz, settings.sigma_multiplier*sigma)
        thresholds.append(dict(width_s=m*dt, sigma_hz=sigma, threshold_hz=threshold))
        valid &= (t >= settings.analyze_start) & (t <= settings.analyze_end)
        peaks, _ = find_peaks(np.where(valid, score, -np.inf), height=threshold,
                              distance=max(1, round(.10/dt)))
        for i in peaks:
            hits.append(dict(time_s=float(t[i]), width_s=m*dt,
                             contrast_hz=float(score[i]), reference_ratio=float(score[i]/max(sigma,1e-12)),
                             family='short' if width <= .20 else 'wide'))
    events = []
    for family in ('short', 'wide'):
        chosen = []
        separation = settings.merge_s if family == 'short' else max(.8, settings.merge_s)
        for h in sorted((h for h in hits if h['family']==family), key=lambda h:h['reference_ratio'], reverse=True):
            if all(abs(h['time_s']-c['time_s']) >= separation for c in chosen):
                chosen.append(h)
        for h in chosen:
            center = h['time_s']
            # Local raw minimum around detected contrast; same narrow-window
            # characterization used in the preceding offline reviews.
            half=max(.10,h['width_s']/2)
            offset=t-center
            base_mask=(abs(offset)>=half+.04)&(abs(offset)<=half+.04+max(.20,h['width_s']))
            inside=(abs(offset)<=half)&(t>=settings.analyze_start)&(t<=settings.analyze_end)
            if sum(base_mask)<6 or not np.any(inside):
                continue
            r=f-np.polyval(np.polyfit(offset[base_mask], f[base_mask],1),offset)
            ids=np.flatnonzero(inside); j=ids[np.argmin(r[ids])]
            try:
                detail=manual_event(d,float(t[j]))
            except ValueError:
                continue
            events.append({**detail, **h, 'time_s':float(t[j]),
                           'depth_hz':detail['depth_hz'], 'note':''})
    return sorted(events,key=lambda h:h['time_s']), thresholds


def settings_dict(s):
    return asdict(s)


def _noise_metrics(t, f, label, start, end, degree=1):
    """Quantify raw and detrended frequency noise in one context window."""
    mask=(t>=start)&(t<=end)
    tt=t[mask]; ff=f[mask]
    if len(tt)<8 or np.ptp(tt)<=0:
        return dict(context=label,start_s=float(start),end_s=float(end),sample_count=int(len(tt)),
                    mean_frequency_hz=np.nan,raw_std_hz=np.nan,drift_hz_per_min=np.nan,
                    detrended_std_hz=np.nan,peak_to_peak_residual_hz=np.nan,
                    threshold_10sigma_hz=np.nan,status='insufficient_samples')
    coeff=np.polyfit(tt,ff,degree)
    drift=np.polyval(coeff,tt); residual=ff-drift
    return dict(context=label,start_s=float(start),end_s=float(end),sample_count=int(len(tt)),
                mean_frequency_hz=float(np.mean(ff)),raw_std_hz=float(np.std(ff)),
                drift_hz_per_min=float((coeff[-2] if degree>=1 else 0)*60),
                detrended_std_hz=float(np.std(residual)),
                peak_to_peak_residual_hz=float(np.ptp(residual)),
                threshold_10sigma_hz=float(10*np.std(residual)),status='ok')


def event_context_noise(data, center_s, window_s=1.0, guard_s=.10, detrend_degree=1):
    """Return equal-duration before/after noise metrics around an event.

    The event center ± guard is excluded. Windows are clipped to the recording.
    No interpolation or smoothing is applied. Metrics are in Hz RMS.
    """
    if window_s<=0 or guard_s<0 or detrend_degree not in (1,2):
        raise ValueError('window_s는 양수, guard_s는 0 이상, detrend_degree는 1 또는 2여야 합니다.')
    t=data.Time_s.to_numpy(dtype=float); f=data.Frequency_Hz.to_numpy(dtype=float)
    t0=float(t[0]); t1=float(t[-1]); c=float(center_s)
    pre_start=max(t0,c-guard_s-window_s); pre_end=min(t1,c-guard_s)
    post_start=max(t0,c+guard_s); post_end=min(t1,c+guard_s+window_s)
    rows=[_noise_metrics(t,f,'before',pre_start,pre_end,detrend_degree),
          _noise_metrics(t,f,'after',post_start,post_end,detrend_degree)]
    for row in rows:
        row['event_time_s']=c; row['window_requested_s']=float(window_s); row['guard_s']=float(guard_s)
        row['detrend_method']=f'polynomial_degree_{detrend_degree}'
    return rows


def batch_context_noise(data, events, window_s=1.0, guard_s=.10, detrend_degree=1):
    """Quantify equal before/after contexts for all retained event rows."""
    rows=[]
    for index,event in enumerate(events,1):
        for row in event_context_noise(data,float(event['time_s']),window_s,guard_s,detrend_degree):
            rows.append({'event_index':index,**event,**row})
    return pd.DataFrame(rows)
