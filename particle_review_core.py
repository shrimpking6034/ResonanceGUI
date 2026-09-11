"""Offline raw-event review. No instrument dependencies or hardware access."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, find_peaks


@dataclass
class Settings:
    widths: tuple = (.03, .05, .10, .20)
    reference_start: float = 1.
    reference_end: float = 6.
    analyze_start: float = 0.
    analyze_end: float = 1e9
    sigma_multiplier: float = 6.
    floor_hz: float = .15
    merge_s: float = .30


@dataclass
class ParticlePropertySettings:
    """Inputs for converting a resonance-depth event into particle properties.

    ``calibration_method='reference'`` calibrates pg/Hz from the selected
    representative of round-trip pair means. ``'sensitivity'`` uses a
    separately measured sensitivity directly and is appropriate for unknown
    samples after a reference run.
    """
    mode_number: int = 6
    diameter_min_um: float = 190.
    diameter_max_um: float = 220.
    fluid_density_kg_m3: float = 997.
    calibration_method: str = 'reference'
    reference_density_kg_m3: float = 7930.
    sensitivity_pg_hz: float = 1.
    representative_method: str = 'mean'
    inference_method: str = 'mass_only'


def load_sensitivity_profiles(path):
    """Load persistent, mode-specific mass-sensitivity calibrations."""
    path=Path(path)
    if not path.exists():
        return {}
    payload=json.loads(path.read_text(encoding='utf-8'))
    if payload.get('version') != 1 or not isinstance(payload.get('calibrations'),dict):
        raise ValueError('저장된 감도 보정 파일 형식이 올바르지 않습니다.')
    profiles={}
    for key,value in payload['calibrations'].items():
        if not isinstance(value,dict):continue
        try:
            mode=int(value['mode_number']); sensitivity=float(value['sensitivity_pg_hz'])
        except (KeyError,TypeError,ValueError):
            continue
        if mode >= 1 and np.isfinite(sensitivity) and sensitivity > 0:
            profiles[str(mode)]={**value,'mode_number':mode,'sensitivity_pg_hz':sensitivity}
    return profiles


def save_sensitivity_profile(path, calibration, source_file=None):
    """Persist one reference calibration without discarding other modes."""
    mode=int(calibration['mode_number']); sensitivity=float(calibration['sensitivity_pg_hz'])
    if mode < 1 or not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError('저장할 감도 보정값이 올바르지 않습니다.')
    path=Path(path)
    profiles=load_sensitivity_profiles(path) if path.exists() else {}
    profile={key:value for key,value in calibration.items()
             if isinstance(value,(str,int,float,bool)) and (not isinstance(value,float) or np.isfinite(value))}
    profile.update(mode_number=mode,sensitivity_pg_hz=sensitivity,
                   calibrated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                   source_file=str(Path(source_file).resolve()) if source_file else None)
    profiles[str(mode)]=profile
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'version':1,'calibrations':profiles},ensure_ascii=False,indent=2),encoding='utf-8')
    return profile


def estimate_particle_properties(events, settings):
    """Summarize one particle from consecutive forward/reverse event pairs.

    Individual events remain quality-control observations.  Absolute mass is
    calculated once from the representative of the pair means.  Density is
    inferred only when diameter is supplied as the external known quantity;
    equivalent diameter is inferred only when particle density is externally
    known.  Positive ``depth_hz`` denotes a downward frequency shift.
    """
    if not isinstance(settings, ParticlePropertySettings):
        settings = ParticlePropertySettings(**settings)
    if settings.mode_number < 1:
        raise ValueError('모드 번호는 1 이상이어야 합니다.')
    d_min=float(settings.diameter_min_um); d_max=float(settings.diameter_max_um)
    if not np.isfinite([d_min,d_max]).all() or d_min <= 0 or d_max < d_min:
        raise ValueError('입자 지름은 0보다 커야 하며 최댓값은 최솟값 이상이어야 합니다.')
    rho_f=float(settings.fluid_density_kg_m3)
    if not np.isfinite(rho_f) or rho_f < 0:
        raise ValueError('유체 밀도는 0 이상의 유한한 값이어야 합니다.')
    d_nominal=(d_min+d_max)/2.
    volume=lambda diameter_um: np.pi/6.*(diameter_um*1e-6)**3
    v_min=volume(d_min); v_nominal=volume(d_nominal); v_max=volume(d_max)
    depths=np.asarray([float(e.get('depth_hz',np.nan)) for e in events],dtype=float)
    valid=np.isfinite(depths)&(depths>0)
    if not np.any(valid):
        raise ValueError('양의 주파수 깊이(depth_hz)를 가진 후보가 필요합니다.')
    representative_method=str(settings.representative_method).strip().lower()
    if representative_method not in ('mean','median'):
        raise ValueError("대표값 방식은 'mean' 또는 'median'이어야 합니다.")
    inference_method=str(settings.inference_method).strip().lower()
    if inference_method not in ('mass_only','density_from_diameter','diameter_from_density'):
        raise ValueError('물성 계산 방식이 올바르지 않습니다.')

    # Pair strictly by the retained event order.  A missing/invalid member must
    # not shift all later directions by one, so incomplete pairs are excluded.
    enriched=[dict(event) for event in events]
    pair_rows=[]
    for start in range(0,len(events),2):
        pair_index=start//2+1
        indices=list(range(start,min(start+2,len(events))))
        pair_depths=depths[indices]
        complete=len(indices)==2 and bool(np.all(np.isfinite(pair_depths)&(pair_depths>0)))
        pair_depth=float(np.mean(pair_depths)) if complete else np.nan
        for offset,index in enumerate(indices):
            row=enriched[index]
            for key in ('buoyant_mass_pg','density_nominal_kg_m3','density_low_kg_m3',
                        'density_high_kg_m3','dry_mass_nominal_pg','equivalent_diameter_um'):
                row.pop(key,None)
            row.update(round_trip_pair=pair_index,
                       direction='forward' if offset==0 else 'reverse',
                       pair_depth_hz=pair_depth,pair_complete=complete)
        if complete:
            pair_rows.append(dict(pair_index=pair_index,
                                  forward_depth_hz=float(pair_depths[0]),
                                  reverse_depth_hz=float(pair_depths[1]),
                                  pair_depth_hz=pair_depth))
    if not pair_rows:
        raise ValueError('대표값 계산에는 양의 depth_hz를 가진 완전한 왕복쌍이 최소 1개 필요합니다.')
    pair_depths=np.asarray([row['pair_depth_hz'] for row in pair_rows],dtype=float)
    representative_depth=float(np.mean(pair_depths) if representative_method=='mean' else np.median(pair_depths))

    def sample_stats(values):
        values=np.asarray(values,dtype=float)
        mean=float(np.mean(values)); count=len(values)
        std=float(np.std(values,ddof=1)) if count>1 else np.nan
        return dict(mean_hz=mean,median_hz=float(np.median(values)),std_hz=std,
                    sem_hz=float(std/np.sqrt(count)) if count>1 else np.nan,
                    cv_percent=float(std/mean*100) if count>1 and mean else np.nan,
                    count=int(count))

    individual_stats=sample_stats(depths[valid])
    pair_stats=sample_stats(pair_depths)
    forward=np.asarray([row['forward_depth_hz'] for row in pair_rows])
    reverse=np.asarray([row['reverse_depth_hz'] for row in pair_rows])
    direction_mean_forward=float(np.mean(forward)); direction_mean_reverse=float(np.mean(reverse))
    direction_difference_percent=(abs(direction_mean_forward-direction_mean_reverse)
                                  /individual_stats['mean_hz']*100)
    method=str(settings.calibration_method).strip().lower()
    if method == 'reference':
        rho_ref=float(settings.reference_density_kg_m3)
        if not np.isfinite(rho_ref) or rho_ref <= rho_f:
            raise ValueError('기준 입자 밀도는 유체 밀도보다 커야 합니다.')
        reference_buoyant_mass_kg=(rho_ref-rho_f)*v_nominal
        sensitivity_pg_hz=reference_buoyant_mass_kg/representative_depth*1e15
    elif method == 'sensitivity':
        sensitivity_pg_hz=float(settings.sensitivity_pg_hz)
        if not np.isfinite(sensitivity_pg_hz) or sensitivity_pg_hz <= 0:
            raise ValueError('질량 감도는 0보다 큰 유한한 값이어야 합니다.')
        reference_buoyant_mass_kg=np.nan
    else:
        raise ValueError("보정 방식은 'reference' 또는 'sensitivity'여야 합니다.")

    representative_buoyant_mass_pg=representative_depth*sensitivity_pg_hz
    representative_buoyant_mass_kg=representative_buoyant_mass_pg*1e-15
    density_nominal=density_low=density_high=equivalent_um=np.nan
    if inference_method=='density_from_diameter':
        density_nominal=rho_f+representative_buoyant_mass_kg/v_nominal
        density_low=rho_f+representative_buoyant_mass_kg/v_max
        density_high=rho_f+representative_buoyant_mass_kg/v_min
    elif inference_method=='diameter_from_density':
        rho_known=float(settings.reference_density_kg_m3)
        if not np.isfinite(rho_known) or rho_known <= rho_f:
            raise ValueError('알려진 입자 밀도는 유체 밀도보다 커야 합니다.')
        equivalent_um=(6*representative_buoyant_mass_kg/(np.pi*(rho_known-rho_f)))**(1/3)*1e6
    calibration=dict(mode_number=int(settings.mode_number),calibration_method=method,
                     sensitivity_pg_hz=float(sensitivity_pg_hz),
                     representative_method=representative_method,
                     representative_depth_hz=representative_depth,
                     representative_buoyant_mass_pg=float(representative_buoyant_mass_pg),
                     inference_method=inference_method,
                     representative_density_kg_m3=float(density_nominal),
                     representative_density_low_kg_m3=float(density_low),
                     representative_density_high_kg_m3=float(density_high),
                     representative_equivalent_diameter_um=float(equivalent_um),
                     pair_rows=pair_rows,individual_stats=individual_stats,pair_stats=pair_stats,
                     direction_mean_forward_hz=direction_mean_forward,
                     direction_mean_reverse_hz=direction_mean_reverse,
                     direction_difference_percent=float(direction_difference_percent),
                     diameter_min_um=d_min,diameter_nominal_um=d_nominal,diameter_max_um=d_max,
                     fluid_density_kg_m3=rho_f,reference_density_kg_m3=float(settings.reference_density_kg_m3),
                     reference_buoyant_mass_pg=float(reference_buoyant_mass_kg*1e15) if np.isfinite(reference_buoyant_mass_kg) else np.nan,
                     valid_event_count=int(np.count_nonzero(valid)),complete_pair_count=len(pair_rows),
                     excluded_event_count=len(events)-2*len(pair_rows),
                     self_calibration_warning=(method=='reference'),
                     assumptions='consecutive forward/reverse pairs; frequency depth proportional to buoyant mass')
    return enriched,calibration


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
    d = pd.read_csv(path, comment='#', encoding='utf-8-sig', low_memory=False)
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
    return dict(time_s=float(t[i]), peak_time_s=float(t[i]),
                event_start_s=float(t[i]), event_end_s=float(t[i]), duration_s=0.,
                selection_source='manual', width_s=0., contrast_hz=np.nan,
                depth_hz=float(base[i]-d.Frequency_Hz.iloc[i]),
                reference_ratio=np.nan, family='manual',
                pressure_delay_s=float(t[i]-prev[-1]) if len(prev) else np.nan,
                status='미검토', note='수동 추가')


def characterize_event(d, center, search_width, activity_floor_hz=0.):
    """Estimate a complete negative-excursion interval around a detector hit.

    ``time_s`` is the energy-weighted center of the selected excursion, while
    ``peak_time_s`` retains the deepest raw sample.  Short quiet gaps are
    bridged so a damped/oscillatory particle waveform is stored as one event.
    """
    t=d.Time_s.to_numpy(dtype=float); f=d.Frequency_Hz.to_numpy(dtype=float)
    dt=float(np.median(np.diff(t)))
    center=float(center); search_width=max(float(search_width),dt)
    baseline=local_baseline(d,center); residual=f-baseline
    smooth_s=min(.008,max(3*dt,search_width/10))
    n=max(3,round(smooth_s/dt)); n+=1-n%2
    smooth=np.convolve(residual,np.ones(n)/n,mode='same')
    half=max(.12,min(.60,1.5*search_width+.05))
    search=(t>=center-half)&(t<=center+half)
    ids=np.flatnonzero(search)
    if not len(ids):raise ValueError('후보 주변에 자동 구간을 계산할 표본이 없습니다.')
    peak=int(ids[np.argmin(smooth[ids])])
    noise_mask=(abs(t-center)>=max(.18,half))&(abs(t-center)<=.55)
    noise=robust_sigma(smooth[noise_mask]) if np.count_nonzero(noise_mask)>=20 else robust_sigma(smooth[ids])
    depth=max(0.,float(-smooth[peak]))
    threshold=max(float(activity_floor_hz),3*noise,min(.12,.20*depth))
    active=search&(smooth<=-threshold)

    # Bridge short gaps between oscillation lobes, but never outside search.
    active_ids=np.flatnonzero(active)
    max_gap=max(.018,min(.06,.35*search_width))
    for left,right in zip(active_ids[:-1],active_ids[1:]):
        if t[right]-t[left]<=max_gap:
            active[left:right+1]=True
    if not active[peak]:active[peak]=True
    left=peak
    while left>ids[0] and active[left-1]:left-=1
    right=peak
    while right<ids[-1] and active[right+1]:right+=1
    pad=max(dt,.004)
    lo=max(float(t[0]),float(t[left])-pad); hi=min(float(t[-1]),float(t[right])+pad)
    selected=(t>=lo)&(t<=hi)
    raw_peak=int(np.flatnonzero(selected)[np.argmin(residual[selected])])
    weights=np.maximum(-smooth[selected]-threshold,0.)
    selected_t=t[selected]
    representative=float(np.average(selected_t,weights=weights)) if np.sum(weights)>0 else float((lo+hi)/2)
    return dict(time_s=representative,peak_time_s=float(t[raw_peak]),
                event_start_s=lo,event_end_s=hi,duration_s=hi-lo,
                selection_source='auto')


def calibrate_detector(d, settings):
    """Freeze a local-depth threshold from a particle-free reference window.

    The returned rows are intentionally serializable so the same calibration
    can be reused by the live PLL detector without repeatedly estimating noise
    from data which may already contain a particle.
    """
    t, f = d.Time_s.to_numpy(), d.Frequency_Hz.to_numpy()
    dt = float(np.median(np.diff(t)))
    if np.max(abs(np.diff(t)-dt)) > .05*dt:
        raise ValueError('표본 간격이 5% 이상 불규칙하거나 누락 구간이 있습니다. 그래프 열람은 가능하지만 자동 분석은 지원하지 않습니다.')
    ref = (t >= settings.reference_start) & (t <= settings.reference_end)
    if sum(ref) < 30:
        raise ValueError('잡음 기준 구간에 최소 30표본이 필요합니다.')
    if settings.analyze_end <= settings.analyze_start:
        raise ValueError('분석 종료 시간은 시작 시간보다 커야 합니다.')
    widths=tuple(float(width) for width in settings.widths)
    if not widths:
        raise ValueError('short 탐색 폭을 하나 이상 입력하세요.')
    if not np.isfinite(widths).all() or any(width <= 0 or width > .20 for width in widths):
        raise ValueError('short 전용 탐색 폭은 0초 초과, 0.20초 이하로 입력하세요.')
    depth,valid=_local_depth_trace(t,f)
    ref_valid=ref&valid
    if np.count_nonzero(ref_valid)<30:
        raise ValueError('깊이 기준 구간이 기록 경계와 겹칩니다. 기준 구간을 넓히거나 옮기세요.')
    sigma=robust_sigma(depth[ref_valid])
    threshold=max(float(settings.floor_hz),float(settings.sigma_multiplier)*sigma)
    return [dict(requested_width_s=max(widths),width_s=max(widths),metric='local_depth',
                 sigma_hz=sigma,depth_sigma_hz=sigma,threshold_hz=threshold,
                 diff_sigma_hz=robust_sigma(np.diff(f[ref])))]


def _local_depth_trace(t,f,inner_s=.25,outer_s=.55):
    """Vectorized equivalent of ``local_baseline`` evaluated at each sample."""
    t=np.asarray(t,dtype=float); f=np.asarray(f,dtype=float)
    dt=float(np.median(np.diff(t)))
    radius=max(1,int(np.ceil(float(outer_s)/dt)))
    offsets=np.arange(-radius,radius+1,dtype=float)*dt
    side=(np.abs(offsets)>=float(inner_s))&(np.abs(offsets)<=float(outer_s))
    if np.count_nonzero(side)<6:
        raise ValueError('깊이 탐색에 필요한 양쪽 배경 표본이 부족합니다.')
    x=offsets[side]
    design=np.column_stack((np.ones(len(x)),x,x*x))
    weights=np.zeros(len(offsets),dtype=float)
    weights[side]=(np.linalg.pinv(design)[0])
    baseline=fftconvolve(f,weights,mode='same')
    depth=baseline-f
    valid=np.zeros(len(t),dtype=bool)
    if len(t)>2*radius:
        valid[radius:len(t)-radius]=True
    return depth,valid


def detect_calibrated(d, settings, thresholds):
    """Run the offline detector with previously frozen reference thresholds."""
    t, f = d.Time_s.to_numpy(), d.Frequency_Hz.to_numpy()
    dt = float(np.median(np.diff(t)))
    if np.max(abs(np.diff(t)-dt)) > .05*dt:
        raise ValueError('표본 간격이 5% 이상 불규칙하거나 누락 구간이 있습니다. 그래프 열람은 가능하지만 자동 분석은 지원하지 않습니다.')
    if settings.analyze_end <= settings.analyze_start:
        raise ValueError('분석 종료 시간은 시작 시간보다 커야 합니다.')
    if not thresholds:
        raise ValueError('깊이 문턱 정보가 없습니다. blank 기준 구간을 다시 학습하세요.')
    row=thresholds[0]
    sigma=float(row.get('depth_sigma_hz',row['sigma_hz']))
    threshold=float(row['threshold_hz'])
    diff_sigma=float(row.get('diff_sigma_hz',np.nan))
    width=float(row.get('requested_width_s',row.get('width_s',.20)))
    if width<=0 or width>.20:
        raise ValueError('short 전용 탐색 폭은 0초 초과, 0.20초 이하여야 합니다.')
    depth,valid=_local_depth_trace(t,f)
    valid&=(t>=settings.analyze_start)&(t<=settings.analyze_end)
    peak_depth=np.where(valid,depth,-np.inf)
    peaks,_=find_peaks(peak_depth,height=threshold,
                       distance=max(1,round(float(settings.merge_s)/dt)))
    hits=[dict(time_s=float(t[i]),width_s=width,contrast_hz=np.nan,
               reference_ratio=float(depth[i]/max(sigma,1e-12)),
               detection_threshold_hz=threshold,diff_sigma_hz=diff_sigma,
               search_depth_hz=float(depth[i]),family='short') for i in peaks]
    events=[]
    for h in hits:
        center = h['time_s']
        hit_threshold=float(h['detection_threshold_hz'])
        # Refine the local-depth peak with the exact timestamp-based
        # baseline used by manual review.
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
        # Require the exact local depth to pass the frozen depth threshold
        # and reject abrupt jumps far above blank behavior.
        if detail['depth_hz'] < hit_threshold:
            continue
        quality_half=max(.25,min(.60,h['width_s']+.15))
        quality=(t>=t[j]-quality_half)&(t<=t[j]+quality_half)
        local_jump=float(np.max(np.abs(np.diff(f[quality])))) if np.count_nonzero(quality)>1 else 0.
        jump_limit=max(.25,12*diff_sigma) if np.isfinite(diff_sigma) else .25
        if local_jump>jump_limit:
            continue
        try:
            interval=characterize_event(d,float(t[j]),h['width_s'],activity_floor_hz=min(hit_threshold,.08))
        except ValueError:
            interval=dict(time_s=float(t[j]),peak_time_s=float(t[j]),event_start_s=float(t[j]),
                          event_end_s=float(t[j]),duration_s=0.,selection_source='auto')
        events.append({**detail, **h, **interval,
                       'depth_hz':detail['depth_hz'], 'local_max_jump_hz':local_jump,
                       'quality':'ok','note':''})
    return sorted(events,key=lambda h:h['time_s'])


def detect(d, settings):
    thresholds = calibrate_detector(d, settings)
    return detect_calibrated(d, settings, thresholds), thresholds


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
