"""MIT-paper-style seventh-mode peak processing with explicit inferred defaults."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from particle_review_core import robust_sigma
from particle_review_mit_noise import analyze_mit_noise


def _empty(index, event, status, warning):
    return dict(event_index=index, time_s=float(event['time_s']), antinode_count=0,
                direction='', first_delta_hz=np.nan, last_delta_hz=np.nan,
                mean_delta_hz=np.nan, relative_difference_percent=np.nan,
                mit_noise_sigma_hz=np.nan, snr_mit=np.nan, status=status, warning=warning)


def _quadratic_extremum(t, y, center_index, radius, direction):
    left=max(0,center_index-radius); right=min(len(t),center_index+radius+1)
    tt=t[left:right]; yy=y[left:right]
    if len(tt)<5:
        raise ValueError('국소 2차 fitting 표본 부족')
    origin=float(t[center_index]); coef=np.polyfit(tt-origin,yy,2)
    if abs(coef[0])<1e-15:
        vertex=0.
    else:
        vertex=float(-coef[1]/(2*coef[0]))
    vertex=float(np.clip(vertex,float(tt[0]-origin),float(tt[-1]-origin)))
    # A minimum must curve upward and a maximum downward.
    if direction=='down' and coef[0]<=0 or direction=='up' and coef[0]>=0:
        vertex=0.
    return origin+vertex,float(np.polyval(coef,vertex))


def analyze_mit_signal(data, events, sg_order=4, sg_window=27,
                       prominence_sigma=2.0, baseline_multiplier=3.0,
                       local_fit_fraction=0.20, antinode_count=7,
                       outer_tolerance_percent=20.0, noise_guard_s=.10,
                       noise_min_duration_s=1.0):
    """Analyze reviewed transits following the paper's stated frequency pipeline.

    Stated values: seven antinodes, cubic immediate baseline, baseline length
    three times the first-to-second antinode interval, local quadratic extrema,
    outer-antinode agreement <=20%, and 1-Hz order-2 HP baseline noise.
    Inferred defaults for the supplied recording: SG 4/27, prominence 2 sigma,
    local-fit half width 20%
    of the median antinode interval, and the mean of the two outer deltas.
    """
    if not events:
        raise ValueError('MIT 피크 분석에는 검토 후보가 필요합니다.')
    if sg_window<3 or sg_window%2==0 or sg_order<1 or sg_order>=sg_window:
        raise ValueError('SG window는 홀수이고 polynomial order보다 커야 합니다.')
    if prominence_sigma<=0 or baseline_multiplier<=0 or not 0<local_fit_fraction<=.5:
        raise ValueError('MIT 피크 분석 설정값이 올바르지 않습니다.')
    t=data.Time_s.to_numpy(dtype=float); raw=data.Frequency_Hz.to_numpy(dtype=float)
    if sg_window>len(t):
        raise ValueError('SG window가 데이터 길이보다 큽니다.')
    dt=np.diff(t); median_dt=float(np.median(dt)); fs=1/median_dt
    if np.max(np.abs(dt-median_dt))>.05*median_dt:
        raise ValueError('표본 간격 변동이 5%를 넘어 SG/MIT 비교를 지원하지 않습니다.')
    smooth=savgol_filter(raw,sg_window,sg_order,mode='interp')
    rows=[]; traces=[]; antinodes=[]
    for index,event in enumerate(events,1):
        start=float(event.get('event_start_s',event['time_s'])); end=float(event.get('event_end_s',event['time_s']))
        if not np.isfinite([start,end]).all() or end<=start:
            rows.append(_empty(index,event,'invalid_event','유효한 이벤트 시작/끝이 필요합니다.')); continue
        approximate_spacing=max((end-start)/max(1,antinode_count-1),4*median_dt)
        pre=(t>=start-baseline_multiplier*approximate_spacing)&(t<start)
        post=(t>end)&(t<=end+baseline_multiplier*approximate_spacing)
        outside=pre|post
        inside=(t>=start)&(t<=end)
        if np.count_nonzero(outside)<12 or np.count_nonzero(inside)<antinode_count*3:
            rows.append(_empty(index,event,'insufficient_samples','3차 baseline 또는 이벤트 표본이 부족합니다.')); continue
        center=float(event['time_s']); x=t-center
        baseline=np.polyval(np.polyfit(x[outside],smooth[outside],3),x)
        residual=smooth-baseline
        down_strength=float(-np.min(residual[inside])); up_strength=float(np.max(residual[inside]))
        direction='down' if down_strength>=up_strength else 'up'; sign=-1 if direction=='down' else 1
        baseline_residual=(smooth-baseline)[outside]
        sigma=max(robust_sigma(baseline_residual),np.finfo(float).eps)
        ids=np.flatnonzero(inside); target=sign*residual[inside]
        distance=max(1,int(round(.40*approximate_spacing/median_dt)))
        peaks,properties=find_peaks(target,distance=distance,prominence=prominence_sigma*sigma)
        if len(peaks)>antinode_count:
            keep=np.argsort(properties['prominences'])[-antinode_count:]
            peaks=np.sort(peaks[keep])
        if len(peaks)!=antinode_count:
            row=_empty(index,event,'antinode_count',f'{antinode_count}개 필요, {len(peaks)}개 검출 · prominence {prominence_sigma:g}σ')
            row['antinode_count']=int(len(peaks)); row['direction']=direction; rows.append(row); continue
        peak_ids=ids[peaks]
        first_spacing=float(t[peak_ids[1]]-t[peak_ids[0]])
        median_spacing=float(np.median(np.diff(t[peak_ids])))
        # Refit exactly as reported: immediate baselines on both sides, each
        # three times the first-to-second antinode interval.
        pre=(t>=t[peak_ids[0]]-baseline_multiplier*first_spacing)&(t<t[peak_ids[0]])
        post=(t>t[peak_ids[-1]])&(t<=t[peak_ids[-1]]+baseline_multiplier*first_spacing)
        outside=pre|post
        if np.count_nonzero(outside)<12:
            rows.append(_empty(index,event,'baseline_failed','명시된 3× antinode 간격 baseline 표본이 부족합니다.')); continue
        coef=np.polyfit(x[outside],smooth[outside],3); baseline=np.polyval(coef,x); residual=smooth-baseline
        radius=max(2,int(round(local_fit_fraction*median_spacing/median_dt)))
        fitted=[]
        try:
            for antinode_index,peak_id in enumerate(peak_ids,1):
                if antinode_index in (1,antinode_count):
                    tt,ff=_quadratic_extremum(t,smooth,peak_id,radius,direction)
                else:
                    tt,ff=float(t[peak_id]),float(smooth[peak_id])
                base=float(np.polyval(coef,tt-center)); delta=float(sign*(ff-base))
                fitted.append((tt,ff,base,delta))
                antinodes.append(dict(event_index=index,antinode_index=antinode_index,time_s=tt,
                                      frequency_hz=ff,baseline_hz=base,delta_hz=delta,
                                      locally_quadratic_fitted=antinode_index in (1,antinode_count)))
        except ValueError as exc:
            rows.append(_empty(index,event,'quadratic_failed',str(exc))); continue
        first_delta=fitted[0][3]; last_delta=fitted[-1][3]; mean_delta=(first_delta+last_delta)/2
        relative=abs(first_delta-last_delta)/abs(mean_delta)*100 if mean_delta else np.inf
        status='ok' if relative<=outer_tolerance_percent and first_delta>0 and last_delta>0 else 'outer_mismatch'
        warning='' if status=='ok' else f'첫/마지막 antinode 차이 {relative:.1f}% > {outer_tolerance_percent:g}% 또는 부호 이상'
        rows.append(dict(event_index=index,time_s=float(event['time_s']),antinode_count=antinode_count,
                         direction=direction,first_delta_hz=first_delta,last_delta_hz=last_delta,
                         mean_delta_hz=mean_delta,relative_difference_percent=relative,
                         mit_noise_sigma_hz=np.nan,snr_mit=np.nan,status=status,warning=warning))
        trace_mask=(t>=t[peak_ids[0]]-baseline_multiplier*first_spacing)&(t<=t[peak_ids[-1]]+baseline_multiplier*first_spacing)
        traces.extend(dict(event_index=index,time_s=float(tt),raw_hz=float(rr),sg_hz=float(ss),
                           baseline_hz=float(bb),residual_hz=float(sign*(ss-bb)))
                      for tt,rr,ss,bb in zip(t[trace_mask],raw[trace_mask],smooth[trace_mask],baseline[trace_mask]))
    summary=pd.DataFrame(rows)
    noise=None
    if len(events)>=2:
        noise=analyze_mit_noise(data,events,guard_s=noise_guard_s,min_duration_s=noise_min_duration_s)
        sigma=float(noise['summary']['pooled_sigma_hz'])
        summary['mit_noise_sigma_hz']=sigma
        summary['snr_mit']=summary.mean_delta_hz/sigma if np.isfinite(sigma) and sigma>0 else np.nan
    return dict(summary=summary,antinodes=pd.DataFrame(antinodes),traces=pd.DataFrame(traces),noise=noise,
                settings=dict(sg_order=int(sg_order),sg_window=int(sg_window),sampling_rate_hz=fs,
                    sg_window_ms=1000*sg_window/fs,prominence_sigma=float(prominence_sigma),
                    baseline_degree=3,baseline_multiplier=float(baseline_multiplier),antinode_count=int(antinode_count),
                    local_extremum_degree=2,local_fit_fraction=float(local_fit_fraction),
                    outer_tolerance_percent=float(outer_tolerance_percent),representative_delta='mean of first and last fitted antinodes',
                    stated_parameters='7 antinodes; cubic baseline; 3x interval; quadratic extrema; 20% agreement; 1 Hz order-2 HP noise',
                    paper_prominence_note='paper reports approximately 5-10 sigma; supplied recording needs 2 sigma to retain all reviewed seven-antinode groups',
                    inferred_parameters='SG 4/27 data-derived default; 2 sigma prominence; 0.20 interval local fit; outer mean'))


def save_mit_signal_analysis(folder, analysis, source, metadata=None):
    """Save peak, antinode, trace and provenance tables without changing raw data."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    analysis['summary'].to_csv(folder/'mit_peak_summary.csv',index=False,encoding='utf-8-sig')
    analysis['antinodes'].to_csv(folder/'mit_antinode_fits.csv',index=False,encoding='utf-8-sig')
    analysis['traces'].to_csv(folder/'mit_processed_traces.csv',index=False,encoding='utf-8-sig')
    noise=analysis.get('noise');noise_summary=noise['summary'] if noise else None
    if noise:
        noise['segments'].to_csv(folder/'mit_noise_segments.csv',index=False,encoding='utf-8-sig')
        noise['residuals'].to_csv(folder/'mit_noise_residuals.csv',index=False,encoding='utf-8-sig')
    payload=dict(source=str(source),metadata=metadata or {},settings=analysis['settings'],noise_summary=noise_summary)
    (folder/'mit_signal_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    s=analysis['settings'];out=analysis['summary']
    lines=['# MIT 방식 7차 모드 피크 분석','',f'원본: `{source}`','',
           f"SG order {s['sg_order']}, window {s['sg_window']} ({s['sg_window_ms']:.3f} ms).",
           '논문 명시값: 7 antinode, 3차 immediate baseline, 첫 간격의 3배 baseline 길이, 외곽 antinode 2차 fitting, 20% 일치 기준, raw inter-peak 1 Hz 2차 high-pass noise.',
           f"현재 데이터 추정값: prominence {s['prominence_sigma']:g}σ, local fit 반폭 {s['local_fit_fraction']:.2f}× antinode 간격, 외곽 두 Δf의 평균.",'',
           '|후보|antinode|첫 Δf(Hz)|마지막 Δf(Hz)|대표 Δf(Hz)|차이(%)|MIT SNR|상태|',
           '|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in out.itertuples():
        def value(x):return f'{x:.5f}' if np.isfinite(x) else '—'
        lines.append(f'|{row.event_index}|{row.antinode_count}|{value(row.first_delta_hz)}|{value(row.last_delta_hz)}|{value(row.mean_delta_hz)}|{value(row.relative_difference_percent)}|{value(row.snr_mit)}|{row.status}|')
    (folder/'mit_signal_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return folder
