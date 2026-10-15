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
                mit_noise_sigma_hz=np.nan, snr_mit=np.nan, selection_source='auto',
                status=status, warning=warning)


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
    if direction=='down' and coef[0]<=0 or direction=='up' and coef[0]>=0:
        raise ValueError('국소 2차 곡률의 부호가 antinode 방향과 다릅니다.')
    if not float(tt[0]-origin)<vertex<float(tt[-1]-origin):
        raise ValueError('국소 2차 꼭짓점이 fitting 구간 밖에 있습니다.')
    fitted=np.polyval(coef,tt-origin)
    return origin+vertex,float(np.polyval(coef,vertex)),tt,fitted,float(np.sqrt(np.mean((yy-fitted)**2)))


def _baseline_mask(t, events, start, end, length, guard):
    """Exclude whole transits, including SG support, from both baseline sides."""
    mask=((t>=start-guard-length)&(t<start-guard))|((t>end+guard)&(t<=end+guard+length))
    for event in events:
        a=float(event.get('event_start_s',event['time_s'])); b=float(event.get('event_end_s',event['time_s']))
        mask&=~((t>=a-guard)&(t<=b+guard))
    return mask


def analyze_mit_signal(data, events, sg_order=4, sg_window=27,
                       prominence_sigma=2.0, baseline_multiplier=3.0,
                       local_fit_fraction=0.20, antinode_count=7,
                       outer_tolerance_percent=20.0, noise_guard_s=.10,
                       noise_min_duration_s=1.0, manual_antinode_times=None,
                       baseline_degree=3, baseline_guard_s=0.0, compute_noise=True,
                       exclusion_events=None):
    """Analyze reviewed transits following the paper's stated frequency pipeline.

    Paper reference: seventh mode, cubic baseline, 3x first spacing, quadratic
    extrema, 20% outer agreement and 1-Hz order-2 HP noise. Implementation
    extensions: mode selection, full-transit/SG-support exclusion, residual
    fitting, fit validation, and explicitly labelled single-event local noise.
    """
    if not events:
        raise ValueError('MIT 피크 분석에는 검토 후보가 필요합니다.')
    if sg_window<3 or sg_window%2==0 or sg_order<1 or sg_order>=sg_window:
        raise ValueError('SG window는 홀수이고 polynomial order보다 커야 합니다.')
    if prominence_sigma<=0 or baseline_multiplier<=0 or not 0<local_fit_fraction<=.5 or antinode_count<1:
        raise ValueError('MIT 피크 분석 설정값이 올바르지 않습니다.')
    if baseline_degree not in (1,2,3) or baseline_guard_s<0:
        raise ValueError('baseline 차수는 1–3, 추가 guard는 0 이상이어야 합니다.')
    manual_antinode_times=manual_antinode_times or {}
    excluded=events if exclusion_events is None else exclusion_events
    t=data.Time_s.to_numpy(dtype=float); raw=data.Frequency_Hz.to_numpy(dtype=float)
    if sg_window>len(t):
        raise ValueError('SG window가 데이터 길이보다 큽니다.')
    dt=np.diff(t); median_dt=float(np.median(dt)); fs=1/median_dt
    if np.max(np.abs(dt-median_dt))>.05*median_dt:
        raise ValueError('표본 간격 변동이 5%를 넘어 SG/MIT 비교를 지원하지 않습니다.')
    smooth=savgol_filter(raw,sg_window,sg_order,mode='interp')
    guard=baseline_guard_s+(sg_window-1)*median_dt/2
    rows=[]; traces=[]; antinodes=[]; fit_curves=[]
    for index,event in enumerate(events,1):
        start=float(event.get('event_start_s',event['time_s'])); end=float(event.get('event_end_s',event['time_s']))
        if not np.isfinite([start,end]).all() or end<=start:
            rows.append(_empty(index,event,'invalid_event','수동 후보는 시각만 저장되었습니다. 이벤트 시작/끝을 지정하고 「구간 적용 후 재분석」을 누르세요.'))
            preview=(t>=float(event['time_s'])-.5)&(t<=float(event['time_s'])+.5)
            traces.extend(dict(event_index=index,time_s=float(tt),raw_hz=float(rr),sg_hz=float(ss),
                               baseline_hz=np.nan,residual_hz=np.nan,baseline_used=False)
                          for tt,rr,ss in zip(t[preview],raw[preview],smooth[preview]))
            continue
        approximate_spacing=max((end-start)/max(1,antinode_count-1),4*median_dt)
        outside=_baseline_mask(t,excluded,start,end,baseline_multiplier*approximate_spacing,guard)
        inside=(t>=start)&(t<=end)
        if np.count_nonzero(outside&(t<start))<6 or np.count_nonzero(outside&(t>end))<6 or np.count_nonzero(inside)<antinode_count*3:
            rows.append(_empty(index,event,'insufficient_samples','3차 baseline 또는 이벤트 표본이 부족합니다.')); continue
        center=float(event['time_s']); x=t-center
        baseline=np.polyval(np.polyfit(x[outside],smooth[outside],baseline_degree),x)
        residual=smooth-baseline
        down_strength=float(-np.min(residual[inside])); up_strength=float(np.max(residual[inside]))
        direction='down' if down_strength>=up_strength else 'up'; sign=-1 if direction=='down' else 1
        baseline_residual=(smooth-baseline)[outside]
        sigma=max(robust_sigma(baseline_residual),np.finfo(float).eps*max(1.,float(np.max(np.abs(raw))))*100)
        ids=np.flatnonzero(inside); target=sign*residual[inside]
        distance=max(1,int(round(.40*approximate_spacing/median_dt)))
        requested=manual_antinode_times.get(index,manual_antinode_times.get(str(index)))
        selection_source='manual' if requested is not None else 'auto'
        if requested is None:
            peaks,properties=find_peaks(target,distance=distance,prominence=prominence_sigma*sigma,
                                        height=prominence_sigma*sigma)
            peak_ids=ids[peaks]
        else:
            requested=np.asarray(requested,dtype=float)
            if np.any(~np.isfinite(requested)) or np.any((requested<start)|(requested>end)):
                row=_empty(index,event,'manual_invalid','수동 antinode는 이벤트 시작과 종료 사이에 있어야 합니다.')
                row['selection_source']='manual'; rows.append(row); continue
            peak_ids=np.asarray(sorted({int(ids[np.argmin(abs(t[ids]-value))]) for value in requested}),dtype=int)
        if len(peak_ids)!=antinode_count:
            row=_empty(index,event,'antinode_count',f'{antinode_count}개 필요, {len(peak_ids)}개 검출 · prominence {prominence_sigma:g}σ')
            if selection_source=='manual':
                row['warning']=f'{antinode_count}개 필요, 수동 지정 {len(peak_ids)}개'
            row['antinode_count']=int(len(peak_ids)); row['direction']=direction
            row['selection_source']=selection_source; rows.append(row)
            for antinode_index,peak_id in enumerate(peak_ids,1):
                antinodes.append(dict(event_index=index,antinode_index=antinode_index,
                    time_s=float(t[peak_id]),frequency_hz=float(smooth[peak_id]),
                    baseline_hz=float(baseline[peak_id]),delta_hz=float(sign*residual[peak_id]),
                    locally_quadratic_fitted=False,selection_source=selection_source))
            trace_mask=(t>=start-guard-baseline_multiplier*approximate_spacing)&(t<=end+guard+baseline_multiplier*approximate_spacing)
            traces.extend(dict(event_index=index,time_s=float(tt),raw_hz=float(rr),sg_hz=float(ss),
                               baseline_hz=float(bb),residual_hz=float(sign*(ss-bb)),baseline_used=bool(used))
                          for tt,rr,ss,bb,used in zip(t[trace_mask],raw[trace_mask],smooth[trace_mask],baseline[trace_mask],outside[trace_mask]))
            continue
        first_spacing=(float(t[peak_ids[1]]-t[peak_ids[0]])
                       if len(peak_ids)>1 else float(approximate_spacing))
        median_spacing=(float(np.median(np.diff(t[peak_ids])))
                        if len(peak_ids)>1 else float(approximate_spacing))
        outside=_baseline_mask(t,excluded,start,end,baseline_multiplier*first_spacing,guard)
        if np.count_nonzero(outside&(t<start))<6 or np.count_nonzero(outside&(t>end))<6:
            rows.append(_empty(index,event,'baseline_failed','명시된 3× antinode 간격 baseline 표본이 부족합니다.')); continue
        coef=np.polyfit(x[outside],smooth[outside],baseline_degree); baseline=np.polyval(coef,x); residual=smooth-baseline
        radius=max(2,int(round(local_fit_fraction*median_spacing/median_dt)))
        fitted=[]
        try:
            for antinode_index,peak_id in enumerate(peak_ids,1):
                if antinode_index in (1,antinode_count):
                    tt,dd,fit_t,fit_y,rmse=_quadratic_extremum(t,sign*residual,peak_id,radius,'up')
                    base=float(np.polyval(coef,tt-center)); ff=base+sign*dd
                    fit_curves.extend(dict(event_index=index,antinode_index=antinode_index,time_s=float(ft),
                        delta_hz=float(fy),frequency_hz=float(np.polyval(coef,ft-center)+sign*fy))
                        for ft,fy in zip(fit_t,fit_y))
                else:
                    tt,ff=float(t[peak_id]),float(smooth[peak_id]);rmse=np.nan
                base=float(np.polyval(coef,tt-center)); delta=float(sign*(ff-base))
                fitted.append((tt,ff,base,delta))
                antinodes.append(dict(event_index=index,antinode_index=antinode_index,time_s=tt,
                                      frequency_hz=ff,baseline_hz=base,delta_hz=delta,
                                      locally_quadratic_fitted=antinode_index in (1,antinode_count),
                                      fit_start_s=float(fit_t[0]) if antinode_index in (1,antinode_count) else np.nan,
                                      fit_end_s=float(fit_t[-1]) if antinode_index in (1,antinode_count) else np.nan,
                                      fit_rmse_hz=rmse,
                                      selection_source=selection_source))
        except ValueError as exc:
            row=_empty(index,event,'quadratic_failed',str(exc))
            row.update(antinode_count=len(peak_ids),direction=direction,selection_source=selection_source)
            rows.append(row); continue
        first_delta=fitted[0][3]; last_delta=fitted[-1][3]; mean_delta=(first_delta+last_delta)/2
        relative=abs(first_delta-last_delta)/abs(mean_delta)*100 if mean_delta else np.inf
        status='ok' if relative<=outer_tolerance_percent and first_delta>0 and last_delta>0 else 'outer_mismatch'
        warning='' if status=='ok' else f'첫/마지막 antinode 차이 {relative:.1f}% > {outer_tolerance_percent:g}% 또는 부호 이상'
        if any(point[3]<=sigma for point in fitted):
            status='invalid_shape';warning+=' · baseline 대비 유의한 깊이가 없는 antinode: 정량 채택 보류'
        if sg_window/fs>median_spacing*.35:warning+=' · SG window가 antinode 간격의 35% 초과: window 민감도 확인'
        spacing_cv=float(np.std(np.diff(t[peak_ids]))/median_spacing) if len(peak_ids)>1 else 0.
        if spacing_cv>.3:warning+=' · antinode 간격 불균일: 모드 형상/압력 전환 확인'
        pressure_changed=False
        for col in ('PressureCommand_A_V','PressureCommand_B_V'):
            if col in data:
                command=data[col].to_numpy(dtype=float)
                changes=t[1:][np.isfinite(command[1:])&np.isfinite(command[:-1])&(np.diff(command)!=0)]
                pressure_changed|=bool(np.any((changes>=start-guard)&(changes<=end+guard)))
        if pressure_changed:
            status='pressure_transition';warning+=' · 이벤트 주변 압력 명령 변경: 정량 채택 보류'
        rows.append(dict(event_index=index,time_s=float(event['time_s']),antinode_count=antinode_count,
                         direction=direction,first_delta_hz=first_delta,last_delta_hz=last_delta,
                         mean_delta_hz=mean_delta,relative_difference_percent=relative,
                         mit_noise_sigma_hz=np.nan,snr_mit=np.nan,status=status,warning=warning.strip(' ·'),
                         baseline_rmse_hz=float(np.sqrt(np.mean(residual[outside]**2))),
                         detection_sigma_sg_hz=sigma,
                         antinode_spacing_cv=spacing_cv,sg_spacing_ratio=sg_window/fs/median_spacing,
                         baseline_guard_s=guard))
        rows[-1]['selection_source']=selection_source
        trace_mask=(t>=start-guard-baseline_multiplier*first_spacing)&(t<=end+guard+baseline_multiplier*first_spacing)
        traces.extend(dict(event_index=index,time_s=float(tt),raw_hz=float(rr),sg_hz=float(ss),
                           baseline_hz=float(bb),residual_hz=float(sign*(ss-bb)),baseline_used=bool(used))
                      for tt,rr,ss,bb,used in zip(t[trace_mask],raw[trace_mask],smooth[trace_mask],baseline[trace_mask],outside[trace_mask]))
    summary=pd.DataFrame(rows)
    summary['event_start_s']=[float(e.get('event_start_s',e['time_s'])) for e in events]
    summary['event_end_s']=[float(e.get('event_end_s',e['time_s'])) for e in events]
    noise=None
    if compute_noise and (summary.status!='invalid_event').any():
        noise=analyze_mit_noise(data,events,guard_s=max(noise_guard_s,guard),min_duration_s=noise_min_duration_s,
                                include_edges=len(events)==1,edge_duration_s=max(5.,noise_min_duration_s))
        sigma=float(noise['summary']['pooled_sigma_hz'])
        summary['mit_noise_sigma_hz']=sigma
        summary['snr_mit']=summary.mean_delta_hz/sigma if np.isfinite(sigma) and sigma>0 else np.nan
        noise_warnings='; '.join(sorted(set(str(v) for v in noise['segments'].warning if str(v))))
        summary['noise_warning']=noise_warnings if np.isfinite(sigma) else '유효 raw baseline noise 구간 없음'
    summary['noise_source']='local_before_after' if len(events)==1 else 'inter_peak'
    return dict(summary=summary,antinodes=pd.DataFrame(antinodes,columns=['event_index','antinode_index','time_s',
                'frequency_hz','baseline_hz','delta_hz','locally_quadratic_fitted','fit_start_s','fit_end_s','fit_rmse_hz','selection_source']),
                traces=pd.DataFrame(traces,columns=['event_index','time_s','raw_hz','sg_hz','baseline_hz','residual_hz','baseline_used']),noise=noise,
                fit_curves=pd.DataFrame(fit_curves),
                settings=dict(sg_order=int(sg_order),sg_window=int(sg_window),sampling_rate_hz=fs,
                    data_start_s=float(t[0]),data_end_s=float(t[-1]),
                    sg_window_ms=1000*sg_window/fs,prominence_sigma=float(prominence_sigma),
                    baseline_degree=int(baseline_degree),baseline_multiplier=float(baseline_multiplier),antinode_count=int(antinode_count),
                    baseline_guard_s=float(baseline_guard_s),effective_baseline_guard_s=guard,
                    baseline_method='exclude complete event and SG half-window; exclude neighboring events',
                    extremum_method='quadratic fit to signed baseline-subtracted SG; reject invalid vertex/curvature',
                    local_extremum_degree=2,local_fit_fraction=float(local_fit_fraction),
                    outer_tolerance_percent=float(outer_tolerance_percent),representative_delta='mean of first and last fitted antinodes',
                    manual_antinode_times={str(key):[float(value) for value in values]
                                           for key,values in manual_antinode_times.items()},
                    stated_parameters='paper uses seventh mode, cubic baseline, 3x interval, quadratic extrema, 20% agreement, 1 Hz order-2 HP noise',
                    paper_prominence_note='paper reports approximately 5-10 sigma; current prominence is user-selected on SG baseline residual sigma',
                    inferred_parameters='user SG/order/mode/prominence/local width; full-event exclusion plus SG support; signed residual fitting; outer mean; single-event local noise extension'))


def compare_mit_signal_settings(data, events, settings, manual_antinode_times=None, exclusion_events=None):
    """Report processing sensitivity, not a statistical confidence interval."""
    fs=1/float(np.median(np.diff(data.Time_s.to_numpy(dtype=float))))
    base_window=int(settings.get('sg_window',27)); rows=[]
    # Three windows and three baseline degrees; repeat only distinct settings.
    variants={(base_window,degree) for degree in (1,2,3)}
    for factor in (.5,1.5):
        n=2*int(np.floor(base_window*factor/2))+1
        minimum=int(settings.get('sg_order',4))+1
        if minimum%2==0:minimum+=1
        variants.add((max(minimum,n),settings.get('baseline_degree',3)))
    lo=min(float(e.get('event_start_s',e['time_s'])) for e in events)-6
    hi=max(float(e.get('event_end_s',e['time_s'])) for e in events)+6
    local=data[(data.Time_s>=lo)&(data.Time_s<=hi)].reset_index(drop=True)
    for window,degree in sorted(variants):
        options=dict(settings,sg_window=window,baseline_degree=degree,compute_noise=False)
        result=analyze_mit_signal(local,events,manual_antinode_times=manual_antinode_times,
                                  exclusion_events=exclusion_events,**options)
        for row in result['summary'].to_dict('records'):
            rows.append({**row,'sg_window_ms':1000*window/fs,'baseline_degree':degree})
    return pd.DataFrame(rows)


def save_mit_signal_analysis(folder, analysis, source, metadata=None):
    """Save peak, antinode, trace and provenance tables without changing raw data."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    analysis['summary'].to_csv(folder/'mit_peak_summary.csv',index=False,encoding='utf-8-sig')
    analysis['antinodes'].to_csv(folder/'mit_antinode_fits.csv',index=False,encoding='utf-8-sig')
    analysis['traces'].to_csv(folder/'mit_processed_traces.csv',index=False,encoding='utf-8-sig')
    analysis.get('fit_curves',pd.DataFrame()).to_csv(folder/'mit_quadratic_curves.csv',index=False,encoding='utf-8-sig')
    if 'sensitivity' in analysis:
        analysis['sensitivity'].to_csv(folder/'mit_processing_sensitivity.csv',index=False,encoding='utf-8-sig')
    noise=analysis.get('noise');noise_summary=noise['summary'] if noise else None
    if noise:
        noise['segments'].to_csv(folder/'mit_noise_segments.csv',index=False,encoding='utf-8-sig')
        noise['residuals'].to_csv(folder/'mit_noise_residuals.csv',index=False,encoding='utf-8-sig')
    payload=dict(source=str(source),metadata=metadata or {},settings=analysis['settings'],noise_summary=noise_summary)
    (folder/'mit_signal_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    s=analysis['settings'];out=analysis['summary']
    lines=[f"# MIT 방식 {s['antinode_count']}차 모드 피크 분석",'',f'원본: `{source}`','',
           f"SG order {s['sg_order']}, window {s['sg_window']} ({s['sg_window_ms']:.3f} ms).",
           f"선택 모드 기대값: {s['antinode_count']} antinode. baseline {s['baseline_degree']}차, 이벤트 전체 + SG 반폭 제외, 양쪽 길이 첫 간격의 3배. 외곽 baseline-subtracted SG에 2차 fitting, 20% 일치 기준.",
           f"Noise: {analysis['summary'].noise_source.iloc[0]}; 1 Hz order-2 high-pass of raw baseline. Single-event local noise is an extension, not inter-peak paper noise.",
           '설정 민감도 범위는 통계적 신뢰구간이 아닙니다. SG 표본 상관을 고려한 Δf 신뢰구간은 계산하지 않습니다.',
           f"현재 데이터 추정값: prominence {s['prominence_sigma']:g}σ, local fit 반폭 {s['local_fit_fraction']:.2f}× antinode 간격, 외곽 두 Δf의 평균.",'',
           '|후보|antinode|첫 Δf(Hz)|마지막 Δf(Hz)|대표 Δf(Hz)|차이(%)|MIT SNR|상태|',
           '|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in out.itertuples():
        def value(x):return f'{x:.5f}' if np.isfinite(x) else '—'
        lines.append(f'|{row.event_index}|{row.antinode_count}|{value(row.first_delta_hz)}|{value(row.last_delta_hz)}|{value(row.mean_delta_hz)}|{value(row.relative_difference_percent)}|{value(row.snr_mit)}|{row.status}|')
    (folder/'mit_signal_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return folder
