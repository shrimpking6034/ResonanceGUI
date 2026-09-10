"""Shared frequency-noise/SNR calculation for the review GUI and CLI."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from particle_review_core import batch_context_noise


def analyze_noise(data, events, window_s=1.0, guard_s=.25, degree=1):
    if not events:
        raise ValueError('분석할 후보가 없습니다.')
    contexts = batch_context_noise(data, events, window_s, guard_s, degree)
    results, samples = [], []
    for i, event in enumerate(events, 1):
        rows = contexts[contexts.event_index == i].set_index('context')
        b, a = rows.loc['before'], rows.loc['after']
        warnings = []
        for label, ctx in [('before', b), ('after', a)]:
            if ctx['status'] != 'ok':
                warnings.append(label + ': 표본 부족')
            if ctx.end_s-ctx.start_s < window_s-1e-8:
                warnings.append(label + ': 기록 경계에서 구간 잘림')
            part = data[(data.Time_s>=ctx.start_s)&(data.Time_s<=ctx.end_s)]
            t = part.Time_s.to_numpy(); f = part.Frequency_Hz.to_numpy()
            if len(t)>1:
                dt = np.diff(t)
                if np.any(dt<=0) or np.max(dt)>1.5*np.median(dt):
                    warnings.append(label + ': 표본 간격 불규칙/누락')
            if ctx['status'] != 'ok':
                continue
            drift = np.polyval(np.polyfit(t-float(event['time_s']), f, degree), t-float(event['time_s']))
            samples.extend(dict(event_index=i, context=label, time_s=tt,
                                frequency_hz=ff, drift_hz=dd, residual_hz=ff-dd)
                           for tt, ff, dd in zip(t, f, drift))
        pooled = np.sqrt((b.sample_count*b.detrended_std_hz**2+a.sample_count*a.detrended_std_hz**2)/(b.sample_count+a.sample_count)) if b.sample_count+a.sample_count else np.nan
        depth = abs(float(event.get('depth_hz', np.nan)))
        if not np.isfinite(pooled) or pooled<=0 or not np.isfinite(depth):
            warnings.append('SNR 계산 불가')
        if not warnings and max(b.detrended_std_hz,a.detrended_std_hz)>2*min(b.detrended_std_hz,a.detrended_std_hz):
            warnings.append('전후 σ 차이 >2배: guard/과도응답 확인')
        def ratio(sigma):
            return depth/sigma if np.isfinite(sigma) and sigma>0 and np.isfinite(depth) else np.nan
        snr = ratio(pooled)
        results.append(dict(event_index=i, time_s=float(event['time_s']), depth_hz=depth,
            review_status=event.get('status',''), note=event.get('note',''),
            sigma_before_hz=b.detrended_std_hz, sigma_after_hz=a.detrended_std_hz,
            sigma_pooled_hz=pooled, threshold_10sigma_hz=10*pooled,
            snr_before=ratio(b.detrended_std_hz), snr_after=ratio(a.detrended_std_hz),
            snr_pooled=snr, snr_previous_mean_sigma=ratio((b.detrended_std_hz+a.detrended_std_hz)/2),
            meets_10sigma=bool(snr>=10) if np.isfinite(snr) else None,
            warning='; '.join(warnings)))
    return dict(summary=pd.DataFrame(results), contexts=contexts,
                residuals=pd.DataFrame(samples,columns=['event_index','context','time_s','frequency_hz','drift_hz','residual_hz']),
                settings=dict(window_s=window_s,guard_s=guard_s,degree=degree,std_ddof=0,
                    signal='abs(saved depth_hz)', pooled_definition='sqrt(sum(n*sigma^2)/sum(n))',
                    detrending='separate fit per side; no smoothing or interpolation'))


def save_noise_analysis(folder, analysis, source, metadata=None):
    """Save a computed snapshot. Caller provides a new output folder."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    out, contexts, residuals = (analysis[k] for k in ['summary','contexts','residuals'])
    settings = analysis['settings']; guard = settings['guard_s']
    out.to_csv(folder/'snr_summary.csv',index=False,encoding='utf-8-sig')
    contexts.to_csv(folder/'snr_context_noise.csv',index=False,encoding='utf-8-sig')
    residuals.to_csv(folder/'snr_residuals.csv',index=False,encoding='utf-8-sig')
    (folder/'snr_analysis_settings.json').write_text(json.dumps(dict(source=str(source),metadata=metadata or {},**settings),ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 주변 노이즈 및 SNR','',f'원본: `{source}`','',
        f"전후 각각 {settings['window_s']:g}초, 중심 ±{guard:g}초 제외, 각 구간 {settings['degree']}차 drift 제거. σ는 residual 표준편차(ddof=0).",
        '통합 σ = sqrt(sum(n × σ²) / sum(n)); SNR = abs(depth_hz) / 통합 σ.', '',
        '|시각(s)|Δf(Hz)|전 σ(Hz)|후 σ(Hz)|통합 σ(Hz)|SNR|10σ|안내|',
        '|---:|---:|---:|---:|---:|---:|:---:|---|']
    for row in out.itertuples():
        verdict='계산 불가' if not np.isfinite(row.snr_pooled) else ('충족' if row.meets_10sigma else '미달')
        lines.append(f'|{row.time_s:.4f}|{row.depth_hz:.4f}|{row.sigma_before_hz:.4f}|{row.sigma_after_hz:.4f}|{row.sigma_pooled_hz:.4f}|{row.snr_pooled:.2f}|{verdict}|{row.warning}|')
    lines += ['',f'SNR 평균 {out.snr_pooled.mean():.2f}, 중앙값 {out.snr_pooled.median():.2f}.',
        '', '선택 후보의 국소 peak/RMS SNR이다. 제외 구간 밖에 이벤트가 남으면 noise에 포함된다. 그래프에서 guard를 확인해야 하며, 입자 기원이나 전체 검출률을 이 수치만으로 확정하지 않는다.']
    (folder/'snr_report.md').write_text('\n'.join(lines),encoding='utf-8')
    # Paginate to keep export memory bounded for large populations.
    for page, start in enumerate(range(0,len(out),6),1):
        group=out.iloc[start:start+6]
        fig=Figure(figsize=(12,2.7*len(group)),layout='constrained'); FigureCanvasAgg(fig)
        axes=fig.subplots(len(group),2,squeeze=False)
        for row, pair in zip(group.itertuples(),axes):
            for label,color in [('before','tab:blue'),('after','tab:orange')]:
                p=residuals[(residuals.event_index==row.event_index)&(residuals.context==label)]
                x=p.time_s-row.time_s
                pair[0].plot(x,p.frequency_hz,'.-',color=color,ms=2,lw=.7)
                pair[0].plot(x,p.drift_hz,'--',color='black',lw=.8)
                pair[1].plot(x,p.residual_hz,'.-',color=color,ms=2,lw=.7,label=label)
            for ax in pair:
                ax.axvspan(-guard,guard,color='grey',alpha=.15); ax.grid(alpha=.2)
                ax.set_xlabel('Time from event (s)')
            pair[0].set(title=f'#{row.event_index}: {row.time_s:.4f} s',ylabel='Frequency (Hz)')
            pair[1].set(title=f'SNR={row.snr_pooled:.2f} | sigma={row.sigma_pooled_hz:.4f} Hz',ylabel='Residual (Hz)')
            pair[1].legend(fontsize=8)
        name='snr_context_overview.png' if page==1 else f'snr_context_overview_{page:03d}.png'
        fig.savefig(folder/name,dpi=150); fig.clear()
    return folder
