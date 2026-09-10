"""Standalone figures for a snapshot of retained review rows."""
from pathlib import Path
from datetime import datetime
import json
import textwrap
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
from particle_review_core import local_baseline, pressure_changes, batch_context_noise
from particle_review_core import sample_window_stats


def export_event_figures(data, events, parent, source, span, residual=False,
                         settings=None, progress=None, cancelled=None, review_context=None):
    folder=Path(parent)/('particle_graphs_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    t=data.Time_s.to_numpy(); f=data.Frequency_Hz.to_numpy()
    changes=pressure_changes(data).Time_s.to_numpy()
    fontpath=Path('C:/Windows/Fonts/malgun.ttf')
    font=FontProperties(fname=str(fontpath)) if fontpath.exists() else FontProperties()
    completed=[]; error=None
    context_rows=batch_context_noise(data,events,window_s=max(.1,min(1.0,span/2)),guard_s=min(.1,span/10),detrend_degree=1)
    noise_settings=(review_context or {}).get('noise_settings')
    analysis=None
    if noise_settings:
        from particle_review_noise import analyze_noise
        analysis=analyze_noise(data,events,**noise_settings)
        context_rows=analysis['contexts']
    try:
        for index,event in enumerate(events,1):
            if cancelled and cancelled():break
            center=float(event['time_s']); lo=max(t[0],center-span/2); hi=min(t[-1],center+span/2)
            mask=(t>=lo)&(t<=hi)
            if sum(mask)<2:raise ValueError(f'{center:.4f}초 주변에 그래프 표본이 부족합니다.')
            values=f; applied=False
            if residual:
                values=f-local_baseline(data,center); applied=True
            fig=Figure(figsize=(12,8),layout='constrained'); FigureCanvasAgg(fig)
            axes=fig.subplots(4,1,sharex=True)
            axes[0].plot(t[mask],values[mask],'.-',color='#175aaa',lw=.8,ms=3)
            axes[0].set_ylabel('Residual (Hz)' if applied else 'Frequency (Hz)')
            for ax,col,scale,color,label in [
                (axes[1],'Phase_deg',1,'#8b45a6','Phase (deg)'),
                (axes[2],'Amplitude_R_V',1000,'#df7d17','Amplitude (mV)'),
            ]:
                if col in data:ax.plot(t[mask],data[col].to_numpy()[mask]*scale,color=color,lw=.8)
                else:ax.text(.5,.5,'Not recorded',transform=ax.transAxes,ha='center')
                ax.set_ylabel(label)
            for col,color,label in [('PressureCommand_A_V','#239648','A'),('PressureCommand_B_V','#cf3d45','B')]:
                if col in data:axes[3].step(t[mask],data[col].to_numpy()[mask],where='post',color=color,label=label,lw=1)
            if axes[3].lines:axes[3].legend(loc='upper right')
            else:axes[3].text(.5,.5,'Not recorded',transform=axes[3].transAxes,ha='center')
            axes[3].set_ylabel('Command (V)'); axes[3].set_xlabel('Time (s)')
            for ax in axes:
                ax.grid(alpha=.2); ax.set_xlim(lo,hi)
                ax.ticklabel_format(axis='both',style='plain',useOffset=False)
                ax.axvline(center,color='#c85050',lw=.7,ls=':')
                for change in changes[(changes>=lo)&(changes<=hi)]:ax.axvline(change,color='gray',ls='--',lw=.7)
            sample_stats=sample_window_stats(t,lo,hi)
            heading=f'{Path(source).name}\n#{index:03d} | {center:.4f} s | {event.get("family", "")} | 검토: {event.get("status", "")} | 확대 폭 {span:g} s | 화면 표본 {sample_stats["sample_count"]:,}개'
            note=str(event.get('note',''))
            if note:heading+='\n'+'\n'.join(textwrap.wrap('메모: '+note,width=90))
            fig.suptitle(heading,fontproperties=font,fontsize=10)
            filename=f'{index:03d}_{center:012.4f}s.png'
            fig.savefig(folder/filename,dpi=160); fig.clear()
            completed.append({**event,'graph_file':filename,'plot_start_s':lo,'plot_end_s':hi,'residual_applied':applied,
                              'plot_sample_count':sample_stats['sample_count'],'plot_sampling_rate_hz':sample_stats['sampling_rate_hz']})
            if progress:progress(index,len(events))
    except Exception as exc:
        error=str(exc)
    finally:
        completed_df=pd.DataFrame(completed)
        if len(completed_df) and len(context_rows):
            pivot=context_rows.pivot(index='event_index',columns='context',values=['raw_std_hz','detrended_std_hz','peak_to_peak_residual_hz','threshold_10sigma_hz'])
            pivot.columns=['_'.join(col) for col in pivot.columns]
            pivot.index=pivot.index-1
            completed_df=completed_df.join(pivot,how='left')
        completed_df.to_csv(folder/'exported_events.csv',index=False,encoding='utf-8-sig')
        context_rows.to_csv(folder/'context_noise.csv',index=False,encoding='utf-8-sig')
        # Persist all retained annotations as a normal reloadable review as well.
        pd.DataFrame(events).to_csv(folder/'retained_review.csv',index=False,encoding='utf-8-sig')
        metadata=dict(source=str(Path(source).resolve()),settings=settings,sample_count=len(data),
                      time_start=float(t[0]),time_end=float(t[-1]),thresholds=[],
                      metadata={},pressure_changes=pressure_changes(data).to_dict('records'),
                      span_s=span,residual_requested=residual,context_noise_window_s=max(.1,min(1.0,span/2)),context_noise_guard_s=min(.1,span/10),requested_count=len(events),
                      completed_count=len(completed),cancelled=bool(cancelled and cancelled()),error=error)
        if review_context:
            metadata['metadata']=review_context.get('metadata',{})
            metadata['thresholds']=review_context.get('thresholds',[])
        if analysis is not None:
            metadata['noise_settings']=noise_settings
            metadata['context_noise_window_s']=noise_settings['window_s']
            metadata['context_noise_guard_s']=noise_settings['guard_s']
        (folder/'retained_review.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    if error:raise RuntimeError(f'{error}\n완료된 {len(completed)}개와 검토 목록은 보존되었습니다: {folder}')
    if analysis is not None and not (cancelled and cancelled()):
        from particle_review_noise import save_noise_analysis
        save_noise_analysis(folder/'noise_snr',analysis,source,(review_context or {}).get('metadata',{}))
    return folder,len(completed)
