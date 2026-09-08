"""Add before/after frequency-noise metrics to an existing retained-review folder."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from particle_review_core import load_raw, batch_context_noise, local_baseline


def main(folder, window_s=1.0, guard_s=.10, degree=1):
    folder=Path(folder).resolve()
    info=json.loads((folder/'retained_review.json').read_text(encoding='utf-8'))
    source=Path(info['source']); data, metadata=load_raw(source)
    review=pd.read_csv(folder/'retained_review.csv',encoding='utf-8-sig')
    events=review.to_dict('records')
    rows=batch_context_noise(data,events,window_s,guard_s,degree)
    rows.to_csv(folder/'context_noise.csv',index=False,encoding='utf-8-sig')
    summary=rows.groupby('context')[['mean_frequency_hz','raw_std_hz','detrended_std_hz','peak_to_peak_residual_hz','threshold_10sigma_hz']].agg(['mean','median','min','max']).reset_index()
    summary.columns=['context']+[f'{a}_{b}' for a,b in summary.columns.tolist()[1:]]
    payload=dict(source=str(source),review_folder=str(folder),event_count=len(events),window_s=window_s,guard_s=guard_s,detrend_method=f'polynomial_degree_{degree}',definition='sigma_f = std(f - drift) in equal before/after windows; raw samples, no smoothing/interpolation',metadata=metadata,summary=summary.to_dict('records'),context_noise_csv='context_noise.csv')
    (folder/'context_noise.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    # Overview: all event windows with raw signal, fitted drift, and residual.
    t=data.Time_s.to_numpy(); f=data.Frequency_Hz.to_numpy(); n=len(events)
    fig,axes=plt.subplots(n,2,figsize=(13,max(3,2.2*n)),squeeze=False,layout='constrained')
    for i,event in enumerate(events):
        c=float(event['time_s']); x=t-c; before=(x>=-guard_s-window_s)&(x<=-guard_s); after=(x>=guard_s)&(x<=guard_s+window_s); mask=before|after
        coeff=np.polyfit(x[mask],f[mask],degree); drift=np.polyval(coeff,x[mask]); residual=f[mask]-drift
        ax=axes[i,0]; ax.plot(x[mask],f[mask],'.',ms=2,color='#175aaa',label='raw f(t)'); ax.plot(x[mask],drift,'-',color='#d84242',lw=1,label='drift fit'); ax.axvspan(-guard_s,guard_s,color='#999999',alpha=.15); ax.axvline(0,color='k',lw=.7); ax.legend(fontsize=7,loc='best'); ax.set_title(f'#{i+1} {c:.4f} s'); ax.set_ylabel('Frequency (Hz)'); ax.grid(alpha=.2)
        ax=axes[i,1]; ax.plot(x[mask],residual,'.',ms=2,color='#8b45a6'); ax.axvspan(-guard_s,guard_s,color='#999999',alpha=.15); ax.axvline(0,color='k',lw=.7); ax.set_title(f'noise residual | σ={np.std(residual):.4f} Hz'); ax.set_ylabel('Residual (Hz)'); ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Time from event (s)'); axes[-1,1].set_xlabel('Time from event (s)')
    fig.suptitle(f'Before/after frequency noise | window={window_s:g} s, guard={guard_s:g} s, degree={degree}')
    fig.savefig(folder/'context_noise_overview.png',dpi=160); plt.close(fig)
    print(rows[['event_index','event_time_s','context','sample_count','raw_std_hz','detrended_std_hz','peak_to_peak_residual_hz','threshold_10sigma_hz']].to_string(index=False))
    print('\nSaved:',folder/'context_noise.csv',folder/'context_noise.json',folder/'context_noise_overview.png',sep='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('folder'); p.add_argument('--window',type=float,default=1.0); p.add_argument('--guard',type=float,default=.10); p.add_argument('--degree',type=int,choices=[1,2],default=1); a=p.parse_args(); main(a.folder,a.window,a.guard,a.degree)
