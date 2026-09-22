"""Offline visualization; leave acquisition data and GUI unchanged."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'outputs' / 'drift_removed_20260921'
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(root / 'raw_20260921_191157_7thmode_VT1.csv', comment='#',
                   usecols=['Time_s', 'Frequency_Hz'])
t, f = data.Time_s.to_numpy(), data.Frequency_Hz.to_numpy()
assert np.isfinite(t).all() and np.isfinite(f).all() and (np.diff(t) > 0).all()
fs = 1 / np.median(np.diff(t))
window = int(round(fs)) | 1
baseline = savgol_filter(f, window, 2, mode='interp')
residual = f - baseline
# Exclude the half-window endpoints from reported statistics.
valid = (t >= t[0] + .5) & (t <= t[-1] - .5)
recent = t >= t[-1] - 60
stats_mask = recent & valid
stats = {'baseline_method': 'centered Savitzky-Golay, quadratic, 1 second',
         'window_samples': window, 'nominal_fs_hz': fs,
         'statistics_interval_s': [float(t[stats_mask][0]), float(t[stats_mask][-1])],
         'raw_std_hz': float(np.std(f[stats_mask])),
         'residual_std_hz': float(np.std(residual[stats_mask]))}

plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 11,
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), gridspec_kw={'height_ratios': [1.15, 1, 1]})
fig.suptitle('느린 드리프트 제거 전후 · 7th mode VT1', fontsize=19, fontweight='bold', x=.08, ha='left')
fig.text(.08, .935, '기준선: 1초 Savitzky–Golay, 2차 다항식  |  잔차 = 원신호 - 기준선  |  마지막 60초', color='#526079')
ax[0].plot(t[recent], f[recent], color='#346fbd', lw=.35, label='원신호', rasterized=True)
ax[0].plot(t[recent], baseline[recent], color='#ee8b28', lw=2, label='느린 기준선 (1초)')
ax[0].set_ylabel('Frequency (Hz)')
ax[0].set_title('원신호와 느린 기준선', loc='left', fontweight='bold')
ax[0].ticklabel_format(axis='y', style='plain', useOffset=False)
ax[0].legend(loc='upper left', ncol=2)
ax[1].plot(t[recent], residual[recent], color='#267f88', lw=.35, rasterized=True)
ax[1].set_title(f'드리프트 제거 후: 잔차 σ = {stats["residual_std_hz"]:.3f} Hz   (같은 구간 원신호 σ = {stats["raw_std_hz"]:.3f} Hz)', loc='left', fontweight='bold')
ax[1].set_ylabel('Residual (Hz)')
for a in ax[:2]:
    a.set_xlim(t[-1]-60, t[-1])
    a.axvspan(t[-1]-.5, t[-1], color='#e8b779', alpha=.3)
zoom = (t >= 260) & (t <= 260.5)
ax[2].plot(t[zoom], residual[zoom], color='#267f88', lw=.9)
ax[2].set_title('국소 파형 확대: 260.0–260.5초', loc='left', fontweight='bold')
ax[2].set_xlim(260, 260.5)
ax[2].set_ylabel('Residual (Hz)')
for a in ax[1:]:
    a.axhline(0, color='#667085', lw=.7, ls='--')
    a.set_ylim(-.4, .4)
for a in ax:
    a.set_xlabel('Time (s)')
    a.grid(alpha=.2)
fig.text(.08, .025, '음영: 끝단 0.5초 (통계 제외). 시간축은 CSV의 재구성 시간 기준. 이 잔차는 1초 기준선에 대한 값이며 입자 신호 보존은 별도 확인이 필요합니다.', fontsize=9, color='#526079')
fig.tight_layout(rect=[.03, .05, .99, .92], h_pad=1.6)
fig.savefig(out / 'drift_removed.png', dpi=170)
(out / 'statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
print(json.dumps(stats, indent=2))
print(out / 'drift_removed.png')

