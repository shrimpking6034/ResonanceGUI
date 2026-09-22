"""Read-only source analysis of the reviewed 2026-09-21 BW80 particle run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Particle_data/processed/20260921/particle_graphs_20260921_FPGA_SR560_5thmode_BW80'
review = json.loads((SOURCE / 'retained_review.json').read_text(encoding='utf-8'))
raw = pd.read_csv(review['source'], comment='#', low_memory=False)
events = pd.read_csv(SOURCE / 'retained_review.csv')
commands = raw[['PressureCommand_A_V', 'PressureCommand_B_V']]
changes = raw.loc[commands.ne(commands.shift()).any(axis=1)].reset_index(drop=True)
rows = []
for index, event in events.iterrows():
    k = np.searchsorted(changes.Time_s, event.event_start_s, side='right') - 1
    command = changes.iloc[k]
    stop = changes.iloc[k + 1]
    assert command.PressureCommand_A_V != command.PressureCommand_B_V
    assert stop.PressureCommand_A_V == stop.PressureCommand_B_V == 0
    assert command.Time_s <= event.event_start_s < event.event_end_s < stop.Time_s
    rows.append(dict(event=index + 1, time=event.time_s,
        direction='R to L' if command.PressureCommand_A_V > command.PressureCommand_B_V else 'L to R',
        arrival=event.event_start_s-command.Time_s,
        width=event.duration_s, post=stop.Time_s-event.event_end_s,
        drive=stop.Time_s-command.Time_s))
results = pd.DataFrame(rows)
assert len(results) == 29
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout='constrained')
colors = {'L to R': '#2878b5', 'R to L': '#d55e00'}
for direction, group in results.groupby('direction'):
    axes[0].plot(group.time, group.arrival, 'o-', ms=5, color=colors[direction], label=direction)
axes[0].set(xlabel='Recording time (s)', ylabel='Command ON to event start (s)', title='Arrival delay by commanded direction')
axes[0].legend()
for j in range(1, len(results)):
    current, previous = results.iloc[j], results.iloc[j-1]
    if current.direction != 'R to L' or previous.direction != 'L to R':
        continue
    axes[1].scatter(previous.post, current.arrival, c=colors['R to L'], s=35)
    if current.event in (22, 24, 26, 28):
        axes[1].annotate(f"#{current.event}", (previous.post,current.arrival), xytext=(5,5), textcoords='offset points')
axes[1].set(xlabel='Previous L-to-R: event end to command OFF (s)',
    ylabel='Next R-to-L arrival delay (s)', title='Long forward overrun precedes late return')
for ax in axes:
    ax.grid(alpha=.22)
fig.suptitle('29 reviewed events | 0.2 V commands | measured pressure unavailable', fontsize=12)
fig.savefig(ROOT / 'diagnostics/pingpong_20260921_timing.png', dpi=160)
print(results.groupby('direction')[['arrival','width','post','drive']].median().round(4).to_string())
print('First 20 total command ON seconds:', results.iloc[:20].groupby('direction').drive.sum().to_dict())
print('Measured pressure nonmissing:', raw.filter(regex='^Measured.*Pressure|^MeasuredPressure').notna().sum().to_dict())
