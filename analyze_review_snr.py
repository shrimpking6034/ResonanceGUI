"""Optional CLI entry point; the review GUI uses the same analysis functions."""
import json
import sys
from pathlib import Path
import pandas as pd
from particle_review_core import load_raw
from particle_review_noise import analyze_noise, save_noise_analysis


def main(folder, guard=.1):
    folder=Path(folder).resolve()
    info=json.loads((folder/'retained_review.json').read_text(encoding='utf-8'))
    data,metadata=load_raw(Path(info['source']))
    events=pd.read_csv(folder/'retained_review.csv').to_dict('records')
    analysis=analyze_noise(data,events,window_s=1.0,guard_s=guard,degree=1)
    target=folder if guard==.1 else folder/f'guard_{guard:.2f}s'
    save_noise_analysis(target,analysis,info['source'],metadata)
    print(analysis['summary'].to_string(index=False))
    return analysis


if __name__=='__main__':
    main(sys.argv[1],float(sys.argv[2]) if len(sys.argv)>2 else .1)
