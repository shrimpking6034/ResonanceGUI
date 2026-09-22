import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from particle_review_mit_signal import analyze_mit_signal,save_mit_signal_analysis
from particle_review_noise import analyze_noise,recompute_event_depths


class MitSignalTests(unittest.TestCase):
    @staticmethod
    def seven_antinode_data():
        t=np.arange(0.,20.,.001)
        f=100+.01*np.sin(2*np.pi*.2*t)+.003*np.sin(2*np.pi*180*t)
        events=[]
        for center in (5.,10.,15.):
            for antinode in center+np.linspace(-.06,.06,7):
                f-=np.exp(-.5*((t-antinode)/.0025)**2)
            events.append(dict(time_s=center,event_start_s=center-.07,event_end_s=center+.07,depth_hz=1.,status='미검토',note=''))
        return pd.DataFrame(dict(Time_s=t,Frequency_Hz=f)),events

    def test_mit_pipeline_recovers_seven_antinodes_and_snr(self):
        data,events=self.seven_antinode_data()
        result=analyze_mit_signal(data,events,sg_order=4,sg_window=27,prominence_sigma=2,
                                  noise_guard_s=.1,noise_min_duration_s=1.)
        self.assertEqual(result['summary'].antinode_count.tolist(),[7,7,7])
        self.assertTrue((result['summary'].status=='ok').all())
        self.assertTrue((result['summary'].snr_mit>0).all())
        self.assertEqual(len(result['antinodes']),21)
        self.assertEqual(result['settings']['baseline_degree'],3)
        self.assertEqual(result['settings']['local_extremum_degree'],2)
        with tempfile.TemporaryDirectory() as folder:
            save_mit_signal_analysis(folder,result,'raw.csv')
            self.assertTrue((Path(folder)/'mit_peak_summary.csv').exists())
            settings=json.loads((Path(folder)/'mit_signal_analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(settings['settings']['sg_window'],27)

    def test_sg_noise_recomputes_depth_on_filtered_signal(self):
        data,events=self.seven_antinode_data()
        filtered=data.copy();filtered['Frequency_Hz']=data.Frequency_Hz.rolling(9,center=True,min_periods=1).mean()
        result=analyze_noise(filtered,events,window_s=.5,guard_s=.1,degree=1,
                             signal_source='test filtered',recompute_depth=True)
        self.assertEqual(result['settings']['signal_source'],'test filtered')
        self.assertTrue(result['summary'].depth_hz.notna().all())
        self.assertTrue((result['summary'].depth_hz!=result['summary'].saved_raw_depth_hz).all())
        updated=recompute_event_depths(filtered,events,'SG test')
        self.assertTrue(all(item['depth_signal_source']=='SG test' for item in updated))
        self.assertTrue(all('raw_depth_hz' in item for item in updated))


if __name__=='__main__':unittest.main()
