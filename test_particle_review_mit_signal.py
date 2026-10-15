import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from particle_review_mit_signal import analyze_mit_signal,save_mit_signal_analysis,compare_mit_signal_settings,_quadratic_extremum
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

    def test_expected_count_follows_selected_mode(self):
        t=np.arange(0.,12.,.001);f=100+0*t;center=6.
        for antinode in center+np.linspace(-.04,.04,5):
            f-=np.exp(-.5*((t-antinode)/.0025)**2)
        data=pd.DataFrame(dict(Time_s=t,Frequency_Hz=f))
        events=[dict(time_s=center,event_start_s=center-.05,event_end_s=center+.05,depth_hz=1.)]
        result=analyze_mit_signal(data,events,sg_order=4,sg_window=27,
                                  prominence_sigma=2,antinode_count=5)
        self.assertEqual(result['settings']['antinode_count'],5)
        self.assertEqual(result['summary'].antinode_count.tolist(),[5])
        self.assertEqual(len(result['antinodes']),5)

    def test_manual_times_can_restore_missing_detection(self):
        t=np.arange(0.,10.,.001);center=5.;expected=center+np.linspace(-.06,.06,7)
        f=100+0*t
        for antinode in np.delete(expected,3):
            f-=np.exp(-.5*((t-antinode)/.0025)**2)
        data=pd.DataFrame(dict(Time_s=t,Frequency_Hz=f))
        event=dict(time_s=center,event_start_s=center-.07,event_end_s=center+.07,depth_hz=1.)
        automatic=analyze_mit_signal(data,[event],sg_order=2,sg_window=27,
                                     prominence_sigma=10,antinode_count=7)
        self.assertNotEqual(int(automatic['summary'].iloc[0].antinode_count),7)
        self.assertGreater(len(automatic['antinodes']),0)
        manual=analyze_mit_signal(data,[event],sg_order=2,sg_window=27,
                                  prominence_sigma=10,antinode_count=7,
                                  manual_antinode_times={1:expected.tolist()})
        self.assertEqual(int(manual['summary'].iloc[0].antinode_count),7)
        self.assertEqual(manual['summary'].iloc[0].selection_source,'manual')
        self.assertTrue((manual['antinodes'].selection_source=='manual').all())

    def test_baseline_excludes_transit_and_sg_support(self):
        data,events=self.seven_antinode_data()
        event=events[1]
        result=analyze_mit_signal(data,[event],sg_window=27,baseline_guard_s=.01)
        trace=result['traces'];used=trace[trace.baseline_used]
        guard=result['settings']['effective_baseline_guard_s']
        self.assertTrue(((used.time_s<event['event_start_s']-guard)|(used.time_s>event['event_end_s']+guard)).all())
        self.assertTrue((used.time_s<event['event_start_s']).any())
        self.assertTrue((used.time_s>event['event_end_s']).any())
        self.assertGreater(len(result['fit_curves']),0)
        self.assertEqual(result['summary'].iloc[0].noise_source,'local_before_after')
        self.assertTrue(np.isfinite(result['summary'].iloc[0].snr_mit))

    def test_drift_does_not_shift_residual_fitted_extrema(self):
        t=np.arange(0,10,.001);centers=np.linspace(4.8,5.2,5)
        raw=100+.8*(t-5)
        for a in centers:raw-=.5*np.exp(-.5*((t-a)/.012)**2)
        event=dict(time_s=5.,event_start_s=4.75,event_end_s=5.25)
        result=analyze_mit_signal(pd.DataFrame(dict(Time_s=t,Frequency_Hz=raw)),[event],
                                  sg_order=3,sg_window=11,antinode_count=5,baseline_degree=1,local_fit_fraction=.1)
        self.assertEqual(result['summary'].iloc[0].status,'ok')
        a=result['antinodes']
        self.assertAlmostEqual(a.time_s.iloc[0],centers[0],places=3)
        self.assertAlmostEqual(a.time_s.iloc[-1],centers[-1],places=3)
        self.assertAlmostEqual(a.delta_hz.iloc[0],a.delta_hz.iloc[-1],places=3)

    def test_extra_antinodes_are_not_silently_dropped(self):
        data,events=self.seven_antinode_data()
        result=analyze_mit_signal(data,[events[1]],antinode_count=5,sg_window=9)
        self.assertGreater(result['summary'].iloc[0].antinode_count,5)
        self.assertEqual(result['summary'].iloc[0].status,'antinode_count')

    def test_invalid_quadratic_vertex_is_rejected(self):
        t=np.linspace(-1,1,101)
        with self.assertRaises(ValueError):_quadratic_extremum(t,t*t,50,10,'up')
        with self.assertRaises(ValueError):_quadratic_extremum(t,-(t-2)**2,50,10,'up')

    def test_pressure_transition_defers_acceptance(self):
        data,events=self.seven_antinode_data()
        data['PressureCommand_A_V']=(data.Time_s>=10).astype(float)
        result=analyze_mit_signal(data,[events[1]],sg_window=27)
        self.assertEqual(result['summary'].iloc[0].status,'pressure_transition')

    def test_sensitivity_variants_and_exports(self):
        data,events=self.seven_antinode_data()
        result=analyze_mit_signal(data,[events[1]],sg_window=27)
        comparison=compare_mit_signal_settings(data,[events[1]],dict(sg_order=4,sg_window=27))
        self.assertEqual(len(comparison),5)
        self.assertEqual(set(comparison.baseline_degree),{1,2,3})
        self.assertEqual(comparison.sg_window_ms.nunique(),3)
        result['sensitivity']=comparison
        with tempfile.TemporaryDirectory() as folder:
            save_mit_signal_analysis(folder,result,'raw.csv')
            self.assertTrue((Path(folder)/'mit_processing_sensitivity.csv').exists())
            self.assertTrue((Path(folder)/'mit_quadratic_curves.csv').exists())

    def test_gui_sensitivity_and_changed_settings_invalidate_results(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from unittest.mock import patch
        from particle_review_gui import ReviewWindow
        from particle_review_core import manual_event
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        data,events=self.seven_antinode_data()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.csv';data.to_csv(path,index=False)
            window=ReviewWindow(calibration_store_path=Path(folder)/'cal.json')
            errors=[];window.error=errors.append
            try:
                window.load_file(path)
                event=manual_event(window.data,10.);event.update(events[1]);window.events=[event]
                window.refresh_table();window.table.selectRow(0)
                window.mode_number.setValue(7);window.savgol_panel.window.setValue(27)
                window.calculate_mit_signal()
                with patch.object(QtWidgets.QDialog,'exec_',return_value=0):window.compare_mit_sensitivity()
                self.assertEqual(errors,[])
                self.assertEqual(len(window.mit_signal_panel.analysis['sensitivity']),5)
                window.savgol_panel.window.setValue(25)
                self.assertIsNone(window.mit_signal_panel.analysis)
            finally:window.close();app.processEvents()

    def test_point_candidate_has_preview_and_can_apply_bounds_in_mit_tab(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        from particle_review_core import manual_event
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        data,_=self.seven_antinode_data()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.csv';data.to_csv(path,index=False)
            window=ReviewWindow(calibration_store_path=Path(folder)/'cal.json')
            errors=[];window.error=errors.append
            try:
                window.load_file(path);window.events=[manual_event(window.data,10.)]
                window.refresh_table();window.table.selectRow(0)
                window.mode_number.setValue(7);window.savgol_panel.window.setValue(27)
                window.calculate_mit_signal()
                panel=window.mit_signal_panel
                self.assertEqual(errors,[])
                self.assertEqual(panel.analysis['summary'].iloc[0].status,'invalid_event')
                self.assertGreater(len(panel.analysis['traces']),0)
                self.assertEqual(window.events[0]['event_start_s'],window.events[0]['event_end_s'])
                self.assertIn('구간 미지정',panel.interval_info.text())
                panel.event_start.setValue(9.93);panel.event_end.setValue(10.07)
                panel.apply_bounds_btn.click()
                self.assertEqual(errors,[])
                self.assertEqual(window.events[0]['event_start_s'],9.93)
                self.assertEqual(window.events[0]['event_end_s'],10.07)
                self.assertNotEqual(panel.analysis['summary'].iloc[0].status,'invalid_event')
            finally:window.close();app.processEvents()


if __name__=='__main__':unittest.main()
