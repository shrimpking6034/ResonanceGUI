import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from particle_review_noise import analyze_noise,save_noise_analysis


class NoiseTests(unittest.TestCase):
    def test_sample_count_uses_timestamps_and_gui_region(self):
        from particle_review_core import sample_window_stats
        t=np.array([0.,.1,.2,.5])
        self.assertEqual(sample_window_stats(t,.1,.2)['sample_count'],2)
        self.assertEqual(sample_window_stats(t,.21,.49)['sample_count'],0)
        self.assertTrue(np.isnan(sample_window_stats(t,.5,.5)['sampling_rate_hz']))
        self.assertEqual(sample_window_stats(t,0.,.5)['large_gap_count'],1)
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw.csv'; self.data().to_csv(source,index=False)
            w=ReviewWindow(); w.load_file(source)
            w.sample_region.setRegion([3.,3.15])
            self.assertIn('16개',w.sample_info.text())
            w.draw_detail()
            self.assertEqual(tuple(w.sample_region.getRegion()),(3.,3.15))
            w.sample_end.setValue(3.2)
            self.assertIn('21개',w.sample_info.text())
            w.close(); app.processEvents()

    def test_batch_export_respects_noise_settings_and_row_alignment(self):
        from particle_review_export import export_event_figures
        from particle_review_core import manual_event
        d=self.data(); events=[manual_event(d,3.),manual_event(d,5.)]
        with tempfile.TemporaryDirectory() as folder:
            out,count=export_event_figures(d,events,folder,'raw.csv',1.1,review_context={'noise_settings':dict(window_s=.8,guard_s=.3,degree=2)})
            settings=json.loads((out/'noise_snr/snr_analysis_settings.json').read_text(encoding='utf-8'))
            self.assertEqual(settings['degree'],2)
            self.assertAlmostEqual(settings['guard_s'],.3)
            exported=pd.read_csv(out/'exported_events.csv')
            contexts=pd.read_csv(out/'context_noise.csv')
            np.testing.assert_allclose(exported.raw_std_hz_before,contexts[contexts.context=='before'].raw_std_hz)

    def data(self):
        t=np.arange(0,8,.01)
        return pd.DataFrame(dict(Time_s=t,Frequency_Hz=100+2*t+.08*np.sin(t*120)))

    def test_pooled_and_invalid_boundary(self):
        d=self.data(); events=[dict(time_s=4.,depth_hz=1.),dict(time_s=0.,depth_hz=1.)]
        a=analyze_noise(d,events)
        r=a['summary'].iloc[0]; p=a['residuals']; p=p[p.event_index==1]
        self.assertAlmostEqual(r.sigma_pooled_hz,np.sqrt(np.mean(p.residual_hz**2)),places=9)
        self.assertAlmostEqual(r.snr_pooled,1/r.sigma_pooled_hz)
        self.assertTrue(np.isnan(a['summary'].iloc[1].snr_pooled))
        self.assertIsNone(a['summary'].iloc[1].meets_10sigma)
        with tempfile.TemporaryDirectory() as path:
            save_noise_analysis(path,a,'test.csv')
            self.assertTrue((Path(path)/'snr_context_overview.png').exists())
            self.assertEqual(json.loads((Path(path)/'snr_analysis_settings.json').read_text(encoding='utf-8'))['guard_s'],.25)

    def test_gui_compute_settings_invalidate_and_resume(self):
        from unittest.mock import patch
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        from particle_review_core import manual_event
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'raw.csv'; self.data().to_csv(raw,index=False)
            w=ReviewWindow(); w.error=lambda e:self.fail(e); w.load_file(raw)
            w.events=[manual_event(w.data,4.)]; w.refresh_table()
            w.calculate_context_noise()
            self.assertEqual(w.noise_panel.table.rowCount(),1)
            self.assertEqual(w.right_tabs.currentWidget(),w.noise_panel)
            w.noise_guard.setValue(.3)
            self.assertIsNone(w.noise_panel.analysis)
            w.calculate_context_noise()
            self.assertEqual(w.noise_panel.analysis['settings']['guard_s'],.3)
            target=Path(folder)/'review.csv'
            with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):w.export_results()
            w.events=[]; w.refresh_table(); w.noise_guard.setValue(.1)
            with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(str(target),'CSV')):w.import_review()
            self.assertAlmostEqual(w.noise_guard.value(),.3)
            w.close(); app.processEvents()


if __name__=='__main__':unittest.main()
