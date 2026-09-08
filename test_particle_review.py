import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import patch
from particle_review_core import Settings,detect,pressure_changes,load_raw,manual_event,event_context_noise,batch_context_noise


class CoreTests(unittest.TestCase):
    def test_context_noise_before_after_and_no_smoothing(self):
        d=self.synthetic(); rows=event_context_noise(d,10,1,.1,1)
        self.assertEqual([r['context'] for r in rows],['before','after'])
        self.assertEqual([r['sample_count'] for r in rows],[101,101])
        self.assertTrue(all(r['detrended_std_hz']>=0 for r in rows))
        out=batch_context_noise(d,[{'time_s':10,'status':'검토'}],1,.1,1)
        self.assertEqual(len(out),2); self.assertIn('threshold_10sigma_hz',out)
    def synthetic(self):
        t=np.arange(0,20,.01)
        f=30000+.2*t+np.random.default_rng(7).normal(0,.01,len(t))
        f-=.8*np.exp(-.5*((t-10)/.025)**2)
        return pd.DataFrame({'Time_s':t,'Frequency_Hz':f})

    def test_dip_on_rising_baseline(self):
        events,_=detect(self.synthetic(),Settings())
        near=[e for e in events if e['family']=='short' and abs(e['time_s']-10)<.06]
        self.assertEqual(len(near),1)
        self.assertGreater(near[0]['depth_hz'],.7)

    def test_constant_no_candidates(self):
        d=self.synthetic();d['Frequency_Hz']=30000
        self.assertEqual(detect(d,Settings())[0],[])

    def test_gaps_not_silently_resampled(self):
        d=self.synthetic().drop(index=[500])
        with self.assertRaisesRegex(ValueError,'불규칙'):detect(d,Settings())

    def test_pressure_steps_include_stop(self):
        d=pd.DataFrame({'Time_s':[0,1,2,3,4], 'PressureCommand_A_V':[0,.2,.2,.2,0], 'PressureCommand_B_V':[0,0,.4,.4,0]})
        self.assertEqual(list(pressure_changes(d).Time_s),[1,2,4])

    def test_manual_and_range(self):
        d=self.synthetic()
        self.assertAlmostEqual(manual_event(d,10)['depth_hz'],.8,delta=.05)
        events,_=detect(d,Settings(analyze_start=11,analyze_end=19))
        self.assertFalse(any(e['time_s']<11 for e in events))

    def test_known_434_event(self):
        path=Path(__file__).parent/'raw_20260908_170941_b6f23895_6thmode_220um_2.csv'
        if not path.exists():self.skipTest('Experimental CSV not distributed')
        d,_=load_raw(path)
        events,_=detect(d,Settings(reference_start=2,reference_end=12))
        near=[e for e in events if e['family']=='short' and abs(e['time_s']-434.1334)<.03]
        self.assertEqual(len(near),1)
        self.assertAlmostEqual(near[0]['depth_hz'],.9466,delta=.03)


class GuiTests(unittest.TestCase):
    def test_keep_exact_review_status(self):
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        w=ReviewWindow(); d=CoreTests().synthetic()
        w.events=[manual_event(d,t) for t in [8,10,12]]
        for e,status in zip(w.events,['미검토','검토','검토 완료']):e['status']=status
        w.refresh_table()
        with patch.object(QtWidgets.QInputDialog,'getItem',return_value=('검토',True)), patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):w.keep_status()
        self.assertEqual([e['time_s'] for e in w.events],[10])
        w.undo_remove(); self.assertEqual(len(w.events),3)
        w.close(); app.processEvents()

    def test_remove_multiple_undo_preserves_annotations(self):
        from PyQt5 import QtWidgets,QtCore
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        w=ReviewWindow(); d=CoreTests().synthetic()
        w.events=[manual_event(d,time) for time in [8,10,12,14]]
        for i,e in enumerate(w.events):e['note']=str(i)
        w.refresh_table()
        selection=w.table.selectionModel()
        for row in [1,3]:selection.select(w.table.model().index(row,0),QtCore.QItemSelectionModel.Select|QtCore.QItemSelectionModel.Rows)
        w.remove_selected()
        self.assertEqual([e['time_s'] for e in w.events],[8,12])
        w.table.item(0,7).setText('삭제 후 편집')
        w.undo_remove()
        self.assertEqual([e['time_s'] for e in w.events],[8,10,12,14])
        self.assertEqual(w.events[0]['note'],'삭제 후 편집')
        self.assertEqual(w.events[1]['note'],'1')
        w.remove_rows(range(4)); self.assertEqual(w.table.rowCount(),0)
        w.undo_remove(); self.assertEqual(w.table.rowCount(),4)
        w.close(); app.processEvents()

    def test_open_select_save_resume(self):
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow,configure_application
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_application(app)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw.csv'
            CoreTests().synthetic().to_csv(source,index=False)
            w=ReviewWindow()
            w.error=lambda message: self.fail(message)
            w.load_file(str(source)); w.pending_settings=w.get_settings()
            w.analysis_done(*detect(w.data,w.pending_settings))
            self.assertGreater(w.table.rowCount(),0)
            w.table.selectRow(0); w.residual.setChecked(True)
            w.table.item(0,6).setText('후보 유지')
            w.table.item(0,7).setText('테스트 메모')
            target=Path(folder)/'review.csv'
            with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):
                w.export_results()
            self.assertTrue(target.with_suffix('.json').exists())
            w.events=[]; w.refresh_table()
            with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(str(target),'CSV')):
                w.import_review()
            self.assertEqual(w.events[0]['status'],'후보 유지')
            self.assertEqual(w.events[0]['note'],'테스트 메모')
            w.close(); app.processEvents()


class ExportTests(unittest.TestCase):
    def test_all_retained_rows_unique_images_and_notes(self):
        from particle_review_export import export_event_figures
        import json
        from PIL import Image
        d=CoreTests().synthetic()
        event=manual_event(d,10); event.update(status='검토',note='6')
        with tempfile.TemporaryDirectory() as folder:
            result,count=export_event_figures(d,[event,dict(event,family='wide')],folder,'raw.csv',1.1,True,review_context={'metadata':{'Fluid':'DI'},'thresholds':[{'threshold_hz':.15}]})
            self.assertEqual(count,2)
            images=list(result.glob('*.png')); self.assertEqual(len(images),2)
            for image in images:
                with Image.open(image) as im:self.assertGreater(im.width,1000)
            exported=pd.read_csv(result/'exported_events.csv',dtype={'note':str})
            self.assertEqual(exported.note.tolist(),['6','6'])
            self.assertTrue(exported.residual_applied.all())
            info=json.loads((result/'retained_review.json').read_text(encoding='utf-8'))
            self.assertEqual(info['completed_count'],2)
            self.assertEqual(info['metadata']['Fluid'],'DI')

    def test_cancel_preserves_partial_and_all_review_rows(self):
        from particle_review_export import export_event_figures
        import json
        d=CoreTests().synthetic(); events=[manual_event(d,t) for t in [8,10,12]]
        stop=[False]
        with tempfile.TemporaryDirectory() as folder:
            result,count=export_event_figures(d,events,folder,'raw.csv',1.1,progress=lambda n,total:stop.__setitem__(0,True),cancelled=lambda:stop[0])
            self.assertEqual(count,1)
            self.assertEqual(len(pd.read_csv(result/'retained_review.csv')),3)
            info=json.loads((result/'retained_review.json').read_text(encoding='utf-8'))
            self.assertTrue(info['cancelled'])


if __name__=='__main__':unittest.main()
