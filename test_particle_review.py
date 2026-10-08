import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import patch
from particle_review_core import (Settings, detect, calibrate_detector, detect_calibrated,
                                  pressure_changes, load_raw, manual_event,
                                  event_context_noise, batch_context_noise,
                                  ParticlePropertySettings, estimate_particle_properties,
                                  load_sensitivity_profiles, save_sensitivity_profile)


class CoreTests(unittest.TestCase):
    def test_sensitivity_profile_persists_by_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sensitivity.json'
            save_sensitivity_profile(path,{'mode_number':6,'sensitivity_pg_hz':12.5},'reference.csv')
            save_sensitivity_profile(path,{'mode_number':4,'sensitivity_pg_hz':8.25})
            profiles=load_sensitivity_profiles(path)
            self.assertEqual(set(profiles),{'4','6'})
            self.assertEqual(profiles['6']['sensitivity_pg_hz'],12.5)
            self.assertTrue(profiles['6']['source_file'].endswith('reference.csv'))

    def test_round_trip_representative_and_conditional_inference(self):
        events=[{'depth_hz':1.},{'depth_hz':3.},{'depth_hz':2.},{'depth_hz':4.}]
        settings=ParticlePropertySettings(diameter_min_um=200,diameter_max_um=200,
            fluid_density_kg_m3=1000,reference_density_kg_m3=8000,
            representative_method='mean',inference_method='density_from_diameter')
        rows,cal=estimate_particle_properties(events,settings)
        self.assertEqual([row['round_trip_pair'] for row in rows],[1,1,2,2])
        self.assertEqual([row['direction'] for row in rows],['forward','reverse','forward','reverse'])
        self.assertEqual([row['pair_depth_hz'] for row in rows],[2.,2.,3.,3.])
        self.assertAlmostEqual(cal['representative_depth_hz'],2.5)
        self.assertAlmostEqual(cal['representative_density_kg_m3'],8000,places=8)
        self.assertNotIn('buoyant_mass_pg',rows[0])
        direct=ParticlePropertySettings(diameter_min_um=200,diameter_max_um=200,
            fluid_density_kg_m3=1000,calibration_method='sensitivity',sensitivity_pg_hz=cal['sensitivity_pg_hz'],
            representative_method='median',inference_method='diameter_from_density',reference_density_kg_m3=8000)
        _,direct_cal=estimate_particle_properties(events,direct)
        self.assertAlmostEqual(direct_cal['representative_buoyant_mass_pg'],cal['representative_buoyant_mass_pg'])
        self.assertAlmostEqual(direct_cal['representative_equivalent_diameter_um'],200.)

    def test_incomplete_pair_is_not_reindexed(self):
        events=[{'depth_hz':1.},{'depth_hz':np.nan},{'depth_hz':3.},{'depth_hz':5.},{'depth_hz':7.}]
        settings=ParticlePropertySettings(calibration_method='sensitivity',sensitivity_pg_hz=10)
        rows,cal=estimate_particle_properties(events,settings)
        self.assertFalse(rows[0]['pair_complete']); self.assertFalse(rows[1]['pair_complete'])
        self.assertEqual(rows[2]['round_trip_pair'],2); self.assertEqual(rows[2]['pair_depth_hz'],4.)
        self.assertFalse(rows[4]['pair_complete'])
        self.assertEqual(cal['complete_pair_count'],1)
        self.assertEqual(cal['excluded_event_count'],3)

    def test_supplied_twenty_pass_example_statistics(self):
        depths=[.29433965228963643,.34679706926908693,.2886901675919944,.2801358598553634,
                .26308154159778496,.3285047826284426,.28343483544449555,.30902182637146325,
                .2482328147671069,.32509400259004906,.2901539569720626,.3196529902306793,
                .2757570694993774,.34158457966987044,.29159254320984473,.31984899505550857,
                .32299258436614764,.32405235487021855,.29157415531153674,.31018299374409253]
        _,cal=estimate_particle_properties([{'depth_hz':value} for value in depths],
            ParticlePropertySettings(calibration_method='sensitivity',sensitivity_pg_hz=1.))
        self.assertAlmostEqual(cal['representative_depth_hz'],.3027362388)
        self.assertAlmostEqual(cal['individual_stats']['cv_percent'],8.6344545)
        self.assertAlmostEqual(cal['pair_stats']['cv_percent'],4.2458465)
        self.assertAlmostEqual(cal['direction_mean_forward_hz'],.2849849321)
        self.assertAlmostEqual(cal['direction_mean_reverse_hz'],.3204875454)
        self.assertAlmostEqual(cal['direction_difference_percent'],11.7272427)

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
        self.assertLess(near[0]['event_start_s'],near[0]['time_s'])
        self.assertLess(near[0]['time_s'],near[0]['event_end_s'])
        self.assertEqual(near[0]['selection_source'],'auto')
        self.assertTrue(np.isnan(near[0]['contrast_hz']))
        self.assertTrue(all(event['family']=='short' for event in events))

    def test_constant_no_candidates(self):
        d=self.synthetic();d['Frequency_Hz']=30000
        self.assertEqual(detect(d,Settings())[0],[])

    def test_frozen_calibration_matches_offline_detector(self):
        data = self.synthetic()
        settings = Settings(analyze_start=7, analyze_end=13)
        offline, thresholds = detect(data, settings)
        live = detect_calibrated(data, settings, calibrate_detector(data, settings))
        offline_rows = [(row['family'], row['time_s']) for row in offline]
        live_rows = [(row['family'], row['time_s']) for row in live]
        self.assertEqual(live_rows, offline_rows)
        self.assertEqual(len(thresholds),1)
        self.assertEqual(thresholds[0]['metric'],'local_depth')

    def test_wide_search_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'short'):
            detect(self.synthetic(),Settings(widths=(.40,)))

    def test_minimum_depth_controls_detection(self):
        events,_=detect(self.synthetic(),Settings(floor_hz=1.0))
        self.assertEqual(events,[])

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
    def test_reference_calibration_is_remembered_and_auto_applied(self):
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            store=Path(folder)/'sensitivity.json'
            data=CoreTests().synthetic()
            first=ReviewWindow(store)
            first_event=manual_event(data,9.9); first_event['depth_hz']=.8
            second_event=manual_event(data,10); second_event['depth_hz']=.8
            first.events=[first_event,second_event]
            first.diameter_min.setValue(200); first.diameter_max.setValue(200)
            first.fluid_density.setValue(1000); first.reference_density.setValue(8000)
            first.calculate_particle_properties()
            sensitivity=first.mass_sensitivity.value()
            self.assertTrue(store.exists()); self.assertEqual(first.mass_method.currentData(),'sensitivity')
            first.close()
            second=ReviewWindow(store)
            self.assertEqual(second.mass_method.currentData(),'sensitivity')
            self.assertAlmostEqual(second.mass_sensitivity.value(),sensitivity,places=3)
            event1=manual_event(data,9.9); event1['depth_hz']=2.
            event2=manual_event(data,10); event2['depth_hz']=4.
            second.events=[event1,event2]; second.thresholds=[]; second.pending_settings=Settings()
            second.analysis_done(second.events,[])
            self.assertAlmostEqual(second.property_calibration['representative_buoyant_mass_pg'],3*sensitivity,places=2)
            self.assertIn(f'{sensitivity:,.3f}',second.summary_sensitivity.text())
            self.assertIn(f'{3*sensitivity:.4g}',second.summary_mass.text())
            self.assertEqual(second.property_summary_state.text(),'자동 역산 완료')
            second.close(); app.processEvents()

    def test_left_controls_scroll_instead_of_squeezing_table(self):
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        w=ReviewWindow(); w.resize(1050,650); w.show(); app.processEvents()
        self.assertGreaterEqual(w.table.minimumHeight(),200)
        self.assertGreater(w.left_scroll.verticalScrollBar().maximum(),0)
        self.assertTrue(w.summary_sensitivity.isVisible())
        w.close(); app.processEvents()

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

    def test_dragged_interval_updates_and_persists_event(self):
        from PyQt5 import QtWidgets
        from particle_review_gui import ReviewWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw.csv'; CoreTests().synthetic().to_csv(source,index=False)
            target=Path(folder)/'adjusted.csv'
            w=ReviewWindow(); w.error=lambda message:self.fail(message); w.load_file(str(source))
            w.events=[manual_event(w.data,10)]; w.refresh_table(); w.table.selectRow(0)
            w.sample_region.setRegion([9.94,10.08]); w.apply_selected_region()
            event=w.events[0]
            self.assertAlmostEqual(event['time_s'],10.01,places=6)
            self.assertAlmostEqual(event['duration_s'],.14,places=6)
            self.assertEqual(event['selection_source'],'manual')
            self.assertLessEqual(event['event_start_s'],event['peak_time_s'])
            self.assertLessEqual(event['peak_time_s'],event['event_end_s'])
            with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):w.export_results()
            w.events=[]; w.refresh_table()
            with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(str(target),'CSV')):w.import_review()
            self.assertAlmostEqual(w.events[0]['event_start_s'],9.94,places=6)
            self.assertAlmostEqual(w.events[0]['event_end_s'],10.08,places=6)
            self.assertEqual(w.events[0]['selection_source'],'manual')
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
            panels=list((result/'publication').glob('*'))
            self.assertEqual(len(panels),24)
            for quantity in ('frequency','phase','amplitude','pressure'):
                for extension in ('png','pdf','svg'):
                    self.assertEqual(len(list((result/'publication').glob(f'*_{quantity}.{extension}'))),2)
            with Image.open(next((result/'publication').glob('*_frequency.png'))) as im:
                self.assertAlmostEqual(im.width,90/25.4*600,delta=1)
                self.assertEqual(im.width,im.height)
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
