import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QLabel,QAbstractButton,QGroupBox,QComboBox,QMessageBox
import resonance_gui as gui
from resonance_fpga_gui import FPGAMainWindow
from fpga_i18n import DisplayCatalog,HANGUL,source_widget_text,source_combo_text,FileDialog
from test_resonance_fpga_gui import FakeFPGA


class CatalogTests(unittest.TestCase):
    def test_production_messages_have_english_translations(self):
        catalog=DisplayCatalog()
        names=['resonance_gui.py','resonance_fpga_gui.py','fpga_auto_sweep_gui.py',
               'fpga_refinement_gui.py','fpga_initial_resonance.py','fpga_pll_refinement.py',
               'fpga_auto_sweep.py','fpga_delay_calibration.py','fpga_pll_response.py','fpga_stream_health.py',
               'resonance_fit.py','pressure_monitor.py','beam_modes.py','live_savgol.py',
               'baseline_analysis.py','experiment_analysis.py','particle_review_core.py']
        missing=[]
        for name in names:
            tree=ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and HANGUL.search(catalog.english(node.value)):
                    missing.append((name,node.lineno,node.value))
        self.assertEqual(missing,[])

    def test_dynamic_status_preserves_numbers_and_units(self):
        text='FPGA PLL 실행 · 적용 BW 80 Hz / AGC 켬 · lock bit 미제공'
        translated=DisplayCatalog().english(text)
        self.assertIn('80 Hz',translated)
        self.assertIn('AGC On',translated)
        self.assertNotRegex(translated,'[가-힣]')
        self.assertEqual(DisplayCatalog().english('unknown future text'),'unknown future text')
        path=r'C:\Data\모드\경고.csv'
        self.assertEqual(DisplayCatalog().english('저장됨: '+path),'Saved: '+path)


class LanguageGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.fake=FakeFPGA()
        self.state_patch=patch.object(gui,'STATE_DIR',Path(self.folder.name))
        self.connect_patch=patch.object(FPGAMainWindow,'_connect_hardware',lambda window:setattr(window,'fpga',self.fake))
        self.state_patch.start();self.connect_patch.start()
        self.addCleanup(self.state_patch.stop);self.addCleanup(self.connect_patch.stop)
        self.window=FPGAMainWindow()
        self.addCleanup(self.window.close)

    def english(self):
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData('en'))
        self.window.language_manager.refresh()

    def test_static_ui_translates_and_roundtrips_without_device_writes(self):
        w=self.window
        center=w.pll_center_spin.value();delay=w.fpga_pll_delay_spin.value()
        sweep=w.sweep_worker;plotter=w.plotter_worker
        with patch.object(self.fake,'write_named',wraps=self.fake.write_named) as writes:
            self.english()
            self.assertEqual(w.pll_enable_btn.text(),'Enable PLL')
            self.assertEqual(w.tabs.tabText(3),'④ Live particle detection')
            for obj in w.findChildren(QLabel)+w.findChildren(QAbstractButton)+w.findChildren(QGroupBox):
                if obj.property('fpgaLanguageSelector'):continue
                text=obj.text() if hasattr(obj,'text') else obj.title()
                self.assertNotRegex(text,'[가-힣]',text)
            w.language_combo.setCurrentIndex(w.language_combo.findData('ko'))
            self.assertEqual(w.pll_enable_btn.text(),'PLL 활성화')
            writes.assert_not_called()
        self.assertEqual(w.pll_center_spin.value(),center)
        self.assertEqual(w.fpga_pll_delay_spin.value(),delay)
        self.assertIs(w.sweep_worker,sweep);self.assertIs(w.plotter_worker,plotter)

    def test_new_worker_status_translates_and_returns_to_korean(self):
        self.english()
        source='초기 공진 검증: fine (36000–41000 Hz)'
        self.window.pll_design_label.setText(source)
        QTest.qWait(220)
        self.assertEqual(self.window.pll_design_label.text(),'Initial resonance verification: fine (36000–41000 Hz)')
        self.assertEqual(source_widget_text(self.window.pll_design_label),source)
        self.window.live_calibration_progress.setFormat('blank 학습 2.0 / 8.0 s')
        self.window.language_manager.refresh()
        self.assertEqual(self.window.live_calibration_progress.format(),'Blank calibration 2.0 / 8.0 s')
        self.window.language_combo.setCurrentIndex(0)
        self.assertEqual(self.window.pll_design_label.text(),source)
        self.assertEqual(self.window.live_calibration_progress.format(),'blank 학습 2.0 / 8.0 s')

    def test_switch_during_stream_keeps_same_worker_and_receives_samples(self):
        w=self.window
        w.pll_center_spin.setValue(118000)
        w.pll_range_spin.setValue(1000)
        w.live_particle_enabled.setChecked(False)
        self.fake.apply_parameters(w._fpga_parameters())
        self.fake.start_tracking()
        w.pll_enable_btn.blockSignals(True);w.pll_enable_btn.setChecked(True);w.pll_enable_btn.blockSignals(False)
        w._pll_start_validated=True
        w.start_plotter()
        worker=w.plotter_worker
        QTest.qWait(750)
        before=w.received_sample_count
        generation=w.live_particle_generation
        starts=self.fake.stream_starts
        stops=self.fake.stream_stops
        self.english()
        QTest.qWait(150)
        self.assertIs(w.plotter_worker,worker)
        self.assertTrue(worker.isRunning())
        self.assertGreater(w.received_sample_count,before)
        self.assertEqual(w.live_particle_generation,generation)
        self.assertEqual(self.fake.stream_starts,starts)
        self.assertEqual(self.fake.stream_stops,stops)

    def test_language_persists_without_translating_user_metadata(self):
        self.window.metadata_fluid_edit.setText('한국어 DI water')
        self.english()
        self.assertEqual(self.window.metadata_fluid_edit.text(),'한국어 DI water')
        path=Path(self.folder.name)/'fpga'/'language.json'
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['language'],'en')
        self.window._save_last_settings()
        saved=json.loads((path.parent/'last_settings.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['values']['fpga_pll_delay_spin'],self.window.fpga_pll_delay_spin.value())
        second=FPGAMainWindow()
        try:
            self.assertEqual(second.language_combo.currentData(),'en')
            self.assertEqual(second.pll_enable_btn.text(),'Enable PLL')
        finally:second.close()

    def test_combo_item_data_and_canonical_source_are_preserved(self):
        box=QComboBox(self.window)
        box.addItem('전체 모드','canonical-mode')
        self.english()
        self.assertEqual(box.currentText(),'All modes')
        self.assertEqual(box.currentData(),'canonical-mode')
        self.assertEqual(source_combo_text(box),'전체 모드')

    def test_pressure_cleanup_uses_canonical_state_in_english(self):
        self.english()
        self.window.pressure_readback_label.setText('압력 읽기 중지 중…')
        self.window.language_manager.refresh()
        self.window._pressure_finished()
        self.window.language_manager.refresh()
        self.assertEqual(self.window.pressure_readback_label.text(),'Pressure readback stopped · A: —  B: —')

    def test_auto_sweep_lock_keeps_language_and_emergency_stop_available(self):
        self.english()
        self.window._lock_auto_controls(True)
        try:
            self.assertTrue(self.window.language_combo.isEnabled())
            self.assertTrue(self.window.findChild(QAbstractButton,'emergency_stop').isEnabled())
            self.window.language_combo.setCurrentIndex(0)
            self.assertEqual(self.window.pll_enable_btn.text(),'PLL 활성화')
        finally:self.window._lock_auto_controls(False)

    def test_popup_and_plot_titles_use_current_language(self):
        self.english()
        box=QMessageBox(self.window)
        box.setWindowTitle('FPGA PLL 시작 실패')
        box.setText('진폭 최대 구간이 잡음/양자화와 구별되지 않습니다.')
        # Qt's Windows offscreen plugin crashes even for a plain message box
        # show(). Validate the exact dialog object without platform painting.
        self.assertTrue(self.window.language_manager._owned(box))
        self.window.language_manager.refresh(box)
        self.assertEqual(box.windowTitle(),'FPGA PLL startup failed')
        self.assertNotRegex(box.text(),'[가-힣]')
        box.close()
        self.assertEqual(self.window.auto_amp_plot.plotItem.titleLabel.text,'All modes / Coarse · Fine')

    def test_file_dialog_caption_is_localized(self):
        self.english()
        with patch('fpga_i18n.QtFileDialog.getSaveFileName',return_value=('','')) as dialog:
            FileDialog.getSaveFileName(self.window,'스윕 데이터 저장','sweep.csv')
        self.assertEqual(dialog.call_args.args[1],'Save sweep data')


if __name__=='__main__':unittest.main()
