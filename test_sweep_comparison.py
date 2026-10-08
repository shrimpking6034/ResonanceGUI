"""Offline regression tests; no instrument or source measurement mutations."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np

from sweep_comparison import (Style, load_dataset, default_label, ordinal, select_trace,
    build_specs, render_figure, save_figure, write_project, validate_spec, q_value_text, q_table_data, PlotSpec)


def make_session(root, name='auto_sweep_20261002_140305_873655_chamber_close_vac', offset=0):
    folder = root / name
    folder.mkdir()
    records = []
    for mode in (1, 2):
        f = np.linspace(mode * 1000 + offset, mode * 1000 + 20 + offset, 11)
        values = np.exp(-((f - f.mean()) / 4) ** 2) * 100
        stages = []
        for stage in ('coarse', 'coarse_retry', 'fine', 'fine_retry', 'fit'):
            np.savez(folder / f'mode_{mode:02d}_{stage}.npz', freqs=f, R=values, theta=np.linspace(-100, 100, 11))
            if stage != 'fit':
                stages.append({'stage': stage, 'status': 'stopped' if stage == 'fine_retry' else 'complete'})
        records.append({'mode': mode, 'status': 'complete', 'fit_accepted': mode == 1,
                        'q': 249.425 if mode == 1 else 125.25, 'stages': stages})
    (folder / 'session.json').write_text(json.dumps({'records': records}), encoding='utf-8')
    return folder


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = make_session(self.root)
        self.dataset = load_dataset(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_labels_and_ordinals(self):
        self.assertEqual(self.dataset.label, 'chamber close vac')
        self.assertEqual(default_label('auto_sweep_20261002_140305_873655', 3), 'Dataset 3')
        self.assertEqual([ordinal(n) for n in [1, 2, 3, 4, 11, 12, 13, 21]],
                         ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st'])

    def test_attempt_and_quality_selection(self):
        self.assertEqual(select_trace(self.dataset, 1, 'coarse').attempt, 'retry')
        self.assertEqual(select_trace(self.dataset, 1, 'fine').attempt, 'original')
        self.assertIsNone(select_trace(self.dataset, 1, 'fine', 'retry'))
        self.assertEqual(select_trace(self.dataset, 1, 'fine', 'retry', True).status, 'stopped')
        self.assertIsNone(select_trace(self.dataset, 2, 'fit'))
        self.assertIsNotNone(select_trace(self.dataset, 2, 'fit', include_review=True))

    def test_csv_fallback_and_invalid_samples(self):
        npz = self.folder / 'mode_01_coarse.npz'
        npz.write_bytes(b'not an npz')
        npz.with_suffix('.csv').write_text('# Metadata,stage,coarse\nFrequency_Hz,Amplitude_R_counts,Phase_deg\n3,4,5\n1,2,3\nnan,7,8\n', encoding='utf-8')
        data = load_dataset(self.folder)
        trace = select_trace(data, 1, 'coarse', 'original')
        self.assertTrue(trace.file.endswith('.csv'))
        np.testing.assert_equal(trace.freqs, [1, 3])
        self.assertTrue(data.warnings)

    def test_grouping_no_cross_stage_or_mode(self):
        second = load_dataset(make_session(self.root, 'second', offset=2), 1)
        datasets = [self.dataset, second]
        specs = build_specs(datasets, ['coarse', 'fine', 'fit'], ['individual', 'comparison', 'overview'], ['amplitude'])
        self.assertEqual(len(specs), 18)  # 7 coarse + 7 fine + 4 accepted fit
        spec = next(s for s in specs if s.kind == 'comparison' and s.mode == 1 and s.stage == 'fine')
        fig = render_figure(spec, datasets, Style())
        self.assertEqual(len(fig.axes[0].lines), 2)
        np.testing.assert_equal(fig.axes[0].lines[0].get_xdata(), select_trace(self.dataset, 1, 'fine').freqs)
        spec.title, spec.xmin = 'Edited mode', 999
        regenerated = build_specs(datasets, ['fine'], ['comparison'], ['amplitude'], previous=specs)
        self.assertEqual(regenerated[0].title, 'Edited mode')
        self.assertEqual(regenerated[0].xmin, 999)

    def test_overview_gaps_sizes_and_english(self):
        specs = build_specs([self.dataset], ['coarse'], ['overview'], ['amplitude'])
        style = Style(unit='kHz')
        fig = render_figure(specs[0], [self.dataset], style)
        self.assertEqual(len(fig.axes[0].lines), 2)
        np.testing.assert_allclose(fig.get_size_inches() * 25.4, [180, 90])
        self.assertEqual(fig.axes[0].get_xlabel(), 'Frequency (kHz)')
        self.assertEqual(len(fig.axes[0].get_legend().get_texts()), 1)
        for extension in ('pdf', 'svg', 'png'):
            target = self.root / f'output.{extension}'
            save_figure(fig, target, style)
            self.assertGreater(target.stat().st_size, 1000)
        specs[0].title = '한글'
        with self.assertRaises(ValueError):
            validate_spec(specs[0], [self.dataset], style)

    def test_project_round_trip_payload(self):
        specs = build_specs([self.dataset], ['fine'], ['individual'], ['amplitude'])
        specs[0].reviewed = True
        specs[0].ymax = 120
        target = self.root / 'project.json'
        write_project(target, [self.dataset], specs, Style(), {'attempt': 'auto'})
        data = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(data['datasets'][0]['path'], str(self.folder.resolve()))
        self.assertTrue(data['plots'][0]['reviewed'])
        self.assertEqual(data['plots'][0]['ymax'], 120)
        with self.assertRaises(ValueError):
            write_project(self.folder / 'session.json', [self.dataset], specs, Style(), {})

    def test_q_quality_and_invalid_values(self):
        self.assertEqual(q_value_text(self.dataset, 1), '249.4')
        self.assertEqual(q_value_text(self.dataset, 2), 'N/A')
        self.assertEqual(q_value_text(self.dataset, 2, True), '125.2*')
        self.assertEqual(q_value_text(self.dataset, 99, True), 'N/A')
        path = self.folder / 'session.json'
        payload = json.loads(path.read_text())
        for invalid in [None, 'bad', -1, float('nan'), float('inf')]:
            payload['records'][0]['q'] = invalid
            path.write_text(json.dumps(payload))
            self.assertEqual(q_value_text(load_dataset(self.folder), 1, True), 'N/A')

    def test_q_legend_box_table_and_hide(self):
        second = load_dataset(make_session(self.root, 'second'), 1)
        second.traces = [t for t in second.traces if t.mode == 1]
        datasets = [self.dataset, second]
        spec = build_specs(datasets, ['coarse'], ['comparison'], ['amplitude'])[0]
        fig = render_figure(spec, datasets, Style())
        self.assertIn('Q (fit) = 249.4', fig.axes[0].get_legend().get_texts()[0].get_text())
        spec.legend = 'none'
        self.assertEqual(len(render_figure(spec, datasets, Style()).axes[0].artists), 1)
        spec.legend = 'best'
        spec.q_position = 'lower left'
        fig = render_figure(spec, datasets, Style())
        self.assertEqual(len(fig.axes[0].artists), 1)
        self.assertNotIn('Q (fit)', fig.axes[0].get_legend().get_texts()[0].get_text())
        spec.show_q = False
        fig = render_figure(spec, datasets, Style())
        self.assertEqual(len(fig.axes[0].artists), 0)
        overview = build_specs(datasets, ['coarse'], ['overview'], ['amplitude'])[0]
        _, rows, _ = q_table_data(overview, datasets)
        self.assertEqual(rows, [['1st', '249.4', '249.4'], ['2nd', 'N/A', '--']])
        fig = render_figure(overview, datasets, Style())
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[1].tables[0][2, 2].get_text().get_text(), '--')
        overview.show_q = False
        self.assertEqual(len(render_figure(overview, datasets, Style()).axes), 1)
        # Older projects have no Q fields; constructor defaults keep them readable.
        old = dict(spec.__dict__)
        del old['show_q'], old['q_position']
        self.assertTrue(PlotSpec(**old).show_q)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_edit_export_and_reopen_without_hardware(self):
        from sweep_comparison_gui import ComparisonWindow
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = make_session(root)
            window = ComparisonWindow()
            window.show()
            self.app.processEvents()
            window.load_paths([folder])
            self.app.processEvents()
            window.title.setText('First resonance')
            self.assertTrue(window.apply_plot())
            window.axis_fields['xmin'].setText('999')
            self.assertTrue(window.apply_plot())
            saved_key = window.current.key
            window.stage_checks['coarse'].setChecked(False)
            window.stage_checks['coarse'].setChecked(True)
            edited = next(s for s in window.specs if s.key == saved_key)
            self.assertEqual(edited.title, 'First resonance')
            window.plot_table.setCurrentCell(window.specs.index(edited), 2)
            window.q_position.setCurrentText('lower right')
            window.apply_q_all()
            self.assertTrue(all(s.show_q and s.q_position == 'lower right' for s in window.specs))
            window.current.reviewed = True
            window.project_path = str(root / 'review.json')
            self.assertTrue(window.save_project())
            for spec in window.specs:
                spec.selected = spec is window.current
            with patch('sweep_comparison_gui.QFileDialog.getExistingDirectory', return_value=str(root)), \
                 patch('sweep_comparison_gui.QMessageBox.information'):
                window.export_plots()
            report = next(root.glob('sweep_comparison_*/export_report.json'))
            payload = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(len(payload['completed']), 2)
            self.assertFalse(payload['errors'])
            window.dirty = False
            with patch('sweep_comparison_gui.QFileDialog.getOpenFileName', return_value=(str(root / 'review.json'), '')):
                window.open_project()
            self.assertEqual(window.current.title, 'First resonance')
            self.assertEqual(window.current.xmin, 999)
            self.assertTrue(window.current.show_q)
            self.assertEqual(window.current.q_position, 'lower right')
            self.assertTrue(window.current.reviewed)
            changed = folder / 'mode_01_coarse.npz'
            stamp = changed.stat().st_mtime_ns
            os.utime(changed, ns=(stamp + 1000000, stamp + 1000000))
            with patch('sweep_comparison_gui.QFileDialog.getOpenFileName', return_value=(str(root / 'review.json'), '')):
                window.open_project()
            self.assertFalse(any(s.reviewed for s in window.specs))
            self.assertNotIn('fpga_backend', __import__('sys').modules)
            self.assertNotIn('fpga_transport', __import__('sys').modules)
            window.dirty = False
            window.close()


if __name__ == '__main__':
    unittest.main()
