"""Standalone offline GUI. Run: python sweep_comparison_gui.py [folders ...]."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import sys

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox, QColorDialog, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QGroupBox, QScrollArea, QProgressDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from sweep_comparison import (Style, PlotSpec, load_dataset, build_specs, ordinal,
    render_figure, save_figure, write_project, plot_traces, validate_spec, figure_size_mm, q_value_text)


def button(text, callback, layout):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    layout.addWidget(widget)
    return widget


class PreviewArea(QScrollArea):
    resized = pyqtSignal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class ComparisonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_font = Path('C:/Windows/Fonts/malgun.ttf')
        if ui_font.exists():
            QFontDatabase.addApplicationFont(str(ui_font))
            self.setFont(QFont('Malgun Gothic', 9))
        self.setWindowTitle('Offline Sweep Comparison')
        self.resize(1500, 940)
        self.datasets, self.specs = [], []
        self.spec_cache = {}
        self.style = Style()
        self.current = None
        self.loading = False
        self.dirty = False
        self.project_path = None
        self.home = Path(__file__).resolve().parent / '기본연구'
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        bar = QHBoxLayout()
        button('Open project', self.open_project, bar)
        button('Save project', self.save_project, bar)
        button('Save project as...', lambda: self.save_project(save_as=True), bar)
        bar.addStretch()
        self.summary = QLabel('Offline files only | No FPGA connection')
        bar.addWidget(self.summary)
        outer.addLayout(bar)
        splitter = QSplitter()
        outer.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel('1. Folders / legends (double-click legend to edit)'))
        row = QHBoxLayout()
        button('Choose subfolders', self.choose_subfolders, row)
        button('Add folder', self.add_folder, row)
        left_layout.addLayout(row)
        row = QHBoxLayout()
        button('Remove', self.remove_folder, row)
        button('Reload files', self.reload_files, row)
        left_layout.addLayout(row)
        self.folder_table = QTableWidget(0, 4)
        self.folder_table.setHorizontalHeaderLabels(['Use', 'Legend', 'Color', 'Folder'])
        self.folder_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.folder_table.setColumnWidth(0, 38)
        self.folder_table.setColumnWidth(2, 62)
        self.folder_table.setColumnWidth(3, 130)
        self.folder_table.setMaximumHeight(220)
        self.folder_table.itemChanged.connect(self.folder_edited)
        self.folder_table.cellDoubleClicked.connect(self.pick_color)
        left_layout.addWidget(self.folder_table)

        self.stage_checks = self.check_group(left_layout, '2. Stages', ['coarse', 'fine', 'fit'], ['coarse', 'fine', 'fit'])
        self.kind_checks = self.check_group(left_layout, 'Figures', ['individual', 'comparison', 'overview'], ['individual', 'comparison', 'overview'])
        self.quantity_checks = self.check_group(left_layout, 'Quantity', ['amplitude', 'phase'], ['amplitude'])
        row = QHBoxLayout()
        row.addWidget(QLabel('Measurement'))
        self.attempt = QComboBox()
        self.attempt.addItems(['auto', 'original', 'retry'])
        self.attempt.setToolTip('Auto uses the last complete attempt. Fit uses the saved fit curve.')
        self.attempt.currentTextChanged.connect(self.selection_changed)
        row.addWidget(self.attempt)
        left_layout.addLayout(row)
        self.review_data = QCheckBox('Include incomplete / unknown / review data')
        self.review_data.setToolTip('Unchecked: completed sweeps and accepted fits only.')
        self.review_data.toggled.connect(self.selection_changed)
        left_layout.addWidget(self.review_data)
        button('Generate / refresh plot list', self.generate, left_layout)
        self.plot_table = QTableWidget(0, 3)
        self.plot_table.setHorizontalHeaderLabels(['Save', 'Reviewed', 'Plot'])
        self.plot_table.setColumnWidth(0, 42)
        self.plot_table.setColumnWidth(1, 68)
        self.plot_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.plot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.plot_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.plot_table.itemChanged.connect(self.plot_check_changed)
        self.plot_table.currentCellChanged.connect(self.plot_selected)
        left_layout.addWidget(self.plot_table, 1)
        row = QHBoxLayout()
        button('Select all', lambda: self.select_all(True), row)
        button('Select none', lambda: self.select_all(False), row)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        self.figure = Figure(figsize=(5, 5), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        middle_layout.addWidget(self.toolbar)
        self.preview_area = PreviewArea()
        self.preview_area.setAlignment(Qt.AlignCenter)
        self.preview_area.setWidget(self.canvas)
        self.preview_area.resized.connect(self.fit_preview)
        middle_layout.addWidget(self.preview_area, 1)
        button('Use current zoom as saved axis limits', self.capture_zoom, middle_layout)
        self.sources = QLabel('Add saved auto-sweep folders, then generate figures.')
        self.sources.setWordWrap(True)
        self.sources.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_scroll = QScrollArea()
        source_scroll.setWidgetResizable(True)
        source_scroll.setWidget(self.sources)
        source_scroll.setMaximumHeight(155)
        middle_layout.addWidget(source_scroll)
        splitter.addWidget(middle)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        form = QFormLayout()
        right_layout.addWidget(QLabel('3. Current plot'))
        self.title = QLineEdit()
        form.addRow('Title', self.title)
        self.title.editingFinished.connect(self.apply_plot)
        self.axis_fields = {}
        for key, label in [('xmin', 'X min (Hz)'), ('xmax', 'X max (Hz)'), ('ymin', 'Y min'), ('ymax', 'Y max')]:
            field = QLineEdit()
            field.setPlaceholderText('Auto')
            field.editingFinished.connect(self.apply_plot)
            self.axis_fields[key] = field
            form.addRow(label, field)
        self.legend = QComboBox()
        self.legend.addItems(['best', 'upper right', 'upper left', 'lower right', 'lower left', 'outside', 'none'])
        self.legend.currentTextChanged.connect(self.apply_plot)
        form.addRow('Legend position', self.legend)
        self.show_q = QCheckBox('Show Q from saved fit')
        self.show_q.setToolTip('Per-mode: legend or a colored list. Overview: a Q table below the plot.')
        self.show_q.toggled.connect(self.apply_plot)
        form.addRow(self.show_q)
        self.q_position = QComboBox()
        self.q_position.addItems(['legend', 'upper left', 'upper right', 'lower left', 'lower right'])
        self.q_position.currentTextChanged.connect(self.apply_plot)
        form.addRow('Q position (mode)', self.q_position)
        self.log_y = QCheckBox('Logarithmic amplitude')
        self.normalize = QCheckBox('Normalize each peak to 1')
        self.log_y.toggled.connect(self.apply_plot)
        self.normalize.toggled.connect(self.apply_plot)
        form.addRow(self.log_y)
        form.addRow(self.normalize)
        right_layout.addLayout(form)
        button('Apply Q settings to all plots', self.apply_q_all, right_layout)
        button('Reset axis limits', self.reset_axes, right_layout)
        self.template = QLineEdit('{ordinal} mode')
        self.template.setToolTip('Example: {ordinal} mode | {stage}. Applies to all mode plots.')
        right_layout.addWidget(self.template)
        button('Apply title template to mode plots', self.apply_template, right_layout)

        right_layout.addWidget(QLabel('4. Shared publication style'))
        style_form = QFormLayout()
        self.font = QComboBox()
        self.font.addItems(['Arial', 'DejaVu Sans', 'Times New Roman'])
        style_form.addRow('Font', self.font)
        self.style_fields = {}
        for key, name, low, high, step in [('font_size', 'All text (pt)', 6, 24, .5),
                ('line_width', 'Line width (pt)', .3, 4, .1),
                ('square_mm', 'Square size (mm)', 60, 300, 5),
                ('wide_mm', 'Overview width (mm)', 100, 400, 5),
                ('height_mm', 'Overview height (mm)', 60, 300, 5)]:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(getattr(self.style, key))
            self.style_fields[key] = spin
            style_form.addRow(name, spin)
            spin.valueChanged.connect(self.apply_style)
        self.unit = QComboBox()
        self.unit.addItems(['Hz', 'kHz'])
        self.unit.currentTextChanged.connect(self.apply_style)
        self.font.currentTextChanged.connect(self.apply_style)
        style_form.addRow('Frequency unit', self.unit)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(600)
        self.dpi.valueChanged.connect(self.apply_style)
        style_form.addRow('PNG DPI', self.dpi)
        right_layout.addLayout(style_form)
        self.formats = self.check_group(right_layout, '5. Export formats', ['pdf', 'svg', 'png'], ['pdf', 'png'], connect=False)
        self.only_reviewed = QCheckBox('Export reviewed plots only')
        right_layout.addWidget(self.only_reviewed)
        button('Export selected plots...', self.export_plots, right_layout)
        right_layout.addStretch()
        scroll = QScrollArea()
        scroll.setMinimumWidth(295)
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([430, 700, 330])
        self.statusBar().showMessage('Ready. Source measurement files are read only.')

    def check_group(self, layout, name, options, defaults, connect=True):
        group = QGroupBox(name)
        row = QHBoxLayout(group)
        checks = {}
        for option in options:
            check = QCheckBox(option.title())
            check.setChecked(option in defaults)
            if connect:
                check.toggled.connect(self.selection_changed)
            row.addWidget(check)
            checks[option] = check
        layout.addWidget(group)
        return checks

    def options(self):
        return {'stages': [k for k, w in self.stage_checks.items() if w.isChecked()],
                'kinds': [k for k, w in self.kind_checks.items() if w.isChecked()],
                'quantities': [k for k, w in self.quantity_checks.items() if w.isChecked()],
                'attempt': self.attempt.currentText(), 'include_review': self.review_data.isChecked()}

    def mark_dirty(self, invalidate=False):
        self.dirty = True
        if invalidate:
            for spec in list(self.spec_cache.values()) + self.specs:
                spec.reviewed = False
            self.refresh_plot_table()

    def error(self, message):
        QMessageBox.warning(self, 'Offline Sweep Comparison', str(message))

    def choose_subfolders(self):
        parent = QFileDialog.getExistingDirectory(self, 'Choose parent containing sweep folders', str(self.home))
        if not parent:
            return
        paths = sorted(p for p in Path(parent).iterdir() if p.is_dir() and any(p.glob('mode_*.*')))
        if not paths:
            self.error('No saved sweep subfolders were found. Use Add folder for a single session.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Select sweep folders')
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)
        listing = QListWidget()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            listing.addItem(item)
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            self.load_paths([listing.item(i).data(Qt.UserRole) for i in range(listing.count()) if listing.item(i).checkState() == Qt.Checked])

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose saved sweep folder', str(self.home))
        if path:
            self.load_paths([path])

    def load_paths(self, paths):
        errors = []
        for path in paths:
            if str(Path(path).resolve()) in [d.path for d in self.datasets]:
                continue
            try:
                self.datasets.append(load_dataset(path, len(self.datasets)))
            except Exception as exc:
                errors.append(str(exc))
        self.refresh_folders()
        self.mark_dirty(True)
        self.generate()
        if errors:
            self.error('\n'.join(errors))

    def refresh_folders(self):
        self.folder_table.blockSignals(True)
        self.folder_table.setRowCount(len(self.datasets))
        for row, dataset in enumerate(self.datasets):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if dataset.enabled else Qt.Unchecked)
            self.folder_table.setItem(row, 0, item)
            self.folder_table.setItem(row, 1, QTableWidgetItem(dataset.label))
            color = QTableWidgetItem(dataset.color)
            color.setForeground(QColor(dataset.color))
            color.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.folder_table.setItem(row, 2, color)
            name = QTableWidgetItem(Path(dataset.path).name)
            name.setToolTip(dataset.path + '\n' + '\n'.join(dataset.warnings))
            name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.folder_table.setItem(row, 3, name)
        self.folder_table.blockSignals(False)

    def folder_edited(self, item):
        if item.column() == 0:
            self.datasets[item.row()].enabled = item.checkState() == Qt.Checked
            self.selection_changed()
        elif item.column() == 1:
            self.datasets[item.row()].label = item.text()
            self.mark_dirty(True)
            self.draw()

    def pick_color(self, row, column):
        if column != 2:
            return
        color = QColorDialog.getColor(QColor(self.datasets[row].color), self)
        if color.isValid():
            self.datasets[row].color = color.name()
            self.refresh_folders()
            self.mark_dirty(True)
            self.draw()

    def remove_folder(self):
        row = self.folder_table.currentRow()
        if row >= 0:
            self.datasets.pop(row)
            self.refresh_folders()
            self.selection_changed()

    def reload_files(self):
        errors = []
        for i, old in enumerate(self.datasets):
            try:
                new = load_dataset(old.path, i)
                new.label, new.color, new.enabled = old.label, old.color, old.enabled
                self.datasets[i] = new
            except Exception as exc:
                errors.append(str(exc))
        self.refresh_folders()
        self.selection_changed()
        if errors:
            self.error('\n'.join(errors))

    def selection_changed(self, *_):
        if self.loading:
            return
        self.mark_dirty(True)
        self.generate()

    def generate(self):
        old_key = self.current.key if self.current else None
        self.spec_cache.update({s.key: s for s in self.specs})
        self.specs = build_specs(self.datasets, previous=self.spec_cache.values(), **self.options())
        self.spec_cache.update({s.key: s for s in self.specs})
        self.current = None
        self.refresh_plot_table()
        index = next((i for i, s in enumerate(self.specs) if s.key == old_key), 0)
        if self.specs:
            self.plot_table.setCurrentCell(index, 2)
            self.plot_selected(index)
        else:
            self.figure.clear()
            self.canvas.draw_idle()
            self.sources.setText('No matching data. Check folder selection, stage, and measurement status.\n'
                                 'Use Include incomplete / unknown / review data to inspect unverified files.')
        warnings = sum(len(d.warnings) for d in self.datasets)
        self.summary.setText(f'{len(self.datasets)} folders | {len(self.specs)} plots | {warnings} load warnings')
        self.dirty = True

    def refresh_plot_table(self):
        self.plot_table.blockSignals(True)
        self.plot_table.setRowCount(len(self.specs))
        for row, spec in enumerate(self.specs):
            for col, value in enumerate([spec.selected, spec.reviewed]):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if value else Qt.Unchecked)
                self.plot_table.setItem(row, col, item)
            suffix = ''
            if spec.kind == 'individual':
                suffix = ' | ' + next((d.label for d in self.datasets if d.path in spec.datasets), '')
            name = QTableWidgetItem(f'{spec.kind} | {spec.stage} | {spec.title} | {spec.quantity}{suffix}')
            name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            name.setToolTip(name.text())
            self.plot_table.setItem(row, 2, name)
        self.plot_table.blockSignals(False)

    def plot_check_changed(self, item):
        if item.column() in (0, 1):
            setattr(self.specs[item.row()], 'selected' if item.column() == 0 else 'reviewed', item.checkState() == Qt.Checked)
            self.dirty = True

    def select_all(self, value):
        for spec in self.specs:
            spec.selected = value
        self.refresh_plot_table()
        self.dirty = True

    def plot_selected(self, row, *_):
        if not 0 <= row < len(self.specs):
            return
        if self.current is not None and self.current is not self.specs[row] and not self.apply_plot():
            self.plot_table.blockSignals(True)
            self.plot_table.setCurrentCell(self.specs.index(self.current), 2)
            self.plot_table.blockSignals(False)
            self.error('Correct the current plot fields before switching plots.')
            return
        self.current = self.specs[row]
        self.load_plot_fields()
        self.draw()

    def load_plot_fields(self):
        self.loading = True
        spec = self.current
        self.title.setText(spec.title)
        for key, field in self.axis_fields.items():
            field.setText('' if getattr(spec, key) is None else f'{getattr(spec, key):.12g}')
        self.legend.setCurrentText(spec.legend)
        self.show_q.setChecked(spec.show_q)
        self.q_position.setCurrentText(spec.q_position)
        self.q_position.setEnabled(spec.show_q and spec.kind != 'overview')
        self.log_y.setChecked(spec.log_y)
        self.normalize.setChecked(spec.normalized)
        self.log_y.setEnabled(spec.quantity == 'amplitude')
        self.normalize.setEnabled(spec.quantity == 'amplitude')
        self.loading = False

    def apply_plot(self, *_):
        if self.loading or self.current is None:
            return True
        try:
            changes = {key: float(field.text()) if field.text().strip() else None for key, field in self.axis_fields.items()}
            changes.update(title=self.title.text(), legend=self.legend.currentText(), log_y=self.log_y.isChecked(), normalized=self.normalize.isChecked(),
                           show_q=self.show_q.isChecked(), q_position=self.q_position.currentText())
            candidate = PlotSpec(**{**asdict(self.current), **changes})
            validate_spec(candidate, self.datasets, self.style)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return False
        if any(getattr(self.current, key) != value for key, value in changes.items()):
            for key, value in changes.items():
                setattr(self.current, key, value)
            self.current.reviewed = False
            self.dirty = True
            self.refresh_plot_table()
            self.draw()
        self.q_position.setEnabled(self.current.show_q and self.current.kind != 'overview')
        return True

    def apply_q_all(self):
        if not self.apply_plot():
            self.error('Correct the current plot fields before applying Q settings.')
            return
        for spec in list(self.spec_cache.values()) + self.specs:
            if spec.show_q != self.show_q.isChecked() or spec.q_position != self.q_position.currentText():
                spec.show_q, spec.q_position = self.show_q.isChecked(), self.q_position.currentText()
                spec.reviewed = False
        self.dirty = True
        self.refresh_plot_table()
        self.draw()

    def draw(self):
        if self.current is None:
            return
        try:
            render_figure(self.current, self.datasets, self.style, self.attempt.currentText(), self.review_data.isChecked(), self.figure)
            self.fit_preview()
            self.toolbar.update()
            self.canvas.draw_idle()
            lines = []
            for dataset, trace in plot_traces(self.current, self.datasets, self.attempt.currentText(), self.review_data.isChecked()):
                lines.append(f'{dataset.label}: {trace.file} | {trace.status}'
                             f' | {trace.freqs[0]:.3f} - {trace.freqs[-1]:.3f} Hz | {len(trace.freqs)} points'
                             f' | Q (saved fit): {q_value_text(dataset, trace.mode, self.review_data.isChecked())}'
                             + (f' | fit accepted: {trace.accepted}' if trace.stage == 'fit' else ''))
            missing = [d.label for d in self.datasets if d.enabled and d.path not in self.current.datasets]
            if self.current.kind == 'comparison' and missing:
                lines.append('No eligible trace: ' + ', '.join(missing))
            for d in self.datasets:
                if d.path in self.current.datasets:
                    lines.extend(d.warnings)
            self.sources.setText('\n'.join(lines))
            self.statusBar().showMessage('Preview ready. Use the zoom button to preserve toolbar zoom for export.')
        except Exception as exc:
            self.figure.clear()
            self.canvas.draw_idle()
            self.sources.setText(str(exc))
            self.statusBar().showMessage(str(exc))

    def fit_preview(self):
        if self.current is None:
            return
        width, height = figure_size_mm(self.current, self.datasets, self.style, self.attempt.currentText(), self.review_data.isChecked())
        available = self.preview_area.viewport().size()
        scale = max(.1, min((available.width() - 8) / width, (available.height() - 8) / height))
        # Change preview DPI rather than physical figure size; typography and layout
        # therefore match the exported figure at any window size.
        self.figure.set_dpi(scale * 25.4)
        self.canvas.setFixedSize(max(1, round(width * scale)), max(1, round(height * scale)))
        self.figure.set_size_inches(width / 25.4, height / 25.4, forward=False)
        self.canvas.draw_idle()

    def capture_zoom(self):
        if not self.current or not self.figure.axes:
            return
        ax = self.figure.axes[0]
        factor = 1000 if self.style.unit == 'kHz' else 1
        self.current.xmin, self.current.xmax = [v * factor for v in ax.get_xlim()]
        self.current.ymin, self.current.ymax = ax.get_ylim()
        self.current.reviewed = False
        self.dirty = True
        self.load_plot_fields()
        self.refresh_plot_table()

    def reset_axes(self):
        if self.current:
            for field in self.axis_fields.values():
                field.setText('')
            self.apply_plot()

    def apply_template(self):
        try:
            changes = [(s, self.template.text().format(ordinal=ordinal(s.mode), mode=s.mode, stage=s.stage.title())) for s in self.specs if s.mode is not None]
            for spec, title in changes:
                validate_spec(PlotSpec(**{**asdict(spec), 'title': title}), self.datasets, self.style)
            for spec, title in changes:
                spec.title, spec.reviewed = title, False
            self.dirty = True
            self.refresh_plot_table()
            if self.current:
                self.load_plot_fields()
                self.draw()
        except (KeyError, ValueError, IndexError) as exc:
            self.error(f'Invalid title template: {exc}')

    def apply_style(self, *_):
        if self.loading:
            return
        for key, widget in self.style_fields.items():
            setattr(self.style, key, widget.value())
        self.style.font = self.font.currentText()
        self.style.unit = self.unit.currentText()
        self.style.dpi = self.dpi.value()
        self.mark_dirty(True)
        self.draw()

    def save_project(self, checked=False, save_as=False):
        if not self.apply_plot():
            self.error('Correct the current plot fields before saving.')
            return False
        path = self.project_path if not save_as else None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Save comparison project', str(self.home / 'sweep_comparison.json'), 'JSON (*.json)')
        if not path:
            return False
        try:
            self.spec_cache.update({s.key: s for s in self.specs})
            write_project(path, self.datasets, list(self.spec_cache.values()), self.style, self.options())
            self.project_path = path
            self.dirty = False
            self.statusBar().showMessage(f'Project saved: {path}')
            return True
        except Exception as exc:
            self.error(exc)
            return False

    def may_discard(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Unsaved project', 'Save comparison settings before continuing?', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Save:
            return self.save_project()
        return answer == QMessageBox.Discard

    def open_project(self):
        if not self.may_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Open comparison project', str(self.home), 'JSON (*.json)')
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            if payload.get('version') != 1:
                raise ValueError('Unsupported comparison project version')
            datasets, remapped = [], {}
            source_changed = False
            for entry in payload['datasets']:
                folder = entry['path']
                if not Path(folder).is_dir():
                    folder = QFileDialog.getExistingDirectory(self, f'Locate moved folder: {Path(folder).name}', str(self.home))
                    if not folder:
                        return
                dataset = load_dataset(folder, len(datasets))
                source_changed |= entry.get('fingerprint') != dataset.fingerprint
                dataset.label, dataset.color, dataset.enabled = entry['label'], entry['color'], entry['enabled']
                if not QColor(dataset.color).isValid():
                    raise ValueError('Invalid saved color')
                remapped[entry['path']] = dataset.path
                datasets.append(dataset)
            q_migrated = any('show_q' not in entry for entry in payload['plots'])
            specs = [PlotSpec(**entry) for entry in payload['plots']]
            for spec in specs:
                if source_changed or q_migrated:
                    spec.reviewed = False
                spec.datasets = [remapped.get(p, p) for p in spec.datasets]
                key = json.loads(spec.key)
                key[-1] = [remapped.get(p, p) for p in key[-1]]
                spec.key = json.dumps(key)
            style = Style(**payload['style'])
            options = payload['options']
            self.loading = True
            for group, values in [(self.stage_checks, options['stages']), (self.kind_checks, options['kinds']), (self.quantity_checks, options['quantities'])]:
                for key, widget in group.items():
                    widget.setChecked(key in values)
            self.attempt.setCurrentText(options['attempt'])
            self.review_data.setChecked(options['include_review'])
            for key, widget in self.style_fields.items():
                widget.setValue(getattr(style, key))
            self.font.setCurrentText(style.font)
            self.unit.setCurrentText(style.unit)
            self.dpi.setValue(style.dpi)
            self.datasets, self.specs, self.style = datasets, specs, style
            self.spec_cache = {s.key: s for s in specs}
            self.loading = False
            self.refresh_folders()
            self.generate()
            self.project_path = path
            self.dirty = source_changed or q_migrated
            if source_changed:
                self.statusBar().showMessage('Source files changed since this project was saved. Review flags have been reset.')
            elif q_migrated:
                self.statusBar().showMessage('Q display was added to this older project. Review flags have been reset.')
        except Exception as exc:
            self.loading = False
            self.error(exc)

    def export_plots(self):
        if not self.apply_plot():
            self.error('Correct the current plot fields before exporting.')
            return
        specs = [s for s in self.specs if s.selected and (s.reviewed or not self.only_reviewed.isChecked())]
        formats = [k for k, w in self.formats.items() if w.isChecked()]
        if not specs or not formats:
            self.error('Select at least one plot and export format.')
            return
        try:
            for spec in specs:
                validate_spec(spec, self.datasets, self.style)
        except ValueError as exc:
            self.error(exc)
            return
        parent = QFileDialog.getExistingDirectory(self, 'Choose export parent folder', str(self.home))
        if not parent:
            return
        directory = Path(parent) / ('sweep_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        report = {'completed': [], 'errors': [], 'cancelled': False}
        progress = QProgressDialog('Exporting publication figures...', 'Cancel', 0, len(specs), self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        try:
            directory.mkdir(parents=True)
            self.spec_cache.update({s.key: s for s in self.specs})
            write_project(directory / 'comparison_project.json', self.datasets, list(self.spec_cache.values()), self.style, self.options())
            for index, spec in enumerate(specs):
                progress.setValue(index)
                QApplication.processEvents()
                if progress.wasCanceled():
                    report['cancelled'] = True
                    break
                try:
                    fig = render_figure(spec, self.datasets, self.style, self.attempt.currentText(), self.review_data.isChecked())
                    output = directory / spec.kind / spec.stage
                    if spec.kind == 'individual':
                        number = next(i + 1 for i, d in enumerate(self.datasets) if d.path in spec.datasets)
                        output /= f'dataset_{number:02d}'
                    output.mkdir(parents=True, exist_ok=True)
                    stem = ('all_modes' if spec.mode is None else f'mode_{spec.mode:02d}') + '_' + spec.quantity
                    for extension in formats:
                        target = output / f'{stem}.{extension}'
                        save_figure(fig, target, self.style)
                        report['completed'].append({'file': str(target.relative_to(directory)), 'title': spec.title,
                            'sources': [{'folder': d.path, 'file': t.file, 'status': t.status, 'fit_accepted': t.accepted,
                                         'q_saved_fit': d.fit_results.get(t.mode, {})}
                                        for d, t in plot_traces(spec, self.datasets, self.attempt.currentText(), self.review_data.isChecked())]})
                    fig.clear()
                except Exception as exc:
                    report['errors'].append({'plot': spec.key, 'error': str(exc)})
            (directory / 'export_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            progress.setValue(len(specs))
            QMessageBox.information(self, 'Export finished', f"Saved {len(report['completed'])} files.\nErrors: {len(report['errors'])}\nCancelled: {report['cancelled']}\n\n{directory}")
        except Exception as exc:
            self.error(exc)
        finally:
            progress.close()

    def closeEvent(self, event):
        if self.may_discard():
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = ComparisonWindow()
    if len(sys.argv) > 1:
        window.load_paths(sys.argv[1:])
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
