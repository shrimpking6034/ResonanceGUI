"""Offline sweep loading, plot recipes, and publication rendering (no hardware imports)."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import io
import json
import re
import textwrap
from zipfile import BadZipFile

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import rc_context
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker


COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#333333', '#7B3294']
FILE_RE = re.compile(r'^mode_(\d+)_(coarse|fine|fit)(_retry)?\.(npz|csv)$')


def ordinal(n):
    return str(n) + ('th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))


def default_label(path, index=1):
    suffix = re.sub(r'^auto_sweep_\d{8}_\d{6}_\d+(?:_|$)', '', Path(path).name)
    return suffix.replace('_', ' ').strip() or f'Dataset {index}'


def english_text(value):
    # Printable ASCII plus scientific symbols; reject non-Latin language text.
    return all(32 <= ord(c) < 127 or c in 'µμ°ΔθπΩ±−×²³·' for c in value)


@dataclass
class Trace:
    mode: int
    stage: str
    attempt: str
    file: str
    freqs: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    status: str = 'unknown'
    accepted: object = None


@dataclass
class Dataset:
    path: str
    label: str
    color: str
    enabled: bool = True
    traces: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)
    fit_results: dict = field(default_factory=dict)


def read_arrays(path):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            f = np.asarray(data['freqs'], dtype=float).copy()
            r = np.asarray(data['R'], dtype=float).copy()
            p = np.asarray(data['theta'], dtype=float).copy() if 'theta' in data else np.full(f.shape, np.nan)
    else:
        lines = [line for line in path.read_text(encoding='utf-8-sig').splitlines() if not line.startswith('#')]
        rows = list(csv.DictReader(io.StringIO('\n'.join(lines))))
        f = np.array([float(row['Frequency_Hz']) for row in rows])
        r = np.array([float(row['Amplitude_R_counts']) for row in rows])
        p = np.array([float(row.get('Phase_deg') or 'nan') for row in rows])
    if f.ndim != 1 or r.shape != f.shape or p.shape != f.shape:
        raise ValueError('Frequency, amplitude and phase must be matching one-dimensional arrays')
    mask = np.isfinite(f) & np.isfinite(r)
    order = np.argsort(f[mask], kind='stable')
    f, r, p = f[mask][order], r[mask][order], p[mask][order]
    if len(f) < 2:
        raise ValueError('Fewer than two valid samples')
    return f, r, p


def load_dataset(path, index=0):
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise ValueError(f'Folder not found: {directory}')
    dataset = Dataset(str(directory), default_label(directory, index + 1), COLORS[index % len(COLORS)])
    records = {}
    session_file = directory / 'session.json'
    if session_file.exists():
        try:
            records = {int(r['mode']): r for r in json.loads(session_file.read_text(encoding='utf-8-sig'))['records']}
        except (ValueError, KeyError, TypeError) as exc:
            dataset.warnings.append(f'Session metadata could not be read: {exc}')
    for mode, record in records.items():
        try:
            q = float(record.get('q'))
            if not np.isfinite(q) or q <= 0:
                q = None
        except (TypeError, ValueError):
            q = None
        dataset.fit_results[mode] = {'q': q, 'accepted': record.get('fit_accepted') is True}
    grouped = {}
    for path in sorted(directory.iterdir()):
        match = FILE_RE.match(path.name)
        if match and path.is_file():
            mode, stage, retry, extension = match.groups()
            grouped.setdefault((int(mode), stage, 'retry' if retry else 'original'), []).append(path)
    for (mode, stage, attempt), paths in sorted(grouped.items()):
        record = records.get(mode, {})
        stage_name = stage + ('_retry' if attempt == 'retry' else '')
        metadata = next((s for s in record.get('stages', []) if s.get('stage') == stage_name), {})
        status = record.get('status', 'unknown') if stage == 'fit' else metadata.get('status', 'unknown')
        for path in sorted(paths, key=lambda p: p.suffix != '.npz'):
            try:
                # Files are discovered inside the selected directory, never followed from metadata.
                if path.resolve().parent != directory:
                    raise ValueError('Trace file points outside selected folder')
                f, r, p = read_arrays(path)
                dataset.traces.append(Trace(mode, stage, attempt, path.name, f, r, p, status,
                                            record.get('fit_accepted') if stage == 'fit' else None))
                break
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                dataset.warnings.append(f'{path.name}: {exc}')
    if not dataset.traces:
        raise ValueError(f'No readable mode_XX_coarse/fine/fit data in {directory.name}')
    for path in directory.iterdir():
        if path.is_file() and (FILE_RE.match(path.name) or path.name == 'session.json'):
            stat = path.stat()
            dataset.fingerprint[path.name] = [stat.st_size, stat.st_mtime_ns]
    return dataset


def select_trace(dataset, mode, stage, attempt='auto', include_review=False):
    traces = [t for t in dataset.traces if t.mode == mode and t.stage == stage]
    if stage == 'fit':
        return next((t for t in traces if include_review or t.accepted is True), None)
    if attempt != 'auto':
        traces = [t for t in traces if t.attempt == attempt]
    traces = [t for t in traces if include_review or t.status == 'complete']
    return next(iter(sorted(traces, key=lambda t: t.attempt != 'retry')), None)


@dataclass
class PlotSpec:
    key: str
    kind: str
    stage: str
    mode: object
    datasets: list
    quantity: str
    title: str
    selected: bool = True
    reviewed: bool = False
    xmin: object = None
    xmax: object = None
    ymin: object = None
    ymax: object = None
    legend: str = 'best'
    log_y: bool = False
    normalized: bool = False
    show_q: bool = True
    q_position: str = 'legend'


@dataclass
class Style:
    font: str = 'Arial'
    font_size: float = 10
    line_width: float = 1.2
    square_mm: float = 90
    wide_mm: float = 180
    height_mm: float = 90
    unit: str = 'Hz'
    dpi: int = 600


def build_specs(datasets, stages, kinds, quantities, attempt='auto', include_review=False, previous=()):
    prior = {s.key: s for s in previous}
    enabled = [d for d in datasets if d.enabled]
    modes = sorted({t.mode for d in enabled for t in d.traces})
    result = []
    for stage in stages:
        for quantity in quantities:
            for kind in kinds:
                groups = []
                if kind == 'individual':
                    groups = [(m, [d.path]) for d in enabled for m in modes
                              if select_trace(d, m, stage, attempt, include_review)]
                elif kind == 'comparison':
                    groups = [(m, [d.path for d in enabled if select_trace(d, m, stage, attempt, include_review)]) for m in modes]
                elif kind == 'overview':
                    groups = [(None, [d.path for d in enabled if any(select_trace(d, m, stage, attempt, include_review) for m in modes)])]
                for mode, paths in groups:
                    if not paths:
                        continue
                    key = json.dumps([kind, stage, mode, quantity, paths if kind == 'individual' else []])
                    title = f'{ordinal(mode)} mode' if mode is not None else 'All modes'
                    spec = PlotSpec(key, kind, stage, mode, paths, quantity, title)
                    if key in prior:
                        spec = PlotSpec(**{**asdict(prior[key]), 'datasets': paths})
                    result.append(spec)
    return result


def plot_traces(spec, datasets, attempt, include_review):
    for dataset in datasets:
        if not dataset.enabled or dataset.path not in spec.datasets:
            continue
        modes = [spec.mode] if spec.mode is not None else sorted({t.mode for t in dataset.traces})
        for mode in modes:
            trace = select_trace(dataset, mode, spec.stage, attempt, include_review)
            if trace is not None:
                yield dataset, trace


def validate_spec(spec, datasets, style):
    if not english_text(spec.title):
        raise ValueError('Use English text for the plot title.')
    for d in datasets:
        if d.path in spec.datasets and d.enabled and (not d.label.strip() or not english_text(d.label)):
            raise ValueError(f'Use a nonempty English legend for {Path(d.path).name}.')
    for low, high, name in [(spec.xmin, spec.xmax, 'X'), (spec.ymin, spec.ymax, 'Y')]:
        if any(v is not None and not np.isfinite(v) for v in [low, high]):
            raise ValueError(f'{name} limits must be finite.')
        if low is not None and high is not None and low >= high:
            raise ValueError(f'{name} minimum must be below maximum.')
    if spec.log_y and any(v is not None and v <= 0 for v in [spec.ymin, spec.ymax]):
        raise ValueError('Logarithmic Y limits must be positive.')
    if spec.quantity == 'phase' and (spec.log_y or spec.normalized):
        raise ValueError('Phase plots use linear, unnormalized values.')
    if style.unit not in ('Hz', 'kHz'):
        raise ValueError('Frequency unit must be Hz or kHz.')
    if spec.q_position not in ('legend', 'upper left', 'upper right', 'lower left', 'lower right'):
        raise ValueError('Invalid Q position.')


def q_value_text(dataset, mode, include_review=False):
    fit = dataset.fit_results.get(mode, {})
    value = fit.get('q')
    if value is None or (not fit.get('accepted') and not include_review):
        return 'N/A'
    return f"{value:.1f}" + ('' if fit.get('accepted') else '*')


def q_table_data(spec, datasets, attempt='auto', include_review=False):
    pairs = list(plot_traces(spec, datasets, attempt, include_review))
    modes = sorted({t.mode for _, t in pairs})
    present = {(d.path, t.mode) for d, t in pairs}
    columns = [d for d in datasets if any(d.path == p for p, _ in present)]
    headers = ['Mode'] + [d.label for d in columns]
    rows = [[ordinal(mode)] + [q_value_text(d, mode, include_review) if (d.path, mode) in present else '--'
                               for d in columns] for mode in modes]
    return headers, rows, columns


def q_table_layout(spec, datasets, style, attempt='auto', include_review=False):
    headers, rows, columns = q_table_data(spec, datasets, attempt, include_review)
    chars = max(10, int((style.wide_mm * 72 / 25.4 / len(headers)) / (style.font_size * .52)))
    headers = ['\n'.join(textwrap.wrap(h, chars)) for h in headers]
    header_lines = max(h.count('\n') + 1 for h in headers)
    # Reserve physical space for table, title, and status footnote at the shared font size.
    height_mm = (len(rows) + header_lines + 3) * style.font_size * 1.45 * 25.4 / 72
    return headers, rows, columns, header_lines, height_mm


def figure_size_mm(spec, datasets, style, attempt='auto', include_review=False):
    if spec.kind != 'overview':
        return style.square_mm, style.square_mm
    height = style.height_mm
    if spec.show_q:
        height = max(height, 48 + q_table_layout(spec, datasets, style, attempt, include_review)[-1])
    return style.wide_mm, height


def render_figure(spec, datasets, style, attempt='auto', include_review=False, figure=None):
    validate_spec(spec, datasets, style)
    size = figure_size_mm(spec, datasets, style, attempt, include_review)
    fig = figure if figure is not None else Figure()
    if figure is None:
        FigureCanvasAgg(fig)
    fig.clear()
    fig.set_size_inches(*(s / 25.4 for s in size), forward=False)
    fig.set_layout_engine('constrained')
    settings = {'font.family': style.font, 'font.size': style.font_size,
                'axes.titlesize': style.font_size, 'axes.labelsize': style.font_size,
                'xtick.labelsize': style.font_size, 'ytick.labelsize': style.font_size,
                'legend.fontsize': style.font_size, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                'svg.fonttype': 'none', 'axes.linewidth': .8}
    with rc_context(settings):
        table_ax = None
        if spec.kind == 'overview' and spec.show_q:
            headers, rows, columns, header_lines, table_height = q_table_layout(spec, datasets, style, attempt, include_review)
            grid = fig.add_gridspec(2, 1, height_ratios=[size[1] - table_height, table_height])
            ax = fig.add_subplot(grid[0])
            table_ax = fig.add_subplot(grid[1])
        else:
            ax = fig.add_subplot(111)
        seen = set()
        q_lines = []
        plotted = 0
        for dataset, trace in plot_traces(spec, datasets, attempt, include_review):
            values = trace.amplitude.copy() if spec.quantity == 'amplitude' else trace.phase.copy()
            if spec.normalized:
                maximum = np.nanmax(np.abs(values))
                values /= maximum if maximum else 1
            if spec.log_y:
                values[values <= 0] = np.nan
            # Do not connect phase wrap discontinuities or unmeasured modal intervals.
            if spec.quantity == 'phase':
                values[1:][np.abs(np.diff(values)) > 180] = np.nan
            if not np.isfinite(values).any():
                continue
            label = dataset.label if dataset.path not in seen else '_nolegend_'
            if spec.show_q and spec.kind != 'overview':
                q_text = q_value_text(dataset, trace.mode, include_review)
                q_label = 'Q' if spec.stage == 'fit' else 'Q (fit)'
                if q_text.endswith('*'):
                    q_text = q_text[:-1] + ' (review)'
                if spec.q_position == 'legend' and spec.legend != 'none':
                    label += f'\n{q_label} = {q_text}'
                else:
                    q_lines.append(TextArea(f'{dataset.label}\n{q_label} = {q_text}',
                                           textprops={'color': dataset.color, 'fontsize': style.font_size}))
            seen.add(dataset.path)
            ax.plot(trace.freqs / (1000 if style.unit == 'kHz' else 1), values,
                    color=dataset.color, lw=style.line_width, label=label)
            plotted += 1
        if not plotted:
            raise ValueError('No valid data for this plot and the current selection.')
        ax.set_title(spec.title)
        ax.set_xlabel(f'Frequency ({style.unit})')
        ax.set_ylabel('Phase (deg)' if spec.quantity == 'phase' else
                      ('Normalized amplitude' if spec.normalized else 'Amplitude (counts)'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', width=.8)
        if spec.log_y:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        # Stored X limits are always Hz, independent of displayed unit.
        factor = 1000 if style.unit == 'kHz' else 1
        ax.set_xlim(left=spec.xmin / factor if spec.xmin is not None else None,
                    right=spec.xmax / factor if spec.xmax is not None else None)
        ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
        if spec.legend != 'none':
            if spec.legend == 'outside':
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), frameon=False)
            else:
                ax.legend(loc=spec.legend, frameon=False)
        if spec.kind != 'overview':
            ax.set_box_aspect(1)
        if q_lines:
            box = AnchoredOffsetbox(loc='upper right' if spec.q_position == 'legend' else spec.q_position,
                    child=VPacker(children=q_lines, align='left', pad=0, sep=4),
                    frameon=True, pad=.3, borderpad=.5)
            box.patch.set(facecolor='white', edgecolor='none', alpha=.85)
            ax.add_artist(box)
        if table_ax is not None:
            table_ax.set_axis_off()
            table_ax.set_title('Q (fit)', pad=2)
            table = table_ax.table(cellText=rows, colLabels=headers, cellLoc='center', bbox=[0, .20, 1, .76])
            table.auto_set_font_size(False)
            table.set_fontsize(style.font_size)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#dddddd')
                cell.set_linewidth(.4)
                cell.set_height((header_lines if row == 0 else 1) / (len(rows) + header_lines))
                cell.set_facecolor('#f3f3f3' if row == 0 else 'white')
                if col > 0:
                    cell.get_text().set_color(columns[col - 1].color)
            footnote = 'N/A: no accepted fit; --: no trace'
            if include_review:
                footnote = 'N/A: no fitted Q; --: no trace; *: fit needs review'
            table_ax.text(.5, 0, footnote, transform=table_ax.transAxes, ha='center', va='bottom', fontsize=style.font_size)
    return fig


def save_figure(fig, path, style):
    # Keep physical dimensions and font sizes identical across export formats.
    with rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        fig.savefig(path, dpi=style.dpi, facecolor='white')


def project_payload(datasets, specs, style, options):
    return {'version': 1, 'datasets': [{k: getattr(d, k) for k in ('path', 'label', 'color', 'enabled', 'fingerprint')} for d in datasets],
            'plots': [asdict(s) for s in specs], 'style': asdict(style), 'options': options}


def write_project(path, datasets, specs, style, options):
    path = Path(path)
    if any(path.resolve().parent == Path(d.path).resolve() for d in datasets):
        if path.name in ('session.json', 'settings.json', 'summary.csv') or FILE_RE.match(path.name):
            raise ValueError('Choose a project filename that does not overwrite measurement data.')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(project_payload(datasets, specs, style, options), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
