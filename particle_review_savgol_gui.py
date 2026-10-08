"""Interactive Savitzky-Golay preview for offline particle traces."""
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.signal import savgol_filter


class SavgolPanel(QtWidgets.QWidget):
    """Compare raw frequency data with an adjustable SG smoothing result."""
    noiseRequested = QtCore.pyqtSignal()
    mitRequested = QtCore.pyqtSignal()
    propertiesRequested = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.events = []
        self.filtered = None
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout(); layout.addLayout(controls)
        controls.addWidget(QtWidgets.QLabel('후보'))
        self.event = QtWidgets.QComboBox(); controls.addWidget(self.event, 1)
        controls.addWidget(QtWidgets.QLabel('Polynomial order'))
        self.order = QtWidgets.QSpinBox(); self.order.setRange(1, 8); self.order.setValue(4); controls.addWidget(self.order)
        controls.addWidget(QtWidgets.QLabel('Window (ms)'))
        self.window = QtWidgets.QDoubleSpinBox(); self.window.setDecimals(3); self.window.setRange(.001, 60000); self.window.setSingleStep(1); self.window.setValue(9); controls.addWidget(self.window)
        self.window.setToolTip('입력한 ms를 샘플링 속도에 맞춰 가장 가까운 홀수 표본 수로 변환합니다.')
        controls.addWidget(QtWidgets.QLabel('표시 폭 (s)'))
        self.span = QtWidgets.QDoubleSpinBox(); self.span.setRange(.05, 120); self.span.setDecimals(3); self.span.setValue(.45); controls.addWidget(self.span)

        self.info = QtWidgets.QLabel('원시 CSV를 열면 Savitzky-Golay 비교가 표시됩니다.')
        self.info.setWordWrap(True); layout.addWidget(self.info)

        self.graph = pg.GraphicsLayoutWidget(); layout.addWidget(self.graph, 1)
        self.overlay = self.graph.addPlot(row=0, col=0)
        self.overlay.setLabel('left', 'Frequency', units='Hz')
        self.residual = self.graph.addPlot(row=1, col=0)
        self.residual.setLabel('left', 'Raw - SG', units='Hz')
        self.residual.setLabel('bottom', 'Time', units='s')
        self.residual.setXLink(self.overlay)
        for plot in (self.overlay, self.residual):
            plot.showGrid(x=True, y=True, alpha=.2)
            plot.getAxis('left').enableAutoSIPrefix(False)
            plot.getAxis('left').setWidth(105)

        note = QtWidgets.QLabel(
            '회색=원시 주파수 · 주황=Savitzky-Golay 평활 · 초록=원시−평활 residual · 파란 음영=저장된 후보 구간\n'
            '이 탭은 화면 비교 전용이며 원본 데이터와 후보 깊이 계산을 변경하지 않습니다. 논문식 baseline noise σ에는 SG 결과를 사용하지 마세요.'
        )
        note.setWordWrap(True); layout.addWidget(note)
        actions = QtWidgets.QHBoxLayout(); layout.addLayout(actions)
        self.noise_btn = QtWidgets.QPushButton('현재 SG 신호로 기존 노이즈 / SNR 계산')
        self.mit_btn = QtWidgets.QPushButton('현재 SG 설정으로 MIT 피크 분석')
        self.properties_btn = QtWidgets.QPushButton('현재 SG Δf로 왕복쌍 / 물성 계산')
        self.noise_btn.clicked.connect(self.noiseRequested); self.mit_btn.clicked.connect(self.mitRequested)
        self.properties_btn.clicked.connect(self.propertiesRequested)
        actions.addWidget(self.noise_btn); actions.addWidget(self.properties_btn); actions.addWidget(self.mit_btn); actions.addStretch(1)

        self.event.currentIndexChanged.connect(self.redraw)
        self.order.valueChanged.connect(self.redraw)
        self.window.valueChanged.connect(self._window_changed)
        self.span.valueChanged.connect(self.redraw)

    def _window_changed(self, value):
        self.redraw()

    def _sampling_rate(self):
        if self.data is None or len(self.data)<2:
            raise ValueError('샘플링 속도 계산에 원시 데이터가 필요합니다.')
        dt=np.diff(self.data.Time_s.to_numpy(dtype=float))
        if not np.all(np.isfinite(dt)) or np.any(dt<=0):
            raise ValueError('Time_s는 유한하고 증가해야 합니다.')
        median=float(np.median(dt))
        if np.max(np.abs(dt-median))>.05*median:
            raise ValueError('표본 간격 변동이 5%를 넘어 SG 필터를 적용할 수 없습니다.')
        return 1/median

    def _window_samples(self, fs):
        # Nearest odd sample count; exact ties round upward.
        samples=2*int(np.floor(float(self.window.value())*fs/1000/2+1e-9))+1
        minimum=int(self.order.value())+1
        if minimum%2==0:minimum+=1
        return max(3,minimum,samples)

    def set_source(self, data, events, selected_index=None):
        self.data = data
        self.events = list(events or [])
        old = self.event.currentIndex()
        self.event.blockSignals(True); self.event.clear()
        for i, item in enumerate(self.events):
            self.event.addItem(f"{i + 1}번 · {float(item['time_s']):.4f} s", i)
        if self.events:
            target = selected_index if selected_index is not None and 0 <= selected_index < len(self.events) else old
            self.event.setCurrentIndex(target if 0 <= target < len(self.events) else 0)
        self.event.blockSignals(False)
        self.redraw()

    def set_selected_event(self, index):
        if 0 <= index < self.event.count() and index != self.event.currentIndex():
            self.event.setCurrentIndex(index)

    def get_settings(self):
        fs=self._sampling_rate(); samples=self._window_samples(fs)
        if samples>len(self.data):
            raise ValueError('Window 시간이 데이터 길이보다 큽니다.')
        return dict(order=int(self.order.value()), window_ms=float(self.window.value()),
                    window_length=samples, applied_window_ms=1000*samples/fs,
                    sampling_rate_hz=fs, span_s=float(self.span.value()))

    def filtered_data(self):
        """Return a data-frame copy whose frequency column is the current SG result."""
        self.redraw()
        if self.data is None or self.filtered is None:
            raise ValueError('유효한 Savitzky-Golay 설정과 원시 데이터가 필요합니다.')
        result = self.data.copy()
        result['Frequency_Hz'] = self.filtered
        return result

    def set_settings(self, settings):
        if not settings:
            return
        self.order.setValue(int(settings.get('order', 4)))
        if 'window_ms' in settings:
            window_ms=float(settings['window_ms'])
        else:
            # Migrate older saved reviews that stored a sample count.
            fs=self._sampling_rate()
            window_ms=1000*int(settings.get('window_length',27))/fs
        self.window.setValue(window_ms)
        self.span.setValue(float(settings.get('span_s', .45)))
        self.redraw()

    def _center_and_bounds(self):
        t = self.data.Time_s.to_numpy(dtype=float)
        index = self.event.currentIndex()
        if 0 <= index < len(self.events):
            item = self.events[index]
            center = float(item['time_s'])
            start = float(item.get('event_start_s', center))
            end = float(item.get('event_end_s', center))
            if not np.isfinite([start, end]).all() or end <= start:
                start = end = center
            return center, start, end
        center = float((t[0] + t[-1]) / 2)
        return center, center, center

    def redraw(self, *args):
        self.overlay.clear(); self.residual.clear(); self.filtered = None
        if self.data is None or not len(self.data):
            self.info.setText('원시 CSV를 열면 Savitzky-Golay 비교가 표시됩니다.')
            return
        order = int(self.order.value())
        try:
            fs=self._sampling_rate(); window=self._window_samples(fs)
        except ValueError as exc:
            self.info.setText(str(exc)); return
        if window > len(self.data):
            self.info.setText('Window 시간이 데이터 길이보다 큽니다. 더 작은 ms 값을 입력하세요.')
            return
        t = self.data.Time_s.to_numpy(dtype=float)
        y = self.data.Frequency_Hz.to_numpy(dtype=float)
        try:
            self.filtered = savgol_filter(y, window_length=window, polyorder=order, mode='interp')
        except ValueError as exc:
            self.info.setText(str(exc)); return

        center, event_start, event_end = self._center_and_bounds()
        half = float(self.span.value()) / 2
        lo, hi = max(float(t[0]), center - half), min(float(t[-1]), center + half)
        mask = (t >= lo) & (t <= hi)
        if not np.any(mask):
            self.info.setText('현재 표시 범위에 표본이 없습니다.'); return
        region = None
        if event_end > event_start:
            region = [max(lo, event_start), min(hi, event_end)]
            if region[1] <= region[0]:
                region = None
        for plot in (self.overlay, self.residual):
            if region:
                plot.addItem(pg.LinearRegionItem(region, movable=False, brush=(40, 115, 190, 35), pen=None))
        self.overlay.plot(t[mask], y[mask], pen=pg.mkPen('#a5a5a5', width=1))
        self.overlay.plot(t[mask], self.filtered[mask], pen=pg.mkPen('#df7d17', width=2))
        residual = y - self.filtered
        self.residual.plot(t[mask], residual[mask], pen=pg.mkPen('#238b57', width=1))
        self.residual.addLine(y=0, pen=pg.mkPen('#606060', style=QtCore.Qt.DashLine))
        self.overlay.setXRange(lo, hi, padding=0)
        self.overlay.enableAutoRange(axis='y'); self.residual.enableAutoRange(axis='y')

        shown_residual = residual[mask]
        raw_pp = float(np.ptp(y[mask])); smooth_pp = float(np.ptp(self.filtered[mask]))
        self.info.setText(
            f'Fs {fs:,.2f} Hz · order {order} · 요청 {self.window.value():.3f} ms → 적용 {1000 * window / fs:.3f} ms ({window}표본) · 표시 {np.count_nonzero(mask):,}표본\n'
            f'현재 화면 raw−SG SD {np.std(shown_residual, ddof=1):.5f} Hz · peak-to-peak raw/SG {raw_pp:.5f}/{smooth_pp:.5f} Hz'
        )
