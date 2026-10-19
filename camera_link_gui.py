"""Optional GUI adapter; camera failures never enter instrument control paths."""
import math
import uuid
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QSpinBox

from camera_link import CameraLink, DirectionHistory


class CameraLinkMixin:
    def _init_camera_link(self):
        self._camera_link = None
        self._camera_history = DirectionHistory()
        self._camera_session = uuid.uuid4().hex

    def _build_camera_link_group(self, state_dir):
        self._camera_state_dir = state_dir
        box = QGroupBox('카메라 연동 — 독립 프로그램 / 선택 기능')
        layout = QGridLayout(box)
        self.camera_enabled = QCheckBox('검출 이벤트를 카메라로 전송 (기본 OFF)')
        self.camera_port = QSpinBox()
        self.camera_port.setRange(1024, 65535)
        self.camera_port.setValue(5088)
        self.camera_mapping = QComboBox()
        self.camera_mapping.addItems(['좌우 배선 선택 필요', 'A / AO0 = 좌측, B / AO1 = 우측',
                                      'A / AO0 = 우측, B / AO1 = 좌측'])
        self.camera_guard = QDoubleSpinBox()
        self.camera_guard.setRange(0, 10)
        self.camera_guard.setDecimals(3)
        self.camera_guard.setValue(.25)
        self.camera_guard.setSuffix(' s')
        self.camera_guard.setToolTip('입자 구간 전후 이 시간 안에 압력 명령 변경이 있으면 카메라 전송만 생략합니다. 실제 유동 안정화 시간으로 보정하세요.')
        self.camera_status = QLabel('OFF — 기존 측정과 독립')
        self.camera_counts = QLabel('전송 0 / 수락 0 / 저장 0 / 거절·실패 0 / 누락 0 / 미확인 0')
        self.camera_counts.setWordWrap(True)
        self.camera_recent = QLabel('')
        self.camera_recent.setWordWrap(True)
        layout.addWidget(self.camera_enabled, 0, 0, 1, 2)
        layout.addWidget(QLabel('127.0.0.1 포트'), 1, 0)
        layout.addWidget(self.camera_port, 1, 1)
        layout.addWidget(self.camera_mapping, 2, 0, 1, 2)
        layout.addWidget(QLabel('압력 전환 제외 여유'), 3, 0)
        layout.addWidget(self.camera_guard, 3, 1)
        layout.addWidget(self.camera_status, 4, 0, 1, 2)
        layout.addWidget(self.camera_counts, 5, 0, 1, 2)
        layout.addWidget(self.camera_recent, 6, 0, 1, 2)
        note = QLabel('실시간 파티클 검출만 전송합니다. Ping-Pong 반전은 변경하지 않습니다.\n카메라 미연결·저장 중 검출은 누락으로 기록하고 나중에 재전송하지 않습니다.')
        note.setWordWrap(True)
        layout.addWidget(note, 7, 0, 1, 2)
        self.camera_enabled.toggled.connect(self._toggle_camera_link)
        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(250)
        self.camera_timer.timeout.connect(self._refresh_camera_link)
        return box

    def _toggle_camera_link(self, enabled):
        if self._camera_link:
            self._camera_link.close()
            self._camera_link = None
        if enabled and self.camera_mapping.currentIndex() == 0:
            self.camera_enabled.blockSignals(True)
            self.camera_enabled.setChecked(False)
            self.camera_enabled.blockSignals(False)
            self.camera_status.setText('OFF — 먼저 A/B의 실제 좌우 배선을 선택하세요.')
            return
        for widget in (self.camera_port, self.camera_mapping, self.camera_guard):
            widget.setEnabled(not enabled)
        if enabled:
            try:
                log = self._camera_state_dir / 'camera_link' / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex + '.jsonl')
                self._camera_history = DirectionHistory()
                self._camera_session = uuid.uuid4().hex
                self._camera_link = CameraLink(self.camera_port.value(), log_path=log)
                self.camera_status.setToolTip(str(log))
                self.camera_timer.start()
            except Exception as exc:
                self.camera_status.setText('카메라 연동 시작 실패: ' + str(exc))
                self.camera_enabled.setChecked(False)
        else:
            self.camera_timer.stop()
            self.camera_status.setText('OFF — 기존 측정과 독립')

    def _camera_new_measurement(self):
        self._camera_history = DirectionHistory()
        self._camera_session = uuid.uuid4().hex

    def _camera_observe_sample(self, t):
        if self._camera_link is None:
            return
        try:
            values = tuple(self.pressure_commands.values())
            self._camera_history.observe(t, values[0], values[1])
        except Exception:
            # An optional observer must never interrupt raw recording or pressure control.
            pass

    def _send_camera_events(self, events):
        link = self._camera_link
        if link is None:
            return
        for event in events:
            event_id = uuid.uuid4().hex
            try:
                t = float(event['time_s'])
                duration = float(event.get('duration_s', 0))
                direction = self._camera_history.resolve(t, duration / 2, self.camera_guard.value(),
                                                         self.camera_mapping.currentIndex() == 1)
                if direction is None:
                    link.miss('DIRECTION_UNCERTAIN', session_id=self._camera_session, event_id=event_id, event_time_s=t)
                    continue
                message = dict(type='PARTICLE_DETECTED', session_id=self._camera_session,
                               event_id=event_id, event_time_s=t, source='live_particle_analysis',
                               detected_at_local=event.get('detected_at_local'), **direction)
                for key in ('depth_hz', 'duration_s', 'frequency_hz'):
                    value = float(event.get(key, float('nan')))
                    message[key] = value if math.isfinite(value) else None
                link.submit(message)
            except Exception:
                # No socket operation or exception may propagate into the detector callback.
                link.miss('CAMERA_ADAPTER_ERROR', event_id=event_id)

    def _refresh_camera_link(self):
        if self._camera_link is None:
            return
        state = self._camera_link.snapshot()
        suffix = ' — 촬영 가능' if state['ready'] else ' — 측정은 계속됩니다'
        self.camera_status.setText(state['state'] + suffix)
        self.camera_counts.setText('전송 {sent} / 수락 {accepted} / 저장 {saved} / 거절·실패 {rejected} / 누락 {missed} / 미확인 {unknown}'.format(**state))
        recent = state['recent']
        self.camera_recent.setText(('최근: ' + str(recent[-1]))[:600] if recent else '')
        if state['log_error']:
            self.camera_recent.setText('카메라 로그 기록 실패: ' + state['log_error'])
        elif state.get('log_dropped', 0):
            self.camera_recent.setText('카메라 상세 로그 누락: ' + str(state['log_dropped']) + '건 (누적 개수는 유지)')

    def _close_camera_link(self):
        self.camera_timer.stop()
        if self._camera_link:
            self._camera_link.close()  # Daemon exits independently; no instrument shutdown wait.
            self._camera_link = None
