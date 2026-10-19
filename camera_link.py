"""Optional camera IPC. No device control, Qt, or measurement dependencies."""
from bisect import bisect_right
from collections import deque
import json
import math
from pathlib import Path
import queue
import socket
import threading
import time


class DirectionHistory:
    """Bounded command history on the measurement sample time axis."""
    def __init__(self):
        self.changes = []
        self.latest_t = None

    def observe(self, t, a, b):
        item = (float(t), float(a), float(b))
        if not all(math.isfinite(v) for v in item):
            return
        self.latest_t = float(t)
        if self.changes and t < self.changes[-1][0]:
            self.changes.clear()
        if not self.changes or item[1:] != self.changes[-1][1:]:
            self.changes.append(item)
        while len(self.changes) > 2 and self.changes[1][0] < t - 60:
            del self.changes[0]
        if len(self.changes) > 4096:
            del self.changes[:-4096]

    def resolve(self, t, half_width, guard, a_is_left):
        if not all(math.isfinite(v) for v in (t, half_width, guard)):
            return None
        left, right = t - max(0, half_width) - guard, t + max(0, half_width) + guard
        if self.latest_t is None or right > self.latest_t:
            return None
        times = [item[0] for item in self.changes]
        index = bisect_right(times, left) - 1
        if index < 0 or index + 1 < len(times) and times[index + 1] <= right:
            return None
        _, a, b = self.changes[index]
        if abs(a - b) < 1e-6:
            return None
        lv, rv = (a, b) if a_is_left else (b, a)
        return dict(direction='LEFT_TO_RIGHT' if lv > rv else 'RIGHT_TO_LEFT',
                    left_voltage=lv, right_voltage=rv)


class CameraLink:
    """Nonblocking bounded submission; all socket and log I/O is on one daemon."""
    def __init__(self, port=5088, log_path=None, connector=socket.create_connection, ack_timeout=3):
        self.port = int(port)
        self.log_path = Path(log_path) if log_path else None
        self.connector = connector
        self.ack_timeout = ack_timeout
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.jobs = queue.Queue(maxsize=64)
        self.audit = deque(maxlen=512)
        self.recent = deque(maxlen=8)
        self.connected = False
        self.inflight = False
        self.state = 'DISCONNECTED'
        self.status_at = 0.
        self.counts = dict(sent=0, accepted=0, saved=0, rejected=0, missed=0, unknown=0, log_dropped=0)
        self.log_error = ''
        self.thread = threading.Thread(target=self._run, name='CameraLink', daemon=True)
        self.thread.start()

    def _note_locked(self, kind, **fields):
        record = dict(type=kind, host_monotonic_s=time.monotonic(), **fields)
        if len(self.audit) == self.audit.maxlen:
            self.counts['log_dropped'] += 1
        self.audit.append(record)
        self.recent.append(record)

    def _ready_locked(self):
        return self.connected and not self.inflight and self.state == 'READY' and time.monotonic() - self.status_at < 2

    def snapshot(self):
        with self.lock:
            return dict(state=self.state, ready=self._ready_locked(), connected=self.connected,
                        **self.counts, recent=list(self.recent), log_error=self.log_error)

    def miss(self, reason, **identity):
        with self.lock:
            self.counts['missed'] += 1
            self._note_locked('MISSED', reason=reason, **identity)

    def submit(self, message):
        """Never connect, wait for a response, or write files here."""
        with self.lock:
            if self.stop_event.is_set() or not self._ready_locked():
                self.counts['missed'] += 1
                self._note_locked('MISSED', reason=self.state, **self._identity(message))
                return False
            try:
                self.jobs.put_nowait((time.monotonic(), message.copy()))
            except queue.Full:
                self.counts['missed'] += 1
                self._note_locked('MISSED', reason='QUEUE_FULL', **self._identity(message))
                return False
        return True

    @staticmethod
    def _identity(message):
        return {key: message.get(key) for key in ('session_id', 'event_id', 'event_time_s', 'direction')}

    def close(self, wait=False):
        self.stop_event.set()
        if wait:
            self.thread.join(timeout=1)

    def _flush_log(self):
        with self.lock:
            records = list(self.audit)
            self.audit.clear()
        if records and self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a', encoding='utf-8') as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.log_error = str(exc)
                    self.counts['log_dropped'] += len(records)

    def _disconnect(self, pending):
        with self.lock:
            self.connected = False
            self.inflight = False
            self.state = 'DISCONNECTED'
            for key, value in pending.items():
                self.counts['unknown'] += 1
                self._note_locked('UNKNOWN', session_id=key[0], event_id=key[1], previous=value)
            pending.clear()
        while True:
            try:
                _, message = self.jobs.get_nowait()
                self.miss('DISCONNECTED', **self._identity(message))
            except queue.Empty:
                break

    def _handle(self, message, pending):
        if not isinstance(message, dict):
            raise ValueError('Camera response must be an object')
        kind = message.get('type')
        with self.lock:
            if kind == 'STATUS':
                state = message.get('state', 'UNKNOWN')
                self.state = 'READY' if state == 'READY' and message.get('ready') is True else str(state)
                if self.state == 'READY' and message.get('ready') is not True:
                    self.state = 'NOT_READY'
                self.status_at = time.monotonic()
                return
            key = (str(message.get('session_id')), str(message.get('event_id')))
            if key not in pending:
                return  # Other clients' broadcast results are not ours.
            if kind == 'ACCEPTED':
                if pending[key] == 'SENT':
                    self.counts['accepted'] += 1
                pending[key] = 'ACCEPTED'
            elif kind == 'SAVED':
                self.counts['saved'] += 1
                del pending[key]
            elif kind in ('REJECTED', 'FAILED'):
                self.counts['rejected'] += 1
                del pending[key]
            else:
                return
            self.inflight = bool(pending)
            self._note_locked(kind, response=message)

    def _run(self):
        pending = {}
        delay = .5
        try:
            while not self.stop_event.is_set():
                sock = None
                try:
                    sock = self.connector(('127.0.0.1', self.port), timeout=.2)
                    sock.settimeout(.1)
                    with self.lock:
                        self.connected = True
                        self.state = 'CONNECTING'
                    delay = .5
                    data = b''
                    heartbeat_at = status_deadline = time.monotonic()
                    status_deadline += 3
                    ack_deadline = None
                    while not self.stop_event.is_set():
                        self._flush_log()
                        now = time.monotonic()
                        if now >= heartbeat_at:
                            sock.sendall(b'{"type":"HEARTBEAT"}\n')
                            heartbeat_at = now + .5
                        if now > status_deadline:
                            raise TimeoutError('Camera status timeout')
                        if pending and ack_deadline is not None and now > ack_deadline and 'SENT' in pending.values():
                            raise TimeoutError('Camera acceptance timeout')
                        try:
                            queued_at, message = self.jobs.get_nowait()
                        except queue.Empty:
                            message = None
                        if message is not None:
                            with self.lock:
                                ready = self._ready_locked()
                            if not ready or now - queued_at > 1:
                                self.miss('BUSY_OR_STALE', **self._identity(message))
                            else:
                                key = (str(message['session_id']), str(message['event_id']))
                                payload = (json.dumps(message, allow_nan=False) + '\n').encode('utf-8')
                                pending[key] = 'SENT'  # Even a partial send is ambiguous; never retry.
                                ack_deadline = now + self.ack_timeout
                                with self.lock:
                                    self.state = 'WAITING'
                                    self.inflight = True
                                    self.counts['sent'] += 1
                                    self._note_locked('SENT', message=message)
                                sock.sendall(payload)
                        try:
                            chunk = sock.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            raise ConnectionError('Camera disconnected')
                        data += chunk
                        if len(data) > 65536:
                            raise ValueError('Camera response exceeds 64 KiB')
                        while b'\n' in data:
                            line, data = data.split(b'\n', 1)
                            response = json.loads(line)
                            self._handle(response, pending)
                            if response.get('type') == 'STATUS':
                                status_deadline = time.monotonic() + 3
                except (OSError, ValueError, KeyError, TypeError, RecursionError):
                    pass
                finally:
                    if sock:
                        sock.close()
                    self._disconnect(pending)
                    self._flush_log()
                # Flush missed-event logs while backing off, without touching the GUI.
                until = time.monotonic() + delay
                while not self.stop_event.wait(.1) and time.monotonic() < until:
                    self._flush_log()
                delay = min(5, delay * 2)
        finally:
            self._flush_log()
