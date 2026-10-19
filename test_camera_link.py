import json
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from pathlib import Path

from camera_link import CameraLink, DirectionHistory


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for camera IPC')
        time.sleep(.01)


class CameraServer:
    def __init__(self, behavior='save'):
        self.received = []
        self.stop = threading.Event()
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(.1)

                def send(message):
                    self.request.sendall((json.dumps(message) + '\n').encode())

                try:
                    if behavior != 'silent':
                        send(dict(type='STATUS', state='READY', ready=True))
                    data = b''
                    while not owner.stop.is_set():
                        try:
                            chunk = self.request.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            return
                        data += chunk
                        while b'\n' in data:
                            line, data = data.split(b'\n', 1)
                            msg = json.loads(line)
                            if msg['type'] == 'HEARTBEAT':
                                if behavior != 'silent':
                                    send(dict(type='STATUS', state='READY', ready=True))
                                continue
                            owner.received.append(msg)
                            if behavior == 'noack':
                                continue
                            if behavior == 'disconnect':
                                return
                            identity = {k: msg[k] for k in ('session_id', 'event_id')}
                            if behavior == 'reject':
                                send(dict(identity, type='REJECTED', reason='SAVING'))
                            else:
                                send(dict(identity, type='ACCEPTED'))
                                if behavior != 'hold':
                                    send(dict(identity, type='SAVED', file='test.cine'))
                except OSError:
                    pass

        self.server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)


class DirectionTests(unittest.TestCase):
    def test_historical_direction_and_switch_guard(self):
        h = DirectionHistory()
        h.observe(0, 2, 1)
        h.observe(4, 1, 2)
        h.observe(10, 1, 2)
        self.assertEqual(h.resolve(2, .05, .25, True)['direction'], 'LEFT_TO_RIGHT')
        self.assertEqual(h.resolve(6, .05, .25, True)['direction'], 'RIGHT_TO_LEFT')
        self.assertEqual(h.resolve(2, .05, .25, False)['direction'], 'RIGHT_TO_LEFT')
        self.assertIsNone(h.resolve(3.9, .05, .25, True))
        self.assertIsNone(h.resolve(.1, .05, .25, True))
        self.assertIsNone(h.resolve(10, .05, .25, True))

    def test_equal_pressure_and_bounded_history(self):
        h = DirectionHistory()
        h.observe(0, 1, 1)
        h.observe(10, 1, 1)
        self.assertIsNone(h.resolve(5, .1, .25, True))
        for i in range(10000):
            h.observe(i / 10000, i % 2, 1)
        self.assertLessEqual(len(h.changes), 4096)


class ClientTests(unittest.TestCase):
    def client(self, behavior):
        server = CameraServer(behavior)
        self.addCleanup(server.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        link = CameraLink(server.port, Path(tmp.name) / 'camera.jsonl')
        self.addCleanup(lambda: link.close(wait=True))
        return server, link

    @staticmethod
    def msg(event_id='1'):
        return dict(type='PARTICLE_DETECTED', session_id='test', event_id=event_id, direction='LEFT_TO_RIGHT')

    def test_success_and_only_own_results(self):
        server, link = self.client('save')
        wait_for(lambda: link.snapshot()['ready'])
        self.assertTrue(link.submit(self.msg()))
        wait_for(lambda: link.snapshot()['saved'] == 1)
        self.assertEqual(link.snapshot()['sent'], 1)
        self.assertEqual(link.snapshot()['accepted'], 1)
        self.assertEqual(len(server.received), 1)
        link._handle(dict(type='SAVED', session_id='somebody-else', event_id='1'), {})
        self.assertEqual(link.snapshot()['saved'], 1)

    def test_disconnect_does_not_resend(self):
        server, link = self.client('disconnect')
        wait_for(lambda: link.snapshot()['ready'])
        link.submit(self.msg())
        wait_for(lambda: link.snapshot()['unknown'] == 1)
        wait_for(lambda: link.snapshot()['ready'])
        time.sleep(.2)
        self.assertEqual(len(server.received), 1)

    def test_busy_rejection_and_pending_blocks_stale_ready(self):
        server, link = self.client('hold')
        wait_for(lambda: link.snapshot()['ready'])
        link.submit(self.msg())
        wait_for(lambda: link.snapshot()['accepted'] == 1)
        time.sleep(.6)  # Server keeps sending stale READY, but event remains in flight.
        self.assertFalse(link.snapshot()['ready'])
        self.assertFalse(link.submit(self.msg('2')))
        self.assertEqual(len(server.received), 1)

    def test_rejected_response_is_recorded(self):
        _, link = self.client('reject')
        wait_for(lambda: link.snapshot()['ready'])
        link.submit(self.msg())
        wait_for(lambda: link.snapshot()['rejected'] == 1)

    def test_camera_absent_submission_never_connects_or_waits(self):
        calls = []

        def absent(*args, **kwargs):
            calls.append(threading.get_ident())
            raise ConnectionRefusedError()

        link = CameraLink(connector=absent)
        self.addCleanup(lambda: link.close(wait=True))
        start = time.monotonic()
        for i in range(1000):
            link.submit(self.msg(str(i)))
        self.assertLess(time.monotonic() - start, .5)
        self.assertEqual(link.snapshot()['missed'], 1000)
        self.assertTrue(all(i != threading.get_ident() for i in calls))
        self.assertLessEqual(len(link.audit), 512)

    def test_silent_server_never_ready_and_close_is_bounded(self):
        _, link = self.client('silent')
        wait_for(lambda: link.snapshot()['connected'])
        self.assertFalse(link.submit(self.msg()))
        start = time.monotonic()
        link.close(wait=True)
        self.assertFalse(link.thread.is_alive())
        self.assertLess(time.monotonic() - start, 1)

    def test_heartbeat_without_acceptance_times_out_without_replay(self):
        server, link = self.client('noack')
        link.ack_timeout = .3
        wait_for(lambda: link.snapshot()['ready'])
        link.submit(self.msg())
        wait_for(lambda: link.snapshot()['unknown'] == 1)
        wait_for(lambda: link.snapshot()['ready'])
        self.assertEqual(len(server.received), 1)


if __name__ == '__main__':
    unittest.main()
