"""TCP command and UDP multicast transports for the SMR controller."""
from __future__ import annotations

from dataclasses import dataclass
import re
import socket
import threading
import time
from typing import Callable

from fpga_stream import DecodedPacket, PacketSequenceTracker, SequenceResult, decode_packet


class FPGAProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class NetworkProfile:
    board_host: str = "192.168.100.2"
    tcp_port: int = 30
    multicast_group: str = "224.1.1.1"
    udp_port: int = 5007
    interface_host: str = "192.168.100.1"
    command_timeout_s: float = 1.5
    byte_order: str = "little"
    header_has_sequence: bool = False


class SMRTCPClient:
    WRITE_RE = re.compile(r"wrote:\s*([0-9a-f]+)\s+to PIO at\s+([0-9a-f]+)", re.I)
    READ_RE = re.compile(r"read:\s*([0-9a-f]+)\s+from PIO at\s+([0-9a-f]+)", re.I)

    def __init__(self, profile: NetworkProfile):
        self.profile = profile
        self.sock: socket.socket | None = None
        self.lock = threading.RLock()
        self.buffer = bytearray()
        self.banner = ""

    def connect(self) -> str:
        with self.lock:
            if self.sock is not None:
                return self.banner
            sock = socket.create_connection((self.profile.board_host, self.profile.tcp_port),
                                            timeout=self.profile.command_timeout_s)
            sock.settimeout(self.profile.command_timeout_s)
            self.sock = sock
            try:
                self.banner = self._receive_until(lambda text: "Ready" in text)
            except Exception:
                self.close()
                raise
            return self.banner

    def close(self) -> None:
        with self.lock:
            sock, self.sock = self.sock, None
            self.buffer.clear()
            if sock is not None:
                try:
                    # The Nios single-client server has an explicit Quit
                    # command; closing by FIN alone leaves it occupied.
                    sock.sendall(b"q\n")
                except OSError:
                    pass
                try:
                    sock.settimeout(0.25)
                    sock.recv(1024)
                except OSError:
                    pass
                sock.close()

    def _receive_until(self, predicate: Callable[[str], bool]) -> str:
        if self.sock is None:
            raise FPGAProtocolError("TCP client is not connected")
        deadline = time.monotonic() + self.profile.command_timeout_s
        while time.monotonic() < deadline:
            text = self.buffer.decode("ascii", errors="replace")
            if predicate(text):
                self.buffer.clear()
                return text
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise FPGAProtocolError("board closed the TCP connection")
            self.buffer.extend(chunk)
        text = self.buffer.decode("ascii", errors="replace")
        self.buffer.clear()
        raise FPGAProtocolError(f"timeout waiting for board response; received {text!r}")

    def write_register(self, address: int, value: int) -> str:
        if not 0 <= int(address) <= 0xFFFFFFFF or not 0 <= int(value) <= 0xFFFFFFFF:
            raise ValueError("address and value must be uint32")
        with self.lock:
            if self.sock is None:
                raise FPGAProtocolError("TCP client is not connected")
            # Exact spelling follows the command examples saved in the supplied VI.
            self.sock.sendall(f"Pw{int(address)},{int(value)}\n".encode("ascii"))
            response = self._receive_until(lambda text: bool(self.WRITE_RE.search(text)))
            match = self.WRITE_RE.search(response)
            if (not match or int(match.group(2), 16) != int(address)
                    or int(match.group(1), 16) != int(value)):
                raise FPGAProtocolError(
                    f"write response does not match register/value {address}/{value}: {response!r}")
            return response

    def read_register(self, address: int) -> int:
        with self.lock:
            if self.sock is None:
                raise FPGAProtocolError("TCP client is not connected")
            self.sock.sendall(f"Pr{int(address)}\n".encode("ascii"))
            response = self._receive_until(lambda text: bool(self.READ_RE.search(text)))
            match = self.READ_RE.search(response)
            if not match or int(match.group(2), 16) != int(address):
                raise FPGAProtocolError(f"read response does not match address {address}: {response!r}")
            return int(match.group(1), 16)


class SMRUDPReceiver:
    def __init__(self, profile: NetworkProfile, callback: Callable[[DecodedPacket, SequenceResult, float], None],
                 *, signal: str = "frequency"):
        self.profile = profile
        self.callback = callback
        self.signal = signal
        self.sock: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.sequence = PacketSequenceTracker()
        self.error: Exception | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.profile.interface_host, self.profile.udp_port))
        membership = socket.inet_aton(self.profile.multicast_group) + socket.inet_aton(self.profile.interface_host)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.settimeout(0.25)
        self.sock = sock
        self.stop_event.clear()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="smr-udp-receiver", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.sock is not None
        try:
            while not self.stop_event.is_set():
                try:
                    data, _ = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue
                packet = decode_packet(data, signal=self.signal, byte_order=self.profile.byte_order)
                sequence = (self.sequence.observe(packet.header)
                            if self.profile.header_has_sequence else
                            SequenceResult("header_unverified", 0, 0))
                self.callback(packet, sequence, time.monotonic())
        except Exception as exc:
            self.error = exc
        finally:
            if self.sock is not None:
                self.sock.close()
                self.sock = None

    def stop(self, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout_s)
        self.thread = None
