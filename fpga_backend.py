"""High-level, single-owner backend for the supplied SMR FPGA firmware."""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Mapping

import numpy as np

from fpga_registers import (
    DEFAULT_REGISTER_ADDRESSES, ModeFields, PLLRegisterValues, PLLUserParameters,
    frequency_to_word, register_write_map, user_parameters_to_registers,
)
from fpga_stream import DecodedPacket, SequenceResult
from fpga_transport import NetworkProfile, SMRTCPClient, SMRUDPReceiver


@dataclass(frozen=True)
class FPGACapabilities:
    backend: str = "legacy-smr-fpga"
    complex_stream_verified: bool = True
    device_timestamp: bool = False
    hardware_lock_flag: bool = False
    pll_derivative: bool = False
    maximum_channels: int = 1


@dataclass(frozen=True)
class StreamBatch:
    channel: int
    segment_id: int
    packet_index: int
    sequence_classification: str
    missing_packets: int
    host_monotonic_s: float
    values: np.ndarray


class FPGAController:
    """Own one TCP session and one optional multicast receiver.

    Register writes are serialized by SMRTCPClient.  No method automatically
    reconnects or re-enables output after a communication failure.
    """
    def __init__(self, profile: NetworkProfile, *, channel: int = 0,
                 channel_stride: int = 0x20,
                 addresses: Mapping[str, int] = DEFAULT_REGISTER_ADDRESSES):
        if channel < 0:
            raise ValueError("channel must be non-negative")
        self.profile = profile
        self.channel = int(channel)
        self.channel_stride = int(channel_stride)
        self.addresses = dict(addresses)
        self.tcp = SMRTCPClient(profile)
        self.receiver: SMRUDPReceiver | None = None
        self.stream_queue: queue.Queue[StreamBatch] = queue.Queue(maxsize=512)
        self.dropped_local_batches = 0
        self.last_register_values: PLLRegisterValues | None = None
        self.last_parameters: PLLUserParameters | None = None
        self.capabilities = FPGACapabilities(maximum_channels=1)
        self._state_lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self.tcp.sock is not None

    def connect(self) -> str:
        return self.tcp.connect()

    def disconnect(self) -> None:
        self.stop_stream()
        self.tcp.close()

    def _address(self, base: int) -> int:
        return int(base) + self.channel * self.channel_stride

    def write_named(self, name: str, value: int) -> str:
        if name not in self.addresses:
            raise KeyError(f"unknown register name {name!r}")
        return self.tcp.write_register(self._address(self.addresses[name]), int(value) & 0xFFFFFFFF)

    def read_named(self, name: str) -> int:
        if name not in self.addresses:
            raise KeyError(f"unknown register name {name!r}")
        return self.tcp.read_register(self._address(self.addresses[name]))

    def apply_parameters(self, parameters: PLLUserParameters, *, start: bool = False,
                         verify_readback: bool = False) -> PLLRegisterValues:
        """Apply a coherent configuration, writing mode last.

        Configuration is first written with Run and stream disabled.  This is
        the only safe ordering supported without an atomic firmware command.
        """
        if parameters.loop_order != 1:
            raise ValueError(
                "deployed FPGA_Python profile verifies only a first-order target response "
                "for the type-2 PI PLL")
        if not 0 < parameters.minimum_frequency_hz <= parameters.maximum_frequency_hz < 6_250_000:
            raise ValueError("verified NCO profile requires frequencies in (0, 6.25 MHz)")
        safe_mode = ModeFields(**{
            **parameters.mode.__dict__, "run": False, "send_data_to_pc": False,
        })
        safe_parameters = PLLUserParameters(**{**parameters.__dict__, "mode": safe_mode})
        values = user_parameters_to_registers(safe_parameters)
        writes = register_write_map(values, self.addresses)
        mode_address = self.addresses["mode"]
        self.write_named("mode", 0)
        self.write_named("nco_gain", 0)
        for address, word in writes.items():
            if address == mode_address:
                continue
            self.tcp.write_register(self._address(address), word)
        self.tcp.write_register(self._address(mode_address), values.mode)
        if verify_readback:
            for address, expected in writes.items():
                actual = self.tcp.read_register(self._address(address))
                if actual != expected:
                    raise RuntimeError(
                        f"register 0x{address:x} readback 0x{actual:08x}, expected 0x{expected:08x}")
        with self._state_lock:
            self.last_register_values = values
            self.last_parameters = safe_parameters
        if start:
            self.start_tracking(stream=parameters.mode.send_data_to_pc)
        return values

    def apply_verified_sweep_preset(self, minimum_hz: float, maximum_hz: float,
                                    drive_fraction: float) -> None:
        """Captured channel-A LockInSweep preset from working FPGA_Python GUI."""
        if self.channel != 0:
            raise ValueError("captured sweep preset is verified only for channel A")
        if not 0 < minimum_hz <= maximum_hz < 6_250_000:
            raise ValueError("sweep frequencies must lie in (0, 6.25 MHz)")
        if not 0 <= drive_fraction <= 1:
            raise ValueError("drive fraction must be in [0, 1]")
        self.write_named("mode", 0)
        self.write_named("nco_gain", 0)
        preset = [
            (1, frequency_to_word(minimum_hz)),
            (2, frequency_to_word(maximum_hz)),
            (14, 1605631), (17, 77213), (18, 0xFFFFFFFF),
            (5, 0), (6, 0), (20, 0),
            (22, 0x40000000), (23, 0), (24, 0), (25, 0), (26, 0),
            (27, 0x40000000), (28, 0), (29, 0), (30, 0), (31, 0),
        ]
        for address, word in preset:
            self.tcp.write_register(address, word)
        self.write_named("nco_gain", round(drive_fraction * 0xFFFFFFFF))
        mode = ModeFields(enable_agc=True, send_data_to_pc=True,
                          run_nco_at_fixed_frequency=True,
                          signal_of_interest=4, dac_a_output=1)
        self.write_named("mode", mode.encode())
        with self._state_lock:
            self.last_parameters = PLLUserParameters(
                frequency_hz=minimum_hz, minimum_frequency_hz=minimum_hz,
                maximum_frequency_hz=maximum_hz, resonator_q=1200,
                loop_bandwidth_hz=100, loop_order=1,
                cic_rate=32767, cic_bit_shift=24,
                drive_amplitude_fraction=drive_fraction, mode=mode)

    def restart_all_plls(self) -> None:
        """Pulse the captured broadcast restart register after PLL Run."""
        if self.last_parameters is None:
            raise RuntimeError("configure PLL before restart")
        self.tcp.write_register(0xFF00, 0)
        self.tcp.write_register(0xFF00, self.last_parameters.mode.encode())

    def set_fixed_frequency(self, frequency_hz: float) -> float:
        word = frequency_to_word(frequency_hz)
        self.write_named("phase_increment_upon_reset", word)
        return word * 12_500_000.0 / 2**32

    def _set_mode(self, *, run: bool, stream: bool, fixed: bool | None = None) -> int:
        with self._state_lock:
            if self.last_parameters is None:
                raise RuntimeError("apply parameters before changing run state")
            old = self.last_parameters.mode
            mode = ModeFields(**{
                **old.__dict__, "run": bool(run), "send_data_to_pc": bool(stream),
                "run_nco_at_fixed_frequency": old.run_nco_at_fixed_frequency if fixed is None else bool(fixed),
            })
            encoded = mode.encode()
            self.write_named("mode", encoded)
            self.last_parameters = PLLUserParameters(**{**self.last_parameters.__dict__, "mode": mode})
            return encoded

    def start_fixed(self, *, stream: bool = True) -> int:
        return self._set_mode(run=True, stream=stream, fixed=True)

    def start_tracking(self, *, stream: bool = True) -> int:
        return self._set_mode(run=True, stream=stream, fixed=False)

    def stop_output(self) -> None:
        """Attempt both Run=0 and drive=0, even if one write fails."""
        errors = []
        try:
            self.write_named("mode", 0)
        except Exception as exc:
            errors.append(f"Run=0: {exc}")
        try:
            self.write_named("nco_gain", 0)
        except Exception as exc:
            errors.append(f"drive=0: {exc}")
        if self.last_parameters is not None and not errors:
            old = self.last_parameters
            self.last_parameters = PLLUserParameters(**{
                **old.__dict__, "mode": ModeFields(**{
                    **old.mode.__dict__, "run": False,
                    "send_data_to_pc": False})})
        if errors:
            raise RuntimeError("FPGA stop command failed: " + "; ".join(errors))

    def start_stream(self, *, signal: str = "frequency") -> None:
        if self.receiver is not None:
            return
        # Never let payload decoded under a previous signal mode leak into a
        # new acquisition segment.
        while True:
            try:
                self.stream_queue.get_nowait()
            except queue.Empty:
                break
        self.dropped_local_batches = 0
        self.receiver = SMRUDPReceiver(self.profile, self._on_packet, signal=signal)
        self.receiver.start()

    def _on_packet(self, packet: DecodedPacket, sequence: SequenceResult,
                   received_monotonic_s: float) -> None:
        batch = StreamBatch(
            channel=packet.header.channel_id, segment_id=sequence.segment_id,
            packet_index=packet.header.packet_index,
            sequence_classification=sequence.classification,
            missing_packets=sequence.missing_packets,
            host_monotonic_s=received_monotonic_s, values=packet.values,
        )
        try:
            self.stream_queue.put_nowait(batch)
        except queue.Full:
            self.dropped_local_batches += 1

    def drain_stream(self, max_batches: int = 100) -> list[StreamBatch]:
        result = []
        for _ in range(max_batches):
            try:
                result.append(self.stream_queue.get_nowait())
            except queue.Empty:
                break
        return result

    def stop_stream(self) -> None:
        receiver, self.receiver = self.receiver, None
        if receiver is not None:
            receiver.stop()


@dataclass(frozen=True)
class SimulatedResonator:
    f0_hz: float = 29_400.0
    q: float = 1200.0
    amplitude: float = 0.25
    background_real: float = 0.01
    background_imag: float = -0.005
    noise_rms: float = 2e-4


class SimulatedFPGAController:
    """Deterministic development backend with the same lifecycle semantics."""
    capabilities = FPGACapabilities(
        backend="simulated-smr-fpga", complex_stream_verified=True,
        device_timestamp=True, hardware_lock_flag=True)

    def __init__(self, resonator: SimulatedResonator | None = None, seed: int = 1234):
        self.resonator = resonator or SimulatedResonator()
        self.rng = np.random.default_rng(seed)
        self.connected = False
        self.frequency_hz = self.resonator.f0_hz
        self.tracking = False
        self.running = False
        self.streaming = False
        self.last_parameters: PLLUserParameters | None = None
        self.last_register_values: PLLRegisterValues | None = None
        self.started_monotonic = time.monotonic()
        self._tracked_frequency = self.frequency_hz

    def connect(self) -> str:
        self.connected = True
        self.started_monotonic = time.monotonic()
        return "SIMULATED SMR Controller Host Ready"

    def disconnect(self) -> None:
        self.streaming = self.running = self.tracking = self.connected = False

    def apply_parameters(self, parameters: PLLUserParameters, *, start: bool = False,
                         verify_readback: bool = False) -> PLLRegisterValues:
        del verify_readback
        values = user_parameters_to_registers(parameters)
        self.last_parameters, self.last_register_values = parameters, values
        self.frequency_hz = parameters.frequency_hz
        if start:
            self.start_tracking(stream=parameters.mode.send_data_to_pc)
        return values

    def set_fixed_frequency(self, frequency_hz: float) -> float:
        self.frequency_hz = float(frequency_hz)
        return self.frequency_hz

    def complex_response(self, frequency_hz: float, count: int = 1) -> np.ndarray:
        r = self.resonator
        f = float(frequency_hz)
        response = r.amplitude / (1.0 - (f/r.f0_hz)**2 + 1j*f/(r.q*r.f0_hz))
        response += r.background_real + 1j*r.background_imag
        noise = self.rng.normal(0, r.noise_rms, count) + 1j*self.rng.normal(0, r.noise_rms, count)
        return response + noise

    def start_fixed(self, *, stream: bool = True) -> int:
        self.running, self.streaming, self.tracking = True, stream, False
        return 1

    def start_tracking(self, *, stream: bool = True) -> int:
        if self.last_parameters is None:
            raise RuntimeError("apply parameters before tracking")
        self.running, self.streaming, self.tracking = True, stream, True
        self._tracked_frequency = self.resonator.f0_hz
        return 1

    def stop_output(self) -> None:
        self.running = self.streaming = self.tracking = False

    def sample_tracking(self, count: int, rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.tracking:
            raise RuntimeError("tracking is not active")
        start = time.monotonic() - self.started_monotonic
        t = start + np.arange(count, dtype=float) / float(rate_hz)
        drift = 0.35*np.sin(2*np.pi*t/37.0)
        f = self._tracked_frequency + drift + self.rng.normal(0, 0.025, count)
        z = np.array([self.complex_response(value)[0] for value in f])
        return t, f, z
