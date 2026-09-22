"""Bounded channel-A PLL comparison for the observed periodic frequency dips.

Use only after the GUI releases the board's sole TCP connection.  The output
is stopped between cases and in a finally block.  This is a diagnostic trace,
not a hardware lock detector.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np

from fpga_backend import FPGAController
from fpga_registers import ModeFields, PLLUserParameters
from fpga_transport import NetworkProfile


CENTER_HZ = 39324.8
Q = 37.6
SPAN_HZ = 500.0
DURATION_S = 8.0
CASES = (
    ("low_drive_agc_on_bw80", 0.05, True, 80.0),
    ("original_agc_on_bw80", 0.48, True, 80.0),
    ("agc_off_bw80", 0.48, False, 80.0),
    ("agc_off_bw20", 0.48, False, 20.0),
)


def capture_case(controller: FPGAController, label: str, drive: float,
                 agc: bool, bandwidth: float) -> dict[str, np.ndarray]:
    parameters = PLLUserParameters(
        frequency_hz=CENTER_HZ,
        minimum_frequency_hz=CENTER_HZ - SPAN_HZ,
        maximum_frequency_hz=CENTER_HZ + SPAN_HZ,
        resonator_q=Q, loop_bandwidth_hz=bandwidth, loop_order=1,
        cic_rate=32767, cic_bit_shift=21, pll_delay_turns=0.079,
        drive_amplitude_fraction=drive, feedback_gain=0.1,
        mode=ModeFields(enable_agc=agc, send_data_to_pc=True,
                        run_nco_at_fixed_frequency=False, dac_a_output=1),
    )
    values, host_times, channel_ids, packet_indices = [], [], [], []
    try:
        controller.start_stream(signal="frequency")
        controller.apply_parameters(parameters)
        controller.start_tracking(stream=True)
        controller.restart_all_plls()
        start = time.monotonic()
        while time.monotonic() - start < DURATION_S:
            receiver = controller.receiver
            if receiver is not None and receiver.error is not None:
                raise RuntimeError(f"UDP receiver: {receiver.error}")
            for batch in controller.drain_stream(100):
                if batch.channel != 0:
                    continue
                values.extend(batch.values.tolist())
                host_times.append(batch.host_monotonic_s - start)
                channel_ids.append(batch.channel)
                packet_indices.append(batch.packet_index)
            time.sleep(0.005)
        if not values:
            raise RuntimeError(f"{label}: no frequency packets")
        trace = np.asarray(values, dtype=np.float64)
        print(f"{label}: N={len(trace)}, median={np.median(trace):.3f} Hz, "
              f"p01={np.percentile(trace, 1):.3f}, "
              f"p99={np.percentile(trace, 99):.3f}, "
              f"jumps>15Hz={np.count_nonzero(np.abs(np.diff(trace)) > 15)}",
              flush=True)
        return {
            f"{label}_values_hz": trace,
            f"{label}_packet_host_times_s": np.asarray(host_times),
            f"{label}_channel_ids": np.asarray(channel_ids),
            f"{label}_packet_indices": np.asarray(packet_indices),
        }
    finally:
        try:
            controller.stop_output()
        finally:
            controller.stop_stream()
        time.sleep(0.5)


def main() -> None:
    controller = FPGAController(NetworkProfile())
    traces = {}
    try:
        print(controller.connect().strip(), flush=True)
        for args in CASES:
            traces.update(capture_case(controller, *args))
    finally:
        try:
            controller.stop_output()
        finally:
            controller.disconnect()
        print("FPGA Run=0/drive=0 requested", flush=True)
    directory = Path(__file__).resolve().parent / "diagnostics"
    directory.mkdir(exist_ok=True)
    path = directory / f"fpga_pll_stability_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez_compressed(path, **traces)
    print(path, flush=True)


if __name__ == "__main__":
    main()
