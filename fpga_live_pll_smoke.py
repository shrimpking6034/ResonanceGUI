"""Bounded type-2 PI PLL test with a first-order target loop response."""
from __future__ import annotations

import argparse
import time

import numpy as np

from fpga_backend import FPGAController
from fpga_registers import ModeFields, PLLUserParameters
from fpga_transport import NetworkProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--signal", choices=("frequency", "amplitude"),
                        default="frequency")
    parser.add_argument("--drive", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.5)
    args = parser.parse_args()
    if not 0 <= args.drive <= 0.05:
        parser.error("drive must be 0..0.05 full scale")
    if not 0.2 <= args.duration <= 5:
        parser.error("duration must be 0.2..5 seconds")

    controller = FPGAController(NetworkProfile(board_host=args.host))
    try:
        print(controller.connect().strip())
        controller.start_stream(signal="frequency" if args.signal == "frequency"
                                else "unsigned")
        params = PLLUserParameters(
            frequency_hz=118000, minimum_frequency_hz=110000,
            maximum_frequency_hz=120000, resonator_q=140,
            loop_bandwidth_hz=30, loop_order=1,
            cic_rate=32767, cic_bit_shift=21, pll_delay_turns=0.079,
            drive_amplitude_fraction=args.drive, feedback_gain=0.1,
            mode=ModeFields(enable_agc=True, send_data_to_pc=True,
                            run_nco_at_fixed_frequency=False, dac_a_output=1,
                            signal_of_interest=0 if args.signal == "frequency" else 2))
        controller.apply_parameters(params)
        controller.start_tracking(stream=True)
        controller.restart_all_plls()
        print("PLL configuration and restart ACK; drive fraction", args.drive)
        deadline = time.monotonic() + args.duration
        batches = []
        while time.monotonic() < deadline:
            batches.extend(controller.drain_stream(1000))
            time.sleep(0.01)
        batches.extend(controller.drain_stream(1000))
        print("UDP batches:", len(batches),
              "payload word counts:", sorted({len(b.values) for b in batches}))
        if batches:
            values = np.concatenate([b.values for b in batches])
            print(args.signal, "min/median/max:",
                  float(values.min()), float(np.median(values)), float(values.max()))
            print("finite samples:", int(np.isfinite(values).sum()),
                  "of", len(values))
            first = batches[0].host_monotonic_s
            settled_batches = [b for b in batches
                               if b.host_monotonic_s - first >= min(0.5, args.duration/3)]
            if settled_batches:
                settled = np.concatenate([b.values for b in settled_batches])
                print("after startup, p01/p50/p99:",
                      *[float(x) for x in np.percentile(settled, [1, 50, 99])])
                if args.signal == "frequency":
                    print("after startup within 110000..120000 Hz:",
                          float(np.mean((settled >= 110000) & (settled <= 120000))))
        if controller.receiver and controller.receiver.error:
            raise RuntimeError(f"UDP receiver error: {controller.receiver.error}")
        if not batches:
            raise RuntimeError("no UDP packets received")
    finally:
        if controller.connected:
            try:
                controller.stop_output()
                print("Run=0 and drive=0 commands: ACK")
            finally:
                controller.disconnect()


if __name__ == "__main__":
    main()
