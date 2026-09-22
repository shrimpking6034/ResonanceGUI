"""Short, bounded live check of the verified channel-A FPGA sweep stream.

The default drive is zero. Always requests Run=0 and drive=0 on exit.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from fpga_backend import FPGAController
from fpga_transport import NetworkProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--center-hz", type=float, default=118000.0)
    parser.add_argument("--drive", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--verify-stop", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.drive <= 0.05:
        parser.error("smoke-test drive must be between 0 and 0.05 full scale")
    if not 0.2 <= args.duration <= 5:
        parser.error("duration must be 0.2..5 seconds")

    controller = FPGAController(NetworkProfile(board_host=args.host))
    output_commanded = False
    try:
        print(controller.connect().strip())
        controller.start_stream(signal="signed")
        controller.apply_verified_sweep_preset(
            args.center_hz - 100, args.center_hz + 100, args.drive)
        controller.set_fixed_frequency(args.center_hz)
        print("configuration ACK; drive fraction", args.drive)
        controller.start_fixed(stream=True)
        output_commanded = True
        deadline = time.monotonic() + args.duration
        batches = []
        while time.monotonic() < deadline:
            batches.extend(controller.drain_stream(1000))
            time.sleep(0.01)
        batches.extend(controller.drain_stream(1000))
        sizes = sorted({len(batch.values) for batch in batches})
        print("UDP batches:", len(batches), "payload words per batch:", sizes)
        if batches:
            data = np.concatenate([batch.values for batch in batches])
            i, q = data[0::2], data[1::2]
            print("I mean/std:", float(i.mean()), float(i.std()))
            print("Q mean/std:", float(q.mean()), float(q.std()))
            print("header sequence classification:", batches[0].sequence_classification)
        if controller.receiver and controller.receiver.error:
            raise RuntimeError(f"UDP receiver error: {controller.receiver.error}")
        if not batches:
            raise RuntimeError("no UDP packets received")
        if args.verify_stop:
            controller.stop_output()
            controller.drain_stream(1000)
            time.sleep(0.5)
            after_stop = controller.drain_stream(1000)
            print("UDP batches 0.5 s after stop:", len(after_stop))
    finally:
        if controller.connected:
            try:
                controller.stop_output()
                print("Run=0 and drive=0 commands: ACK")
            except Exception as exc:
                print("STOP COMMAND FAILURE:", exc)
                raise
            finally:
                controller.disconnect()
        elif output_commanded:
            print("WARNING: TCP disconnected after Run command; output state is unknown")


if __name__ == "__main__":
    main()
