"""Zero-drive register write/readback diagnostic; restoration is unverified."""
from __future__ import annotations

import argparse

from fpga_backend import FPGAController
from fpga_registers import DEFAULT_REGISTER_ADDRESSES, ModeFields, PLLUserParameters
from fpga_transport import NetworkProfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--tcp-port", type=int, default=30)
    args = parser.parse_args()
    controller = FPGAController(NetworkProfile(board_host=args.host, tcp_port=args.tcp_port))
    snapshot = {}
    try:
        print(controller.connect().strip())
        snapshot = {name: controller.read_named(name)
                    for name in DEFAULT_REGISTER_ADDRESSES}
        controller.write_named("mode", 0)  # explicit Run=0 before any other write
        parameters = PLLUserParameters(
            frequency_hz=29_400.0,
            minimum_frequency_hz=29_200.0,
            maximum_frequency_hz=29_600.0,
            resonator_q=1200.0,
            loop_bandwidth_hz=3.0,
            loop_order=2,
            cic_rate=256,
            cic_bit_shift=4,
            pll_delay_turns=0.0,
            drive_amplitude_fraction=0.0,
            mode=ModeFields(run=False, send_data_to_pc=False,
                            run_nco_at_fixed_frequency=True),
        )
        values = controller.apply_parameters(parameters, verify_readback=False)
        print("safe write acknowledgements: PASS")
        mismatches = []
        for name, value in values.as_dict(wire_unsigned=True).items():
            readback = controller.read_named(name)
            matched = readback == value
            if not matched:
                mismatches.append(name)
            print(f"{name:32s} wrote=0x{value:08x} read=0x{readback:08x} "
                  f"{'MATCH' if matched else 'DIFF'}")
        print("readback mismatches:", ", ".join(mismatches) if mismatches else "none")
    finally:
        if snapshot:
            # Keep Run=0 throughout restoration and restore mode last.
            try:
                controller.write_named("mode", 0)
                for name, value in snapshot.items():
                    if name != "mode":
                        controller.write_named(name, value)
                controller.write_named("mode", snapshot["mode"])
                restored = all(controller.read_named(name) == value
                               for name, value in snapshot.items())
                print("state restoration writes: ACKNOWLEDGED")
                print("state restoration read values:",
                      "same as snapshot" if restored else "DIFFERENT from snapshot")
                print("note: readback did not reflect nonzero test writes; "
                      "matching zeros do not prove physical register restoration")
            except Exception as exc:
                print("state restoration ERROR:", exc)
        controller.disconnect()


if __name__ == "__main__":
    main()
