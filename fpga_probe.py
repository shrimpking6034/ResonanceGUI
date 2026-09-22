"""Read-only connection/register diagnostic for the legacy SMR board."""
from __future__ import annotations

import argparse

from fpga_backend import FPGAController
from fpga_registers import DEFAULT_REGISTER_ADDRESSES
from fpga_transport import NetworkProfile


def main():
    parser = argparse.ArgumentParser(
        description="Connect to an SMR controller and read known registers without enabling Run.")
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--tcp-port", type=int, default=30)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--channel-stride", type=lambda value: int(value, 0), default=0x20)
    args = parser.parse_args()
    controller = FPGAController(
        NetworkProfile(board_host=args.host, tcp_port=args.tcp_port),
        channel=args.channel, channel_stride=args.channel_stride)
    try:
        banner = controller.connect()
        print(banner.strip())
        for name, address in DEFAULT_REGISTER_ADDRESSES.items():
            try:
                value = controller.read_named(name)
            except Exception as exc:
                print(f"{name:32s} base=0x{address:02x} ERROR {exc}")
                break
            print(f"{name:32s} base=0x{address:02x} value=0x{value:08x} ({value})")
    finally:
        # This utility is deliberately read-only; disconnect must not alter mode.
        controller.disconnect()


if __name__ == "__main__":
    main()
