"""Read-only comparison of possible legacy P/r command spellings."""
from __future__ import annotations

import argparse
import socket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=30)
    args = parser.parse_args()
    with socket.create_connection((args.host, args.port), timeout=2.0) as sock:
        sock.settimeout(0.5)
        try:
            print("banner:", repr(sock.recv(4096)))
            for command in (b"PR0\n", b"Pr0\n", b"PR3\n", b"Pr3\n"):
                sock.sendall(command)
                try:
                    response = sock.recv(4096)
                except socket.timeout:
                    response = b"<timeout>"
                print(repr(command), repr(response))
        finally:
            sock.sendall(b"Q\n")
            try:
                print("quit:", repr(sock.recv(4096)))
            except socket.timeout:
                pass


if __name__ == "__main__":
    main()
