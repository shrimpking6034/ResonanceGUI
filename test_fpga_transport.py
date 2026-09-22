import socket
import threading
import unittest

from fpga_transport import NetworkProfile, SMRTCPClient


class FakeSMRServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.commands = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        connection, _ = self.sock.accept()
        with connection:
            connection.sendall(b"SMR Controller Host Ready\r\n")
            pending = b""
            while True:
                data = connection.recv(1024)
                if not data:
                    break
                pending += data
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    command = line.decode("ascii")
                    self.commands.append(command)
                    if command.startswith("Pw"):
                        address, value = map(int, command[2:].split(","))
                        connection.sendall(
                            f"wrote: {value:x} to PIO at {address:x}\r\n".encode())
                    elif command.startswith("Pr"):
                        address = int(command[2:])
                        connection.sendall(
                            f"read: deadbeef from PIO at {address:x}\r\n".encode())
                    elif command == "q":
                        connection.sendall(b"Terminating connection.\r\n")
                        return

    def close(self):
        self.sock.close()
        self.thread.join(1)


class TCPTransportTests(unittest.TestCase):
    def test_commands_and_hex_responses_match_legacy_protocol(self):
        server = FakeSMRServer(); server.start()
        client = SMRTCPClient(NetworkProfile(
            board_host="127.0.0.1", tcp_port=server.port, command_timeout_s=0.5))
        banner = client.connect()
        self.assertIn("Host Ready", banner)
        client.write_register(3, 343597384)
        self.assertEqual(client.read_register(7), 0xDEADBEEF)
        client.close(); server.close()
        self.assertEqual(server.commands, ["Pw3,343597384", "Pr7", "q"])


if __name__ == "__main__":
    unittest.main()
