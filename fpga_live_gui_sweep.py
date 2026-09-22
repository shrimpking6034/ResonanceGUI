"""Exercise the GUI SweepWorker against real FPGA with bounded low drive."""
from __future__ import annotations

import argparse

import numpy as np
from PyQt5.QtCore import QCoreApplication

from fpga_backend import FPGAController
from fpga_registers import PLLUserParameters
from fpga_resonance_gui import SweepWorker
from fpga_transport import NetworkProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--drive", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.drive <= 0.05:
        parser.error("drive must be 0..0.05 full scale")
    app = QCoreApplication([])
    controller = FPGAController(NetworkProfile(board_host=args.host))
    try:
        print(controller.connect().strip())
        params = PLLUserParameters(
            frequency_hz=118000, minimum_frequency_hz=117000,
            maximum_frequency_hz=119000, resonator_q=140,
            loop_bandwidth_hz=30, loop_order=1,
            drive_amplitude_fraction=args.drive)
        worker = SweepWorker(controller, params,
                             np.linspace(117000, 119000, 9), 0.05, 256)
        points, results, failures = [], [], []
        worker.point.connect(lambda *point: points.append(point))
        worker.completed.connect(results.append)
        worker.failed.connect(failures.append)
        worker.run()
        app.processEvents()
        print("sweep points:", len(points), "failures:", failures)
        if results:
            result = results[0]
            print("frequency Hz:", result["freqs"].tolist())
            print("amplitude ADC counts:", result["R"].tolist())
            print("peak Hz:", float(result["freqs"][np.argmax(result["R"])]))
        if failures or len(points) != 9:
            raise RuntimeError("GUI SweepWorker did not complete nine points")
    finally:
        if controller.connected:
            try:
                controller.stop_output()
                print("Run=0 and drive=0 commands: ACK")
            finally:
                controller.disconnect()


if __name__ == "__main__":
    main()
