"""Exercise the actual FPGAWindow tracking path with low bounded output."""
from __future__ import annotations

import argparse
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox

from fpga_resonance_gui import FPGAWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=("frequency", "amplitude"),
                        default="frequency")
    parser.add_argument("--drive", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 <= args.drive <= 0.05:
        parser.error("drive must be 0..0.05 full scale")
    if not 0.2 <= args.duration <= 5:
        parser.error("duration must be 0.2..5 seconds")

    def raise_dialog(*items):
        raise RuntimeError("GUI dialog: " + ", ".join(map(str, items[1:3])))

    QMessageBox.critical = raise_dialog
    app = QApplication([])
    window = FPGAWindow()
    try:
        window.backend_combo.setCurrentText("FPGA hardware")
        window.center.setValue(118000)
        window.minimum.setValue(110000)
        window.maximum.setValue(120000)
        window.q.setValue(140)
        window.loop_bw.setValue(30)
        window.drive_fraction.setValue(args.drive)
        window.stream_choice.setCurrentText(
            "PLL frequency" if args.signal == "frequency" else "PLL amplitude")
        window.connect_backend()
        if window.controller is None:
            raise RuntimeError("GUI TCP connection failed")
        window.start_tracking()
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        values = np.asarray(window.tracking_f, float)
        print("GUI tracking samples:", len(values),
              "rows:", len(window.tracking_rows))
        if len(values):
            print(args.signal, "p01/p50/p99:",
                  *[float(x) for x in np.percentile(values, [1, 50, 99])])
        if len(values) < 128:
            raise RuntimeError("GUI did not receive a full UDP batch")
    finally:
        if window.controller is not None:
            try:
                window.stop_tracking()
                print("GUI stop commands: ACK")
            finally:
                window.disconnect_backend()
        window.close()


if __name__ == "__main__":
    main()
