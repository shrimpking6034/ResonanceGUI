"""Bounded check of the original HF2LI layout running on the FPGA backend."""
from __future__ import annotations

import time

from PyQt5.QtWidgets import QApplication, QMessageBox

from resonance_fpga_gui import FPGAMainWindow


def _raise_dialog(_parent, title, message, *args, **kwargs):
    raise RuntimeError(f"{title}: {message}")


def main():
    QMessageBox.critical = _raise_dialog
    QMessageBox.warning = _raise_dialog
    app = QApplication([])
    window = FPGAMainWindow()
    try:
        if window.fpga is None:
            raise RuntimeError(window.status_label.text())
        window.pll_center_spin.setValue(118000)
        window.pll_range_spin.setValue(10000)
        window.pll_q_spin.setValue(140)
        window.pll_target_bw_spin.setValue(30)
        window.drive_v_spin.setValue(0.005)
        window.pll_enable_btn.setChecked(True)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        print("HF2-style UI Plotter samples:", window.received_sample_count)
        print("frequency display:", window.pll_big_freq.text())
        print("lock display:", window.pll_big_locked.text())
        if window.received_sample_count < 128:
            raise RuntimeError("original-layout Plotter did not receive a UDP batch")
        window.pll_enable_btn.setChecked(False)
        window.f_start_spin.setValue(117000)
        window.f_stop_spin.setValue(119000)
        window.n_points_spin.setValue(10)
        window.sweep_samples_spin.setValue(256)
        window.sweep_min_settle_spin.setValue(0.05)
        window.run_sweep()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and (
                window.sweep_worker and window.sweep_worker.isRunning()):
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        print("HF2-style UI Sweep points:", len(window.sweep_metadata.get("freqs", [])))
        if len(window.sweep_metadata.get("freqs", [])) != 10:
            raise RuntimeError("original-layout Sweep did not finish ten points")
    finally:
        window.close()
        print("FPGA Run=0/drive=0 and TCP disconnect requested")


if __name__ == "__main__":
    main()
