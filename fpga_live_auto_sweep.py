"""Run the GUI's automatic sweep on hardware and retain its result browser.

No hardware operation occurs on import. Pass --run to execute; without it,
only the GUI is opened with the supplied theory/settings.
"""
import argparse
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from resonance_fpga_gui import FPGAMainWindow
from fpga_auto_sweep import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=8)
    parser.add_argument("--length-mm", type=float, default=60)
    parser.add_argument("--outer-mm", type=float, default=.8)
    parser.add_argument("--inner-mm", type=float, default=.6)
    parser.add_argument("--young-gpa", type=float, default=193)
    parser.add_argument("--tube-density", type=float, default=7930)
    parser.add_argument("--fluid-density", type=float, default=1.2)
    parser.add_argument("--drive", type=float, required=True)
    parser.add_argument("--coarse-step", type=float, default=10)
    parser.add_argument("--coarse-percent", type=float, default=10)
    parser.add_argument("--fine-step", type=float, default=1)
    parser.add_argument("--settle", type=float, default=.1)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--load", type=Path, help="Open a saved run instead of measuring")
    parser.add_argument("--verify-stop", action="store_true", help="Read mode/gain without starting output")
    parser.add_argument("--exit-after", action="store_true")
    args = parser.parse_args()
    if not 0 < args.drive <= 1:
        parser.error("drive must be in (0, 1]")
    app = QApplication([])
    window = FPGAMainWindow()
    for name, value in dict(theory_length_spin=args.length_mm, theory_outer_spin=args.outer_mm,
                            theory_inner_spin=args.inner_mm, theory_e_spin=args.young_gpa,
                            theory_density_spin=args.tube_density, theory_fluid_density_spin=args.fluid_density,
                            theory_count_spin=max(args.last, 8), drive_v_spin=args.drive).items():
        getattr(window, name).setValue(value)
    for key, value in dict(first_mode=args.first, last_mode=args.last,
                           coarse_step_hz=args.coarse_step, coarse_percent=args.coarse_percent,
                           fine_step_hz=args.fine_step, coarse_settle_s=args.settle,
                           fine_settle_s=args.settle, coarse_samples=args.samples, fine_samples=args.samples).items():
        window.auto_fields[key].setValue(value)
    window.resize(1600, 1000)
    window.show()
    directory = args.output or Path("outputs")/datetime.now().strftime("fpga_auto_live_%Y%m%d_%H%M%S")

    def start():
        if window.fpga is None:
            print("CONNECT FAILED:", window.status_label.text(), flush=True)
            if args.exit_after:
                window.close()
                app.exit(1)
            return
        window.start_auto_sweep(directory=directory)
        if window.auto_worker is None:
            print("START FAILED", flush=True)
            return
        worker = window.auto_worker
        worker.stage_started.connect(lambda m: print("STAGE", m, flush=True))
        worker.stage_done.connect(lambda m: print("SAVED", m["file"], len(m["data"].get("freqs", [])), m["status"], flush=True))
        worker.point.connect(lambda p: print("PROGRESS", p["mode"], p["stage"], p["count"], p["total"], flush=True) if p["count"] % 100 == 0 else None)
        worker.mode_done.connect(lambda r: print("MODE", r["mode"], r["status"], r.get("f0_hz"), r["notes"], flush=True))

        def finished():
            print("FINISHED", worker.outcome["status"], worker.outcome["error"], worker.outcome["elapsed_s"], flush=True)
            print("RESULTS", directory.resolve(), flush=True)
            window.grab().save(str(directory/"gui_results.png"))
            if args.exit_after:
                window.close()
                app.exit(0 if worker.outcome["status"] == "complete" else 1)
        worker.finished.connect(finished)

    if args.verify_stop:
        if window.fpga is None:
            raise RuntimeError("FPGA not connected for stop verification")
        mode = window.fpga.read_named("mode")
        gain = window.fpga.read_named("nco_gain")
        verification = dict(mode_word=mode, nco_gain_word=gain, output_stopped=(mode == 0 and gain == 0))
        print("STOP READBACK", verification, flush=True)
        if args.load:
            write_json(args.load / "hardware_stop_verification.json", verification)
    if args.load:
        window.open_auto_session(directory=args.load)
    elif args.run:
        QTimer.singleShot(500, start)
    else:
        window.preview_auto_sweep()
    app.exec_()


if __name__ == "__main__":
    main()
