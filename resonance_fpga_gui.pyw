#!C:\Users\user\AppData\Local\Programs\Python\Python310\pythonw.exe
"""Launch the FPGA GUI with the Python used by this workstation."""

from pathlib import Path
import traceback


if __name__ == "__main__":
    try:
        from resonance_fpga_gui import main
        main()
    except Exception:
        log_path = Path(__file__).with_name("resonance_fpga_startup_error.log")
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, f"GUI 실행 오류가 발생했습니다.\n{log_path}",
                "ResonanceGUI", 0x10)
        except Exception:
            pass
