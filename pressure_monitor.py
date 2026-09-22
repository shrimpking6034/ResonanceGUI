"""Read-only ITV0050 monitor acquisition; no analog output writes.

Nominal endpoints from SMC ITV0-OM00002: 1..5 V = 1..900 kPa.
These are uncalibrated monitor estimates, not precision pressure measurements.
"""
import math
import time

from PyQt5.QtCore import QThread, pyqtSignal


def pressure_kpa(voltage):
    voltage = float(voltage)
    if not math.isfinite(voltage) or not 1.0 <= voltage <= 5.0:
        return None
    return 1.0 + (voltage - 1.0) * 899.0 / 4.0


class PressureMonitor(QThread):
    sample = pyqtSignal(float, float, float)
    failed = pyqtSignal(str)

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device

    def run(self):
        try:
            import nidaqmx
            from nidaqmx.constants import TerminalConfiguration
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    f"{self.device}/ai0:1", terminal_config=TerminalConfiguration.RSE,
                    min_val=-10.0, max_val=10.0)
                while not self.isInterruptionRequested():
                    a, b = task.read(timeout=0.2)
                    self.sample.emit(float(a), float(b), time.monotonic())
                    self.msleep(200)
        except Exception as exc:
            self.failed.emit(str(exc))
