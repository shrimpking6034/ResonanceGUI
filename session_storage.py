"""Disk-backed Plotter rows and small persistent GUI state."""
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


RAW_COLUMNS = [
    "SessionID", "Time_s", "Frequency_Hz", "Amplitude_R_V", "Phase_deg", "X_V", "Y_V",
    "PLL_TargetPhase_deg", "PLL_PhaseError_deg", "PLL_FrequencyCommand_Hz",
    "PLL_Correction_Hz", "PLL_LockStatus", "PLLStatusAge_s",
    "PressureCommand_A_V", "PressureCommand_B_V", "FlowDirection", "SwitchState", "SwitchEvent",
    "MeasuredPressure_A_kPa", "MeasuredPressure_B_kPa", "MeasuredDifferentialPressure_kPa",
    "Temperature_C", "DemodReadLatency_s", "PLLReadLatency_s",
]


class PlotterSpool:
    def __init__(self):
        fd, name = tempfile.mkstemp(prefix="resonance_plotter_", suffix=".csv")
        self.path = Path(name)
        self.handle = os.fdopen(fd, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.rows = 0
        self.first_t = None
        self.last_t = None

    def append(self, rows):
        if not rows:
            return
        self.writer.writerows(rows)
        self.rows += len(rows)
        if self.first_t is None:
            self.first_t = float(rows[0][1])
        self.last_t = float(rows[-1][1])
        self.handle.flush()

    def export(self, path, metadata):
        self.handle.flush()
        with open(path, "w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            for key, value in metadata.items():
                writer.writerow(["# Metadata", key, value])
            writer.writerow(RAW_COLUMNS)
            with self.path.open("r", encoding="utf-8", newline="") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    def close(self):
        if not self.handle.closed:
            self.handle.close()

    def discard(self):
        self.close()
        self.path.unlink(missing_ok=True)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_sweep(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
