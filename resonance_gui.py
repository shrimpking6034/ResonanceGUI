"""Resonance sweep + PLL tracking GUI for a steel-tube resonator driven by
two piezo actuators, read out via a Zurich Instruments HF2LI lock-in
amplifier, with pneumatic (2x air regulator) control via an NI-DAQ USB-6361.

Mirrors the three LabOne tools actually used for this measurement:
  - Sweeper: Amplitude & Phase vs Frequency. Math: Resonance Q / Center /
    Phase, computed from the detected peaks. Peaks are clickable directly
    on the graph (or via the list) to load them into the PLL panel.
  - PLL: parameter control only (center f0, Q, setpoint, order, TC, range,
    P/I/D). Its "Response" section shows live Error (deg), Freq Shift (Hz)
    and Locked status -- PLL does not do the time-series plotting itself.
  - Plotter: where you actually watch the data. Sample Frequency and
    Sample R are plotted together on one graph (dual Y axis) vs. time,
    for as long as it is running (whether or not PLL is engaged). Math:
    Histogram + Std over a selectable source channel (use Frequency to
    get the frequency noise std relevant for resonance-based mass
    sensing, not Amplitude).

  Independently, the two air-regulator control voltages are set via the
  NI-DAQ (Dev3/ao0, Dev3/ao1).

Hardware safety limits (set by the user, hard-enforced in code):
  - Piezo drive amplitude: HF2LI Signal Output, clamped to <= 10 Vpk
    (the HF2LI's own maximum output range; the piezo is driven directly
    from the lock-in's signal output via coax).
  - Air regulator control voltage: NI-DAQ AO, clamped to 0-10 V.

Note on PLL tuning (P/I/D): LabOne's own "PID Advisor" (DUT model =
Resonator Frequency, parameters: resonance frequency + Q) computes P/I/D
by simulating the closed-loop transfer function and numerically optimizing
it against a target bandwidth -- that optimizer lives in the LabOne
application, not in a device node, so it cannot be called directly from
here. What IS implemented is the same starting point the manual says the
Advisor itself uses first: a Ziegler-Nichols-style estimate from the
resonator's phase-vs-frequency slope (2Q/f0). Treat the suggested P as a
starting point only -- per the manual's own recommended manual workflow,
enable with I = D = 0 first, verify the sign/behavior, then increase I
gradually while watching the PLL Error / lock quality.

Threading note: the ziAPI ZIConnection is documented as NOT thread-safe
(one connection must only be used from one thread). Each background
worker below therefore opens its own independent connection to the HF2
Data Server rather than sharing the GUI-thread connection.
"""
import csv
from bisect import bisect_left
import json
import sys
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QListWidget,
    QListWidgetItem, QMessageBox, QComboBox, QProgressBar, QTabWidget,
    QScrollArea, QCheckBox, QFileDialog, QLineEdit, QToolButton,
    QSizePolicy, QSplitter,
)
import pyqtgraph as pg
from zi_hf2_api import ZiHF2, ZiAPIError
from beam_modes import TubeBeam, calculate_modes
from sweep_math import make_filter_plan, average_complex_samples, tc_from_bandwidth_3db
from resonance_fit import (fit_complex_resonance, model_response,
                           detect_resonance_peaks, assess_fit_quality, three_db_bounds)
from pll_design import design_pll
from baseline_analysis import (timing_quality, linear_baseline, rolling_metrics,
                               event_timescale_noise, false_event_distribution,
                               transient_flags, correlations, allan_deviation,
                               load_plotter_csv, detect_particle_candidates,
                               calibrate_waveform_thresholds, detect_waveform_events,
                               evaluate_blank, poisson_rate_summary,
                               EVENT_WINDOWS_S, DETECTION_DURATIONS_S, ROLLING_WINDOWS_S)
from experiment_analysis import compare_baselines, write_comparison, integration_diagnostics
from particle_review_core import Settings as ParticleSettings
from particle_review_core import calibrate_detector, detect_calibrated
from live_savgol import filter_frequency
from session_storage import PlotterSpool, atomic_json, atomic_sweep
from allan_export import save_allan_export
from pressure_monitor import PressureMonitor, pressure_kpa

try:
    import nidaqmx
    import nidaqmx.system
    NIDAQMX_IMPORT_ERROR = None
except ImportError as exc:
    nidaqmx = None
    NIDAQMX_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Hardware configuration
#
# HF2_DEVICE and DAQ_DEVICE are DEFAULTS, overwritten by discover_hardware()
# at startup (see main()) before any UI is built. This matters because:
#   - The HF2 serial (e.g. "DEV1213") is fixed to that physical unit, but a
#     fresh install on another PC has no reason to already know it.
#   - The NI-DAQ name (e.g. "Dev3") is NOT tied to the physical card at all
#     -- it's an alias NI-MAX assigns per-PC based on enumeration order/
#     history, so "Dev3" here is almost certainly "Dev1" or something else
#     on a different computer. Hardcoding it would silently fail there.
# ---------------------------------------------------------------------------
HF2_DEVICE = "DEV1213"
HF2_OSC = 0                  # oscillator driving the actuation piezo & demod ref
HF2_SIGOUT = 0                # signal output channel wired to the actuation piezo
HF2_SIGOUT_MIXER = 6          # SIGOUTS/0/AMPLITUDES index used for the drive amplitude
HF2_DEMOD = 0                 # demodulator reading the sense piezo (for R/theta/X/Y)
HF2_PLL = 0

MAX_DRIVE_V = 10.0            # hard safety limit, HF2LI output feeds the piezo directly
STATE_DIR = (Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ResonanceGUI"
             if getattr(sys, "frozen", False) else Path(__file__).resolve().parent / ".resonance_state")
LIVE_LOG_MAX_ROWS = 100000
LIVE_LOG_WINDOW_S = 120.0
APP_VERSION = "2026.09.10.1"
OUTPUT_RANGE_OPTIONS = [("10 mV", 0.01), ("100 mV", 0.1), ("1 V", 1.0), ("10 V", 10.0)]

# Default to the NI-MAX alias used by the original control program.  Startup
# still auto-detects a two-AO device when NI-DAQ is available, but this
# fallback matters on PCs where enumeration is temporarily unavailable.
DAQ_DEVICE = "Dev1"
# Keep the original NI-MAX alias fixed.  Set this to True only when moving
# the program to a different PC and intentionally want the first two-AO
# device to replace the Dev1 default.
AUTO_DETECT_DAQ = False


def _build_daq_channels(daq_device):
    return {
        f"채널 1 ({daq_device}/ao0) — 오른쪽→왼쪽": f"{daq_device}/ao0",
        f"채널 2 ({daq_device}/ao1) — 왼쪽→오른쪽": f"{daq_device}/ao1",
    }


DAQ_CHANNELS = _build_daq_channels(DAQ_DEVICE)
MAX_DAQ_V = 10.0
MIN_DAQ_V = 0.0


def discover_hf2_serial():
    """Ask the local HF2 Data Server which device(s) are currently
    connected over USB and return the first one's serial (e.g. 'DEV1213'),
    or None if the server isn't reachable / nothing is plugged in."""
    try:
        zi = ZiHF2()
        try:
            devices_str = zi.get_string("/zi/devices/connected")
        finally:
            zi.close()
        devices = [d.strip() for d in devices_str.split(",") if d.strip()]
        return devices[0].upper() if devices else None
    except Exception:
        return None


def discover_daq_device():
    """Find the first locally-connected NI-DAQ device with >= 2 analog
    output channels (what the two air regulators need), regardless of
    what alias ('Dev1', 'Dev3', ...) this particular PC assigned it."""
    if nidaqmx is None:
        return None, []
    try:
        system = nidaqmx.system.System.local()
        candidates = []
        for dev in system.devices:
            try:
                if len(dev.ao_physical_chans) >= 2:
                    candidates.append(dev.name)
            except Exception:
                continue
        return (candidates[0] if candidates else None), candidates
    except Exception:
        return None, []

HIST_CHANNELS = ["Frequency (Hz)", "Amplitude R (V)", "Phase (deg)", "X (V)", "Y (V)"]


def np_path(sub):
    return f"/{HF2_DEVICE}/{sub}"


def setup_dual_axis(plot):
    """Standard pyqtgraph recipe for a 2nd Y axis sharing the same X axis."""
    vb2 = pg.ViewBox()
    plot.showAxis('right')
    plot.scene().addItem(vb2)
    plot.getAxis('right').linkToView(vb2)
    vb2.setXLink(plot)

    def update_views():
        vb2.setGeometry(plot.vb.sceneBoundingRect())
        vb2.linkedViewChanged(plot.vb, vb2.XAxis)

    plot.vb.sigResized.connect(update_views)
    update_views()
    return vb2


class CollapsibleSection(QWidget):
    """Compact disclosure section used by the long left control panel."""

    def __init__(self, title, content, expanded=False, parent=None):
        super().__init__(parent)
        self.content = content
        if isinstance(content, QGroupBox):
            content.setTitle("")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setStyleSheet(
            "QToolButton { text-align: left; font-weight: bold; padding: 6px; "
            "border: 1px solid #b8b8b8; background: #eeeeee; }"
        )
        self.content.setVisible(bool(expanded))
        self.toggle.toggled.connect(self._set_expanded)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _set_expanded(self, expanded):
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        self.updateGeometry()


# ---------------------------------------------------------------------------
# HF2LI controller (thin domain layer over the ctypes wrapper)
# ---------------------------------------------------------------------------
class HFController:
    """Wraps one dedicated ZIConnection. Not shared across threads."""

    def __init__(self):
        self.zi = ZiHF2()

    def close(self):
        try:
            self.set_output_on(False)
            self.set_amp_enabled(False)
            self.set_pll_enabled(False)
        except Exception:
            pass
        self.zi.close()

    # --- drive ---
    # LabOne's Signal Output tab has THREE independent controls: "On",
    # the per-amplitude "Add"/Enable checkbox ("Amp"), and the Amp value
    # itself -- each toggled/set separately, none of them implied by the
    # others. Mirror that here instead of bundling them.
    def get_output_range(self):
        return self.zi.get_double(np_path(f"SIGOUTS/{HF2_SIGOUT}/RANGE"))

    def set_output_range(self, volts):
        self.zi.set_double(np_path(f"SIGOUTS/{HF2_SIGOUT}/RANGE"), float(volts))

    def set_drive_amplitude(self, volts):
        # SIGOUTS/n/AMPLITUDES/n is documented as a FRACTION of the output
        # Range (0..1), not an absolute voltage -- LabOne's own "Amp (Vpk)"
        # field just shows fraction*Range converted to volts for display.
        # We mirror that conversion here so this Vpk field means the same
        # thing LabOne's does.
        volts = max(0.0, min(MAX_DRIVE_V, float(volts)))
        output_range = self.get_output_range()
        fraction = (volts / output_range) if output_range > 0 else 0.0
        fraction = max(0.0, min(1.0, fraction))
        self.zi.set_double(np_path(f"SIGOUTS/{HF2_SIGOUT}/AMPLITUDES/{HF2_SIGOUT_MIXER}"), fraction)
        return fraction * output_range

    def get_drive_amplitude(self):
        output_range = self.get_output_range()
        fraction = self.zi.get_double(np_path(f"SIGOUTS/{HF2_SIGOUT}/AMPLITUDES/{HF2_SIGOUT_MIXER}"))
        return fraction * output_range

    def set_amp_enabled(self, on):
        self.zi.set_int(np_path(f"SIGOUTS/{HF2_SIGOUT}/ENABLES/{HF2_SIGOUT_MIXER}"), 1 if on else 0)

    def get_amp_enabled(self):
        return bool(self.zi.get_int(np_path(f"SIGOUTS/{HF2_SIGOUT}/ENABLES/{HF2_SIGOUT_MIXER}")))

    def set_output_on(self, on):
        self.zi.set_int(np_path(f"SIGOUTS/{HF2_SIGOUT}/ON"), 1 if on else 0)

    # --- signal inputs ---
    # SIGINS/n/DIFF is the exact device node behind LabOne's "Diff" toggle.
    # n=0 is Signal Input 1 and n=1 is Signal Input 2.
    def set_input_differential(self, input_index, on):
        self.zi.set_int(
            np_path(f"SIGINS/{int(input_index)}/DIFF"), 1 if on else 0
        )

    def get_input_differential(self, input_index):
        return bool(self.zi.get_int(np_path(f"SIGINS/{int(input_index)}/DIFF")))

    def set_input_range(self, input_index, volts):
        self.zi.set_double(np_path(f"SIGINS/{int(input_index)}/RANGE"), float(volts))

    def get_input_range(self, input_index):
        return self.zi.get_double(np_path(f"SIGINS/{int(input_index)}/RANGE"))

    def set_input_ac_coupling(self, input_index, on):
        self.zi.set_int(np_path(f"SIGINS/{int(input_index)}/AC"), 1 if on else 0)

    def get_input_ac_coupling(self, input_index):
        return bool(self.zi.get_int(np_path(f"SIGINS/{int(input_index)}/AC")))

    def set_input_50_ohm(self, input_index, on):
        self.zi.set_int(np_path(f"SIGINS/{int(input_index)}/IMP50"), 1 if on else 0)

    def get_input_50_ohm(self, input_index):
        return bool(self.zi.get_int(np_path(f"SIGINS/{int(input_index)}/IMP50")))

    def set_demod_enabled(self, on):
        # LabOne's own Plotter subscribes to the demodulator's continuous
        # stream, which is gated by this device-level (not per-connection)
        # Enable flag. Stopping our own polling thread does NOT touch this
        # flag by itself, so LabOne's Plotter keeps running unless we
        # explicitly clear it here.
        self.zi.set_int(np_path(f"DEMODS/{HF2_DEMOD}/ENABLE"), 1 if on else 0)

    # --- sweep primitives ---
    def set_frequency(self, freq_hz):
        self.zi.set_double(np_path(f"OSCS/{HF2_OSC}/FREQ"), freq_hz)

    def get_demod_tc(self):
        return self.zi.get_double(np_path(f"DEMODS/{HF2_DEMOD}/TIMECONSTANT"))

    def set_demod_tc(self, seconds):
        self.zi.set_double(np_path(f"DEMODS/{HF2_DEMOD}/TIMECONSTANT"), float(seconds))

    def get_demod_order(self):
        return int(self.zi.get_int(np_path(f"DEMODS/{HF2_DEMOD}/ORDER")))

    def set_demod_order(self, order):
        self.zi.set_int(np_path(f"DEMODS/{HF2_DEMOD}/ORDER"), int(order))

    def get_demod_rate(self):
        return self.zi.get_double(np_path(f"DEMODS/{HF2_DEMOD}/RATE"))

    def set_demod_rate(self, rate_hz):
        self.zi.set_double(np_path(f"DEMODS/{HF2_DEMOD}/RATE"), float(rate_hz))

    def get_clockbase(self):
        return int(self.zi.get_int(np_path("CLOCKBASE")))

    def read_demod(self):
        """Returns (R [V], theta [deg], osc_freq [Hz], x [V], y [V])."""
        _, r, theta, frequency, x, y = self.read_demod_timestamped()
        return r, theta, frequency, x, y

    def read_demod_timestamped(self):
        """As read_demod(), retaining the device timestamp for sample de-duplication."""
        s = self.zi.get_demod_sample(np_path(f"DEMODS/{HF2_DEMOD}/SAMPLE"))
        r = float(np.hypot(s.x, s.y))
        theta = float(np.degrees(np.arctan2(s.y, s.x)))
        return int(s.timeStamp), r, theta, s.frequency, s.x, s.y

    def read_demod_index_timestamped(self, demod_index):
        """Read one of the six user-addressable HF2 demodulators (0--5)."""
        s = self.zi.get_demod_sample(np_path(f"DEMODS/{int(demod_index)}/SAMPLE"))
        r = float(np.hypot(s.x, s.y))
        theta = float(np.degrees(np.arctan2(s.y, s.x)))
        return int(s.timeStamp), r, theta, s.frequency, s.x, s.y

    # --- PLL ---
    def configure_pll(self, freq_center, order, time_constant, freq_range,
                       setpoint_deg, p, i, d):
        # Force manual mode: AUTOCENTER (and the other Auto* flags) override
        # our FREQCENTER/TIMECONSTANT/P-I-D with an internal auto-search --
        # the manual explicitly says closed-loop tracking requires this off.
        # Leftover state from a previous manual LabOne session had this on,
        # which sent the oscillator to a nonsensical ~19.6 MHz.
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/AUTOCENTER"), 0)
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/AUTOTIMECONSTANT"), 0)
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/AUTOPID"), 0)
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/ORDER"), int(order))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/FREQCENTER"), float(freq_center))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/FREQRANGE"), float(freq_range))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/TIMECONSTANT"), float(time_constant))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/SETPOINT"), float(setpoint_deg))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/P"), float(p))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/I"), float(i))
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/D"), float(d))

    def set_pll_input(self, input_index):
        """Select the public PLL input: 0=SigIn1, 1=SigIn2, 2..5=Aux/DIO."""
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/ADCSELECT"), int(input_index))

    def get_pll_input(self):
        return int(self.zi.get_int(np_path(f"PLLS/{HF2_PLL}/ADCSELECT")))

    def set_pll_center_frequency(self, frequency_hz):
        self.zi.set_double(np_path(f"PLLS/{HF2_PLL}/FREQCENTER"), float(frequency_hz))

    def set_pll_enabled(self, on):
        self.zi.set_int(np_path(f"PLLS/{HF2_PLL}/ENABLE"), 1 if on else 0)

    def get_pll_locked(self):
        return bool(self.zi.get_int(np_path(f"PLLS/{HF2_PLL}/LOCKED")))

    def get_pll_error(self):
        """Current phase error of the PLL (Setpoint - PID input), degrees."""
        return self.zi.get_double(np_path(f"PLLS/{HF2_PLL}/ERROR"))

    def get_pll_freq_delta(self):
        """Current frequency shift of the PLL (Oscillator Freq - Center Freq), Hz."""
        return self.zi.get_double(np_path(f"PLLS/{HF2_PLL}/FREQDELTA"))

    def get_osc_freq(self):
        return self.zi.get_double(np_path(f"OSCS/{HF2_OSC}/FREQ"))

    def get_pll_rate(self):
        return self.zi.get_double(np_path(f"PLLS/{HF2_PLL}/RATE"))

# ---------------------------------------------------------------------------
# Background workers -- each opens its OWN HF2 connection (ziAPI connections
# are documented as not thread-safe).
# ---------------------------------------------------------------------------
class SweepWorker(QThread):
    point_ready = pyqtSignal(int, float, float, float, float, float)  # index, f, R, theta, X, Y
    progress = pyqtSignal(int, int)                       # current, total
    finished_ok = pyqtSignal(object)                       # result dictionary
    failed = pyqtSignal(str)

    def __init__(self, f_start, f_stop, n_points, sweep_settings):
        super().__init__()
        self.f_start = f_start
        self.f_stop = f_stop
        self.n_points = n_points
        self.settings = sweep_settings
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        hf = None
        original_filter = None
        try:
            hf = HFController()
            hf.set_demod_enabled(True)
            freqs = np.linspace(self.f_start, self.f_stop, self.n_points)
            original_filter = (hf.get_demod_tc(), hf.get_demod_order(), hf.get_demod_rate())
            if self.settings["sample_rate_hz"] > 0:
                hf.set_demod_rate(self.settings["sample_rate_hz"])
            effective_sample_rate = max(hf.get_demod_rate(), 1.0)
            rows = []
            for i, f in enumerate(freqs):
                if self._stop:
                    break
                step = abs(freqs[i + 1] - f) if i + 1 < len(freqs) else abs(f - freqs[i - 1])
                effective_order = (original_filter[1] if self.settings["filter_mode"].lower() == "manual"
                                   else self.settings["order"])
                plan = make_filter_plan(
                    frequency_hz=f, step_hz=step,
                    order=effective_order, mode=self.settings["filter_mode"],
                    manual_tc_s=original_filter[0], fixed_bandwidth_hz=self.settings["fixed_bandwidth_hz"],
                    max_bandwidth_hz=self.settings["max_bandwidth_hz"],
                    omega_suppression_db=self.settings["omega_suppression_db"],
                    bandwidth_overlap=self.settings["bandwidth_overlap"],
                    min_settling_s=self.settings["min_settling_s"],
                    settling_residual=self.settings["settling_residual"],
                    sample_count=self.settings["sample_count"],
                    sample_rate_hz=effective_sample_rate,
                    averaging_time_s=self.settings["averaging_time_s"],
                    averaging_tc=self.settings["averaging_tc"],
                )
                if self.settings["filter_mode"].lower() != "manual":
                    hf.set_demod_order(plan.order)
                    hf.set_demod_tc(plan.time_constant_s)
                hf.set_frequency(f)
                time.sleep(max(0.001, plan.settling_s))
                xs, ys = [], []
                deadline = time.monotonic() + plan.measurement_s
                sample_interval = max(0.001, 1.0 / effective_sample_rate)
                last_timestamp = None
                while not self._stop and time.monotonic() < deadline:
                    timestamp, _, _, _, x, y = hf.read_demod_timestamped()
                    if timestamp != last_timestamp:
                        xs.append(x)
                        ys.append(y)
                        last_timestamp = timestamp
                    time.sleep(sample_interval)
                if not xs:
                    break
                z, x_sem, y_sem, xy_cov, n_samples = average_complex_samples(xs, ys)
                r, th = abs(z), float(np.angle(z, deg=True))
                rows.append((f, r, th, z.real, z.imag, x_sem, y_sem, xy_cov, n_samples,
                             plan.time_constant_s, plan.bandwidth_3db_hz,
                             plan.settling_s, plan.measurement_s))
                self.point_ready.emit(i, f, r, th, z.real, z.imag)
                self.progress.emit(i + 1, self.n_points)
            columns = list(zip(*rows)) if rows else [[] for _ in range(13)]
            names = ("freqs", "R", "theta", "X", "Y", "X_sem", "Y_sem", "XY_cov", "n_samples",
                     "time_constants", "bandwidths", "settling_times", "measurement_times")
            self.finished_ok.emit({name: np.asarray(values, dtype=float) for name, values in zip(names, columns)})
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if hf:
                if original_filter:
                    try:
                        hf.set_demod_tc(original_filter[0])
                        hf.set_demod_order(original_filter[1])
                        hf.set_demod_rate(original_filter[2])
                    except Exception:
                        pass
                try:
                    hf.set_demod_enabled(False)
                except Exception:
                    pass
                try:
                    hf.zi.close()
                except Exception:
                    pass


class PLLStatusWorker(QThread):
    """Lightweight PLL 'Response' poller: Locked, Error, Freq Shift, Osc Freq.
    No plotting -- this only drives the small status readouts, per the
    LabOne PLL tab's own functional elements (Error, Freq Shift, Locked LED).
    """
    sample_ready = pyqtSignal(bool, float, float, float, float)  # plus total read latency
    failed = pyqtSignal(str)

    def __init__(self, period_s=0.1):
        super().__init__()
        self.period_s = period_s
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        hf = None
        try:
            hf = HFController()
            while not self._stop:
                tick = time.perf_counter()
                locked = hf.get_pll_locked()
                error = hf.get_pll_error()
                delta = hf.get_pll_freq_delta()
                osc_f = hf.get_osc_freq()
                latency = time.perf_counter() - tick
                self.sample_ready.emit(locked, error, delta, osc_f, latency)
                time.sleep(self.period_s)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if hf:
                try:
                    hf.zi.close()
                except Exception:
                    pass


class PlotterWorker(QThread):
    """Buffered demodulator acquisition independent of GUI refresh speed.

    LabOne collects samples at the configured device rate.  This worker polls
    the subscription queue and emits Python batches at a modest rate, avoiding
    one synchronous API round-trip and one Qt signal per raw sample.
    """
    batch_ready = pyqtSignal(object)
    stream_started = pyqtSignal(float, float)  # requested rate, device rate
    failed = pyqtSignal(str)

    def __init__(self, sample_rate_hz=100.0, emit_period_s=0.05, timing_enabled=False):
        super().__init__()
        self.sample_rate_hz = max(1.0, float(sample_rate_hz))
        self.emit_period_s = max(0.01, float(emit_period_s))
        self.timing_enabled = timing_enabled
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        hf = None
        event = None
        sample_path = np_path(f"DEMODS/{HF2_DEMOD}/SAMPLE")
        subscribed = False
        pending = []
        try:
            hf = HFController()
            hf.set_demod_enabled(True)
            hf.set_demod_rate(self.sample_rate_hz)
            actual_rate_hz = float(hf.get_demod_rate())
            if not np.isfinite(actual_rate_hz) or actual_rate_hz <= 0:
                raise ZiAPIError(f"Invalid HF2 demod stream rate: {actual_rate_hz}")
            clockbase_hz = float(hf.get_clockbase())
            if not np.isfinite(clockbase_hz) or clockbase_hz <= 0:
                raise ZiAPIError(f"Invalid HF2 clockbase: {clockbase_hz}")
            event = hf.zi.allocate_event()
            hf.zi.subscribe(sample_path)
            subscribed = True
            self.stream_started.emit(self.sample_rate_hz, actual_rate_hz)
            first_timestamp = None
            last_timestamp = None
            last_emit = time.perf_counter()
            poll_timeout_ms = max(1, min(50, int(round(1000.0 / actual_rate_hz))))
            while not self._stop:
                tick = time.perf_counter()
                _, samples = hf.zi.poll_event(event, poll_timeout_ms)
                poll_latency = time.perf_counter() - tick if self.timing_enabled else float("nan")
                for timestamp, x, y, frequency, _phase, _dio, _trigger, _aux0, _aux1 in samples:
                    if last_timestamp is not None and timestamp <= last_timestamp:
                        continue
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                    t = (timestamp - first_timestamp) / clockbase_hz
                    r = float(np.hypot(x, y))
                    theta = float(np.degrees(np.arctan2(y, x)))
                    pending.append((t, r, theta, frequency, x, y, poll_latency))
                now = time.perf_counter()
                if pending and (now - last_emit >= self.emit_period_s or len(pending) >= actual_rate_hz * self.emit_period_s):
                    self.batch_ready.emit(pending)
                    pending = []
                    last_emit = now
            if pending:
                self.batch_ready.emit(pending)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if hf:
                if subscribed:
                    try:
                        hf.zi.unsubscribe(sample_path)
                    except Exception:
                        pass
                if event:
                    try:
                        hf.zi.deallocate_event(event)
                    except Exception:
                        pass
                try:
                    hf.set_demod_enabled(False)
                except Exception:
                    pass
                try:
                    hf.zi.close()
                except Exception:
                    pass


class LiveParticleAnalysisWorker(QThread):
    """Apply the offline multi-width detector to a bounded live snapshot."""
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, time_s, frequency_hz, settings, thresholds=None, savgol=None):
        super().__init__()
        self.time_s = np.asarray(time_s, dtype=float)
        self.frequency_hz = np.asarray(frequency_hz, dtype=float)
        self.settings = settings
        self.thresholds = thresholds
        self.savgol = savgol or {"enabled": False}

    def run(self):
        try:
            filtered, window_samples = filter_frequency(
                self.time_s, self.frequency_hz, **self.savgol)
            data = pd.DataFrame({"Time_s": self.time_s, "Frequency_Hz": filtered})
            if self.thresholds is None:
                payload = {"mode": "calibration",
                           "thresholds": calibrate_detector(data, self.settings)}
            else:
                payload = {"mode": "detection",
                           "events": detect_calibrated(data, self.settings, self.thresholds),
                           "analyze_end": float(self.settings.analyze_end)}
            payload["savgol_window_samples"] = window_samples
            payload["savgol_effective_window_ms"] = (
                1000.0 * window_samples * float(np.median(np.diff(self.time_s)))
                if window_samples else 0.0)
            self.completed.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class BaselineAnalysisWorker(QThread):
    """Offline-only analysis worker; it never touches HF2LI/DAQ objects."""
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, time_s, frequency_hz, amplitude_v, phase_deg, nominal_interval_s,
                 event_duration_s=0.15, local_baseline_s=0.4, guard_s=0.05, threshold_hz=None,
                 demod_latency_s=None, pll_latency_s=None, rolling_windows=None,
                 calibration_source=None, source_role="particle", detection_method="integral",
                 distance_mm=49.0):
        super().__init__()
        self.time_s = np.asarray(time_s, dtype=float)
        self.frequency_hz = np.asarray(frequency_hz, dtype=float)
        self.amplitude_v = np.asarray(amplitude_v, dtype=float)
        self.phase_deg = np.asarray(phase_deg, dtype=float)
        self.nominal_interval_s = float(nominal_interval_s)
        self.event_duration_s, self.local_baseline_s, self.guard_s = event_duration_s, local_baseline_s, guard_s
        self.threshold_hz = threshold_hz
        self.demod_latency_s = np.asarray(demod_latency_s if demod_latency_s is not None else [], dtype=float)
        self.pll_latency_s = np.asarray(pll_latency_s if pll_latency_s is not None else [], dtype=float)
        self.rolling_windows = tuple(sorted(set(ROLLING_WINDOWS_S) | set(rolling_windows or ())))
        self.calibration_source = calibration_source
        self.source_role = source_role
        self.detection_method = detection_method
        self.distance_mm = float(distance_mm)

    @staticmethod
    def _latency_summary(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values >= 0)]
        if not len(values):
            return {"count": 0, "mean_s": np.nan, "median_s": np.nan, "p95_s": np.nan, "p99_s": np.nan, "max_s": np.nan}
        return {"count": int(len(values)), "mean_s": float(np.mean(values)), "median_s": float(np.median(values)),
                "p95_s": float(np.percentile(values, 95)), "p99_s": float(np.percentile(values, 99)), "max_s": float(np.max(values))}

    def run(self):
        try:
            global_result = linear_baseline(self.time_s, self.frequency_hz)
            detrended = global_result.get("residual_hz") if global_result.get("valid") else None
            result = {
                "timing": timing_quality(self.time_s, self.nominal_interval_s),
                "global": global_result,
                "rolling": {str(window): rolling_metrics(self.time_s, self.frequency_hz, window)
                            for window in self.rolling_windows},
                "event_timescale": {str(window): event_timescale_noise(self.time_s, self.frequency_hz, window)
                                    for window in EVENT_WINDOWS_S if window > 0},
                "correlation": correlations(self.frequency_hz, self.amplitude_v, self.phase_deg, detrended),
                "allan": allan_deviation(self.time_s, self.frequency_hz),
                "transients": transient_flags(self.time_s, self.frequency_hz, self.amplitude_v, self.phase_deg),
                "aux_transients": transient_flags(self.time_s, self.frequency_hz, self.amplitude_v, self.phase_deg, include_frequency=False),
                "read_latency": {"demod_read": self._latency_summary(self.demod_latency_s),
                                 "pll_status_read": self._latency_summary(self.pll_latency_s)},
                "source_role": self.source_role,
            }
            transient = result["aux_transients"]["flag"]
            calibration = None
            if self.calibration_source is not None:
                calibration = calibrate_waveform_thresholds(
                    self.calibration_source["time_s"], self.calibration_source["frequency_hz"],
                    DETECTION_DURATIONS_S, self.local_baseline_s, self.guard_s,
                )
                calibration["source_name"] = self.calibration_source.get("name", "calibration blank")
                calibration["source_path"] = self.calibration_source.get("path", "")
            elif self.threshold_hz is not None:
                calibration = {
                    "thresholds": [{
                        "Duration_s": duration, "PeakThreshold_Hz": float(self.threshold_hz),
                        "IntegralThreshold_Hz_s": float(self.threshold_hz) * duration * 0.5,
                        "TemplateThreshold_Hz": float(self.threshold_hz),
                        "ThresholdMethod": "manual fixed override",
                    } for duration in DETECTION_DURATIONS_S],
                    "source_name": "manual override", "source_path": "",
                    "processing": "manual thresholds; independent calibration not used",
                }
            result["threshold_calibration"] = calibration
            if calibration is None:
                result["particle_candidates"] = {
                    "candidates": [], "all_candidates": [], "excluded_transient": [],
                    "calibration_required": True,
                    "processing": "waveform detection disabled: select an independent calibration blank",
                }
                result["blank_evaluation"] = None
            elif self.source_role == "evaluation_blank":
                evaluation = evaluate_blank(
                    self.time_s, self.frequency_hz, calibration, self.local_baseline_s,
                    self.guard_s, self.detection_method, self.amplitude_v, self.phase_deg,
                    transient, self.distance_mm,
                )
                result["blank_evaluation"] = evaluation
                result["particle_candidates"] = evaluation["detection"]
            else:
                result["particle_candidates"] = detect_waveform_events(
                    self.time_s, self.frequency_hz, calibration["thresholds"],
                    self.local_baseline_s, self.guard_s, self.detection_method,
                    amplitude=self.amplitude_v, phase=self.phase_deg,
                    transient=transient, distance_mm=self.distance_mm,
                )
                result["blank_evaluation"] = None
            median_dt = result["timing"].get("dt_median_s", np.nan)
            samples_per_event = self.event_duration_s / median_dt if np.isfinite(median_dt) and median_dt > 0 else np.nan
            result["event_sampling_adequacy"] = {
                "event_duration_s": float(self.event_duration_s), "median_samples_per_event": float(samples_per_event),
                "assessment": "adequate" if samples_per_event >= 4 else ("limited (<4 samples/window)" if samples_per_event >= 2 else "insufficient (<2 samples/window)"),
                "note": "Uses actual median timestamp spacing; no interpolation or resampling was applied.",
            }
            # Annotate accepted and transient-separated events with timestamp gaps.
            gap_threshold = result["timing"].get("long_gap_threshold_s", np.inf)
            for row in result["particle_candidates"].get("all_candidates", []):
                left = int(np.searchsorted(self.time_s, row["EventStart_s"], side="left"))
                right = int(np.searchsorted(self.time_s, row["EventEnd_s"], side="right"))
                row["TimestampGapFlag"] = bool(np.any(np.diff(self.time_s[max(0, left - 1):right]) > gap_threshold))
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Steel-Tube Resonator Control (HF2LI + NI-DAQ) v{APP_VERSION}")
        self.resize(1550, 950)

        self.hf = None
        self.sweep_worker = None
        self.pll_status_worker = None
        self.plotter_worker = None
        self.plotter_spool = None
        self.calibration_t = []
        self.calibration_f = []
        self._calibration_sample_index = 0
        self.analysis_t, self.analysis_f = [], []
        self._analysis_sample_index = 0
        self._display_sample_index = 0
        self._log_sample_index = 0
        self._recent_raw_dt_s = []
        self._last_raw_t = None
        self.live_particle_worker = None
        self.baseline_analysis_worker = None
        self.baseline_analysis_result = None
        self.baseline_analysis_source = None  # current raw log or imported CSV; never modifies raw data
        self.threshold_calibration_source = None
        self.baseline_source_role = "particle"

        self.sweep_freqs = np.array([])
        self.sweep_R = np.array([])
        self.sweep_theta = np.array([])
        self.sweep_X = np.array([])
        self.sweep_Y = np.array([])
        self.sweep_X_sem = np.array([])
        self.sweep_Y_sem = np.array([])
        self.sweep_metadata = {}
        self._peak_candidates = []  # list of (f0, R0, Q, phase0)
        self.last_resonance_fit = None
        self.pll_calibrated_slope = None
        self.pll_calibrated_setpoint = None
        self.pll_rail_count = 0
        self._latest_pll_locked = False
        self._latest_pll_error_deg = np.nan
        self._latest_pll_correction_hz = np.nan
        self._latest_pll_osc_frequency_hz = np.nan
        self._latest_pll_read_latency_s = np.nan
        self._latest_pll_status_monotonic = np.nan
        self.plotter_actual_rate_hz = np.nan
        self.received_sample_count = 0
        self.ui_refresh_durations_s = []
        self._last_plotter_ui_update_t = -np.inf
        self._last_live_analysis_t = -np.inf
        self.live_particle_thresholds = None
        self.live_particle_events = []
        self.live_particle_last_analyzed_s = 0.0
        self.live_particle_generation = 0
        self.theory_modes = []
        self.theory_lines = []

        # Plotter rolling buffers (all channels, all times share one clock) --
        # trimmed to the display window for live plotting performance.
        self.plot_t = []
        self.plot_R = []
        self.plot_theta = []
        self.plot_f = []
        self.plot_x = []
        self.plot_y = []

        # Bounded recent samples for live analysis. The complete raw session
        # is written to PlotterSpool and exported from that file.
        self.log_t = []
        self.log_R = []
        self.log_theta = []
        self.log_f = []
        self.log_x = []
        self.log_y = []
        self.log_pll_target_phase = []
        self.log_pll_phase_error = []
        self.log_pll_frequency_command = []
        self.log_pll_correction = []
        self.log_pll_locked = []
        self.log_pressure_a = []
        self.log_pressure_b = []
        self.log_flow_direction = []
        self.log_switch_event = []
        self.log_measured_pressure_a = []
        self.log_measured_pressure_b = []
        self.log_temperature_c = []
        self.log_pll_status_age_s = []
        self.measurement_metadata = {}
        self.baseline_active = False
        self._last_baseline_ui_update_t = -np.inf
        self._switch_event_pending = False
        self._last_flow_direction = "Idle"
        self.pressure_commands = {channel: 0.0 for channel in DAQ_CHANNELS.values()}
        self.pressure_worker = None
        self.pressure_sample_time = None
        self.measured_pressure_a = None
        self.measured_pressure_b = None
        self.temperature_c = None

        # Ping-pong (particle repeat-measurement) state machine, driven off
        # the Plotter's own live frequency stream -- see _pingpong_step().
        self.pingpong_active = False
        self.pingpong_state = "baseline"   # 'baseline' | 'transit' | 'settling'
        self.pingpong_baseline_f = None
        self.pingpong_transit_start_t = None
        self.pingpong_transit_min_f = None
        self.pingpong_forward = True
        self.pingpong_cycle_count = 0
        self.pingpong_settle_until_t = None
        self.pingpong_events = []  # list of (t, depth_hz, duration_s)

        self._build_ui()
        self._restore_last_settings()
        self._restore_last_sweep()
        self._install_settings_autosave()
        self._connect_hardware()

    def _settings_widgets(self):
        names = (
            "f_start_spin", "f_stop_spin", "n_points_spin", "sweep_filter_mode_combo",
            "sweep_filter_order_spin", "sweep_fixed_bw_spin", "sweep_max_bw_spin",
            "sweep_suppression_spin", "sweep_overlap_check", "sweep_min_settle_spin",
            "sweep_residual_spin", "sweep_samples_spin", "sweep_avg_time_spin",
            "sweep_avg_tc_spin", "sweep_rate_spin", "pll_center_spin", "pll_input_combo",
            "pll_q_spin", "pll_setpoint_spin", "pll_order_spin", "pll_tc_spin",
            "pll_range_spin", "pll_target_bw_spin", "pll_p_spin", "pll_i_spin",
            "pll_d_spin", "pll_pm_spin", "pll_derivative_check", "pll_integral_check",
            "pll_invert_sign_check",
            "live_savgol_enabled", "live_savgol_window_spin", "live_savgol_order_spin",
        )
        return {name: getattr(self, name) for name in names}

    def _save_last_settings(self):
        values = {}
        for name, widget in self._settings_widgets().items():
            if isinstance(widget, QComboBox):
                values[name] = widget.currentData() if name == "pll_input_combo" else widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            else:
                values[name] = widget.value()
        atomic_json(STATE_DIR / "last_settings.json", {"version": 1, "values": values})

    def _restore_last_settings(self):
        path = STATE_DIR / "last_settings.json"
        if not path.exists():
            return
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get("values", {})
            old_sg_ms = isinstance(values.get("live_savgol_window_spin"), float)
            for name, widget in self._settings_widgets().items():
                if name not in values:
                    continue
                value = values[name]
                if old_sg_ms and name in ("live_savgol_window_spin", "live_savgol_order_spin"):
                    # Earlier versions stored milliseconds and defaulted to
                    # order 2; use the offline defaults for the new units.
                    continue
                if isinstance(widget, QComboBox):
                    if name == "pll_input_combo":
                        index = widget.findData(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    elif widget.findText(value) >= 0:
                        widget.setCurrentText(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(value, (int, float)) and np.isfinite(value):
                    widget.setValue(value)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Last settings could not be restored: {exc}", file=sys.stderr)

    def _install_settings_autosave(self):
        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.timeout.connect(self._save_last_settings)
        for widget in self._settings_widgets().values():
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda *_: self.settings_save_timer.start(500))
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(lambda *_: self.settings_save_timer.start(500))
            else:
                widget.valueChanged.connect(lambda *_: self.settings_save_timer.start(500))

    def _save_last_sweep(self):
        arrays = {key: np.asarray(value) for key, value in self.sweep_metadata.items()}
        if len(arrays.get("freqs", [])):
            atomic_sweep(STATE_DIR / "last_sweep.npz", arrays)

    def _restore_last_sweep(self):
        path = STATE_DIR / "last_sweep.npz"
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                result = {key: data[key].copy() for key in data.files}
            freqs, R, theta = result["freqs"], result["R"], result["theta"]
            self.sweep_freqs, self.sweep_R, self.sweep_theta = freqs, R, theta
            self.sweep_X, self.sweep_Y = result["X"], result["Y"]
            self.sweep_X_sem, self.sweep_Y_sem = result["X_sem"], result["Y_sem"]
            self.sweep_metadata = result
            self.sweep_amp_curve.setData(freqs, R)
            self.sweep_phase_curve.setData(freqs, theta)
            self.detect_peaks(freqs, R, theta)
        except (OSError, KeyError, ValueError) as exc:
            print(f"Last sweep could not be restored: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left control column (scrollable so it never gets clipped by
        # screen height, regardless of how many groups are stacked) ----
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(500)
        left_scroll.setMaximumWidth(700)

        self.status_label = QLabel("연결 안 됨")
        self.status_label.setStyleSheet("font-weight: bold;")
        left.addWidget(self.status_label)

        left.addWidget(self._build_drive_group())
        left.addWidget(self._build_signal_input_group())
        left.addWidget(CollapsibleSection(
            "이론 모드 — 양단 고정 금속관", self._build_theory_group(), expanded=False,
        ))
        left.addWidget(self._build_sweep_group())
        left.addWidget(self._build_peak_group())
        left.addWidget(self._build_pll_group())
        left.addWidget(CollapsibleSection(
            "5. 실험조건 metadata", self._build_experiment_metadata_group(), expanded=False,
        ))
        left.addWidget(self._build_plotter_group())
        left.addWidget(self._build_live_particle_group())
        left.addWidget(self._build_daq_group())
        left.addWidget(self._build_pingpong_group())
        left.addStretch(1)

        emergency_btn = QPushButton("⚠ 긴급 정지 (모든 출력 0)")
        emergency_btn.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: bold; padding: 10px;"
        )
        emergency_btn.clicked.connect(self.emergency_stop)
        left.addWidget(emergency_btn)

        # ---- right plot column (tabs: Sweeper / PLL / Plotter) ----
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sweep_tab(), "① Sweeper (Amp/Phase vs Freq)")
        self.tabs.addTab(self._build_pll_tab(), "② PLL (Response)")
        self.tabs.addTab(self._build_plotter_tab(), "③ Plotter (Freq & R vs Time)")
        self.tabs.addTab(self._build_live_particle_tab(), "④ 실시간 파티클 검출")
        self.tabs.addTab(self._build_baseline_tab(), "⑤ Baseline Analysis")
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([540, 1410])
        root.addWidget(splitter)

    # --- Sweeper tab ---
    def _build_sweep_tab(self):
        w = pg.GraphicsLayoutWidget()
        w.setBackground('w')
        self.sweep_amp_plot = w.addPlot(row=0, col=0)
        self.sweep_amp_plot.setLabel('left', 'Amplitude R (V)')
        self.sweep_amp_plot.showGrid(x=True, y=True)
        self.sweep_amp_curve = self.sweep_amp_plot.plot(pen=pg.mkPen('b', width=2))
        self.sweep_fit_amp_curve = self.sweep_amp_plot.plot(pen=pg.mkPen('#8e44ad', width=2, style=Qt.DashLine))
        self.peak_markers = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('r'), pen=pg.mkPen('k'))
        self.peak_markers.sigClicked.connect(self.on_peak_marker_clicked)
        self.sweep_amp_plot.addItem(self.peak_markers)

        w.nextRow()
        self.sweep_phase_plot = w.addPlot(row=1, col=0)
        self.sweep_phase_plot.setLabel('left', 'Phase (deg)')
        self.sweep_phase_plot.setLabel('bottom', 'Frequency (Hz)')
        self.sweep_phase_plot.showGrid(x=True, y=True)
        self.sweep_phase_curve = self.sweep_phase_plot.plot(pen=pg.mkPen('g', width=2))
        self.sweep_fit_phase_curve = self.sweep_phase_plot.plot(pen=pg.mkPen('#8e44ad', width=2, style=Qt.DashLine))
        self.sweep_phase_plot.setXLink(self.sweep_amp_plot)
        return w

    # --- PLL tab: parameters live in the left panel; this tab is just the
    # live "Response" read-back (Error / Freq Shift / Locked), no graph. ---
    def _build_pll_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addStretch(1)

        self.pll_big_locked = QLabel("PLL: 비활성")
        self.pll_big_locked.setAlignment(Qt.AlignCenter)
        self.pll_big_locked.setStyleSheet("font-size: 32px; font-weight: bold;")
        layout.addWidget(self.pll_big_locked)

        grid = QGridLayout()
        grid.addWidget(QLabel("Error (deg):"), 0, 0)
        self.pll_big_error = QLabel("-")
        self.pll_big_error.setStyleSheet("font-size: 20px;")
        grid.addWidget(self.pll_big_error, 0, 1)

        grid.addWidget(QLabel("Freq Shift (Hz):"), 1, 0)
        self.pll_big_shift = QLabel("-")
        self.pll_big_shift.setStyleSheet("font-size: 20px;")
        grid.addWidget(self.pll_big_shift, 1, 1)

        grid.addWidget(QLabel("Oscillator Freq (Hz):"), 2, 0)
        self.pll_big_freq = QLabel("-")
        self.pll_big_freq.setStyleSheet("font-size: 20px;")
        grid.addWidget(self.pll_big_freq, 2, 1)

        wrap = QWidget()
        wrap.setLayout(grid)
        layout.addWidget(wrap, 0, Qt.AlignCenter)

        hint = QLabel(
            "PLL 탭은 파라미터 제어 결과(Response)만 보여줍니다.\n"
            "주파수/진폭을 시간에 따라 관찰하려면 ③ Plotter 탭을 사용하세요."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)
        layout.addStretch(1)
        return w

    # --- Plotter tab: Frequency + R together, dual Y axis, vs time ---
    def _build_plotter_tab(self):
        w = pg.GraphicsLayoutWidget()
        w.setBackground('w')
        self.plotter_plot = w.addPlot(row=0, col=0)
        self.plotter_plot.setLabel('bottom', 'Time (s)')
        self.plotter_plot.setLabel('left', 'Frequency (Hz)', color='#0000ff')
        self.plotter_plot.getAxis('left').setPen('b')
        self.plotter_plot.getAxis('left').setTextPen('b')
        self.plotter_plot.showGrid(x=True, y=True)
        self.plotter_freq_curve = self.plotter_plot.plot(pen=pg.mkPen('b', width=1.5), name='Sample Frequency')

        self.plotter_vb2 = setup_dual_axis(self.plotter_plot)
        self.plotter_plot.getAxis('right').setLabel('Amplitude R (V)', color='#cc0000')
        self.plotter_plot.getAxis('right').setPen('r')
        self.plotter_plot.getAxis('right').setTextPen('r')
        self.plotter_r_curve = pg.PlotCurveItem(pen=pg.mkPen('r', width=1.5))
        self.plotter_vb2.addItem(self.plotter_r_curve)

        # Marks where the ping-pong state machine detected a particle transit.
        self.pingpong_markers = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('orange'), pen=pg.mkPen('k'))
        self.plotter_plot.addItem(self.pingpong_markers)

        legend = self.plotter_plot.addLegend(offset=(60, 10))
        legend.addItem(self.plotter_freq_curve, "Sample Frequency")
        legend.addItem(self.plotter_r_curve, "Sample R")

        w.nextRow()
        self.plotter_hist_plot = w.addPlot(row=1, col=0)
        self.plotter_hist_plot.setLabel('bottom', 'Value')
        self.plotter_hist_plot.setLabel('left', 'Count')
        self.plotter_hist_plot.showGrid(x=True, y=True)
        self.plotter_hist_item = pg.BarGraphItem(x=[0], height=[0], width=1, brush='b')
        self.plotter_hist_plot.addItem(self.plotter_hist_item)
        return w

    def _build_live_particle_tab(self):
        """Demo-oriented live view: status cards, raw frequency and candidates."""
        page = QWidget()
        page.setStyleSheet("background: #f4f7fb;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("REAL-TIME PARTICLE DETECTION")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #14213d;")
        subtitle = QLabel("PLL frequency · offline multi-width algorithm")
        subtitle.setStyleSheet("color: #60708a; font-size: 12px;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(subtitle)
        layout.addLayout(title_row)

        cards = QHBoxLayout()
        self.live_state_value = QLabel("대기")
        self.live_count_value = QLabel("0")
        self.live_latest_value = QLabel("—")
        self.live_frequency_value = QLabel("—")
        for caption, value, accent in (
            ("DETECTOR", self.live_state_value, "#147d64"),
            ("PARTICLE CANDIDATES", self.live_count_value, "#1b65d6"),
            ("LATEST DEPTH", self.live_latest_value, "#e07a1f"),
            ("PLL FREQUENCY", self.live_frequency_value, "#6f42c1"),
        ):
            card = QWidget()
            card.setStyleSheet(
                "QWidget {background: white; border: 1px solid #dfe6ef; border-radius: 10px;}"
                "QLabel {border: none; background: transparent;}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            label = QLabel(caption)
            label.setStyleSheet("color: #718096; font-size: 10px; font-weight: 700;")
            value.setStyleSheet(f"color: {accent}; font-size: 23px; font-weight: 800;")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            cards.addWidget(card, 1)
        layout.addLayout(cards)

        self.live_calibration_progress = QProgressBar()
        self.live_calibration_progress.setRange(0, 100)
        self.live_calibration_progress.setValue(0)
        self.live_calibration_progress.setFormat("검출 준비 대기")
        self.live_calibration_progress.setStyleSheet(
            "QProgressBar {background: #e6ebf2; border: none; border-radius: 5px; height: 10px; color: #445; text-align: center;}"
            "QProgressBar::chunk {background: #22a884; border-radius: 5px;}"
        )
        layout.addWidget(self.live_calibration_progress)

        self.live_particle_plot = pg.PlotWidget(background='w')
        self.live_particle_plot.setLabel('bottom', 'Time (s)')
        self.live_particle_plot.setLabel('left', 'PLL Frequency (Hz)')
        self.live_particle_plot.showGrid(x=True, y=True, alpha=.25)
        self.live_particle_plot.setMinimumHeight(360)
        self.live_frequency_curve = self.live_particle_plot.plot(
            pen=pg.mkPen('#1b65d6', width=2), name='Raw frequency')
        self.live_filtered_frequency_curve = self.live_particle_plot.plot(
            pen=pg.mkPen('#d13b73', width=2), name='SG filtered')
        self.live_filtered_frequency_curve.setVisible(False)
        live_legend = self.live_particle_plot.addLegend(offset=(12, 12))
        live_legend.addItem(self.live_frequency_curve, 'Raw frequency')
        live_legend.addItem(self.live_filtered_frequency_curve, 'SG filtered')
        self.live_particle_markers = pg.ScatterPlotItem(
            size=15, symbol='t', brush=pg.mkBrush('#ff8a00'),
            pen=pg.mkPen('#9a4c00', width=1.5), name='Detected candidate')
        self.live_particle_plot.addItem(self.live_particle_markers)
        layout.addWidget(self.live_particle_plot, 1)

        self.live_event_list = QListWidget()
        self.live_event_list.setMaximumHeight(135)
        self.live_event_list.setAlternatingRowColors(True)
        self.live_event_list.setStyleSheet(
            "QListWidget {background: white; border: 1px solid #dfe6ef; border-radius: 8px; padding: 4px;}"
        )
        self.live_event_list.addItem("검출된 후보가 여기에 실시간으로 표시됩니다.")
        layout.addWidget(self.live_event_list)
        return page

    def _build_baseline_tab(self):
        """Raw-preserving baseline diagnostics, updated only in baseline mode."""
        w = pg.GraphicsLayoutWidget()
        w.setBackground('w')

        self.baseline_raw_plot = w.addPlot(row=0, col=0)
        self.baseline_raw_plot.setLabel('bottom', 'Time (s)')
        self.baseline_raw_plot.setLabel('left', 'Frequency (Hz)')
        self.baseline_raw_plot.showGrid(x=True, y=True)
        self.baseline_raw_curve = self.baseline_raw_plot.plot(pen=pg.mkPen('#0066cc', width=1), name='Raw Frequency')
        self.baseline_trend_curve = self.baseline_raw_plot.plot(pen=pg.mkPen('#d62728', width=2), name='Linear Trend')
        legend = self.baseline_raw_plot.addLegend(offset=(60, 10))
        legend.addItem(self.baseline_raw_curve, 'Raw Frequency')
        legend.addItem(self.baseline_trend_curve, 'Linear Trend')

        self.baseline_residual_plot = w.addPlot(row=1, col=0)
        self.baseline_residual_plot.setLabel('bottom', 'Time (s)')
        self.baseline_residual_plot.setLabel('left', 'Detrended residual (Hz)')
        self.baseline_residual_plot.showGrid(x=True, y=True)
        self.baseline_residual_curve = self.baseline_residual_plot.plot(pen=pg.mkPen('#6a0dad', width=1))

        self.baseline_amplitude_plot = w.addPlot(row=2, col=0)
        self.baseline_amplitude_plot.setLabel('bottom', 'Time (s)')
        self.baseline_amplitude_plot.setLabel('left', 'Amplitude R (V)')
        self.baseline_amplitude_plot.showGrid(x=True, y=True)
        self.baseline_amplitude_curve = self.baseline_amplitude_plot.plot(pen=pg.mkPen('#d62728', width=1))

        self.baseline_error_plot = w.addPlot(row=3, col=0)
        self.baseline_error_plot.setLabel('bottom', 'Time (s)')
        self.baseline_error_plot.setLabel('left', 'PLL phase error (deg)')
        self.baseline_error_plot.showGrid(x=True, y=True)
        self.baseline_error_curve = self.baseline_error_plot.plot(pen=pg.mkPen('#008000', width=1))

        self.baseline_rolling_plot = w.addPlot(row=4, col=0)
        self.baseline_rolling_plot.setLabel('bottom', 'Time (s)')
        self.baseline_rolling_plot.setLabel('left', 'Rolling detrended σ (Hz)')
        self.baseline_rolling_plot.showGrid(x=True, y=True)
        self.baseline_rolling_curve = self.baseline_rolling_plot.plot(pen=pg.mkPen('#ff7f0e', width=1))

        self.baseline_allan_plot = w.addPlot(row=5, col=0)
        self.baseline_allan_plot.setLabel('bottom', 'Averaging time τ (s)')
        self.baseline_allan_plot.setLabel('left', 'Allan deviation (Hz)')
        self.baseline_allan_plot.showGrid(x=True, y=True)
        self.baseline_allan_plot.setLogMode(x=True, y=True)
        self.baseline_allan_curve = self.baseline_allan_plot.plot(symbol='o', pen=pg.mkPen('#444444', width=1))
        return w

    # ------------------------------------------------------------------
    def _build_drive_group(self):
        g = QGroupBox("1. 구동 피에조 (HF2LI Signal Output)")
        layout = QGridLayout(g)

        range_label = QLabel("Range:")
        range_label.setToolTip(
            "AMPLITUDES 노드는 Range에 대한 fraction(0~1)이라, Range를 바꾸면\n"
            "같은 Amp(Vpk)라도 실제 분해능이 달라집니다. 여기서 바꾸면\n"
            "현재 Amp(Vpk) 값을 유지하도록 자동으로 다시 씁니다."
        )
        layout.addWidget(range_label, 0, 0)
        self.range_combo = QComboBox()
        for label, _ in OUTPUT_RANGE_OPTIONS:
            self.range_combo.addItem(label)
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        self.range_combo.setCurrentIndex(3)  # LabOne reference configuration: 10 V range
        layout.addWidget(self.range_combo, 0, 1)

        layout.addWidget(QLabel(f"Amp (0 ~ {MAX_DRIVE_V:g} Vpk):"), 1, 0)
        self.drive_v_spin = QDoubleSpinBox()
        self.drive_v_spin.setRange(0.0, MAX_DRIVE_V)
        self.drive_v_spin.setDecimals(3)
        self.drive_v_spin.setSingleStep(0.05)
        self.drive_v_spin.setValue(10.0)  # reference configuration; output remains safely disabled at startup
        self.drive_v_spin.valueChanged.connect(self.set_drive_amplitude_value)
        layout.addWidget(self.drive_v_spin, 1, 1)

        # LabOne's Signal Output tab has independent "On" and "Amp" toggles
        # (see its screenshot: two separate blue circles) -- neither implies
        # the other, so both are exposed here the same way.
        self.output_on_btn = QPushButton("On")
        self.output_on_btn.setCheckable(True)
        self.output_on_btn.toggled.connect(self.toggle_output)
        layout.addWidget(self.output_on_btn, 2, 0)

        self.amp_enable_btn = QPushButton("Amp")
        self.amp_enable_btn.setCheckable(True)
        self.amp_enable_btn.toggled.connect(self.toggle_amp_enabled)
        layout.addWidget(self.amp_enable_btn, 2, 1)

        return g

    def _build_signal_input_group(self):
        g = QGroupBox("신호 입력 (HF2LI Signal Inputs)")
        layout = QGridLayout(g)

        note = QLabel("기본값: Input 1=1.3 V, Input 2=1.1 V, AC=ON, 50 Ω/Diff=OFF.")
        note.setWordWrap(True)
        layout.addWidget(note, 0, 0, 1, 4)
        self.input_range_spins, self.input_ac_btns, self.input_50_btns, self.input_diff_btns = [], [], [], []
        for input_index, default_range in enumerate((1.3, 1.1)):
            row = input_index + 1
            layout.addWidget(QLabel(f"Input {input_index + 1} Range (V):"), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1.5)
            spin.setDecimals(3)
            spin.setValue(default_range)
            spin.valueChanged.connect(lambda value, i=input_index: self.set_input_range_value(i, value))
            layout.addWidget(spin, row, 1)
            self.input_range_spins.append(spin)

            ac = QPushButton("AC")
            ac.setCheckable(True)
            ac.setChecked(True)
            ac.setText("AC ✓")
            ac.toggled.connect(lambda checked, i=input_index: self.toggle_input_ac(i, checked))
            layout.addWidget(ac, row, 2)
            self.input_ac_btns.append(ac)

            imp = QPushButton("50 Ω")
            imp.setCheckable(True)
            imp.toggled.connect(lambda checked, i=input_index: self.toggle_input_50_ohm(i, checked))
            layout.addWidget(imp, row, 3)
            self.input_50_btns.append(imp)

            diff = QPushButton("Diff")
            diff.setCheckable(True)
            diff.setToolTip("OFF=single-ended, ON=differential")
            diff.toggled.connect(lambda checked, i=input_index: self.toggle_input_differential(i, checked))
            layout.addWidget(diff, row, 4)
            self.input_diff_btns.append(diff)
        self.input1_diff_btn, self.input2_diff_btn = self.input_diff_btns

        return g

    def _build_theory_group(self):
        g = QGroupBox("이론 모드 — 양단 고정 금속관 (Euler–Bernoulli)")
        layout = QGridLayout(g)
        fields = [
            ("길이 L (mm):", "theory_length_spin", 55.0, 0.1, 10000.0, 3),
            ("외경 Do (mm):", "theory_outer_spin", 0.8, 0.001, 1000.0, 4),
            ("내경 Di (mm):", "theory_inner_spin", 0.6, 0.0, 999.0, 4),
            ("Young률 E (GPa):", "theory_e_spin", 193.0, 0.001, 10000.0, 3),
            ("관 밀도 (kg/m³):", "theory_density_spin", 7930.0, 1.0, 50000.0, 1),
            ("내부 유체 밀도 (kg/m³):", "theory_fluid_density_spin", 997.0, 0.0, 50000.0, 1),
        ]
        for row, (label, attr, value, minimum, maximum, decimals) in enumerate(fields):
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setValue(value)
            setattr(self, attr, spin)
            layout.addWidget(spin, row, 1)
        row = len(fields)
        layout.addWidget(QLabel("계산 모드 수:"), row, 0)
        self.theory_count_spin = QSpinBox()
        self.theory_count_spin.setRange(1, 30)
        self.theory_count_spin.setValue(6)
        layout.addWidget(self.theory_count_spin, row, 1)
        row += 1
        self.theory_show_check = QCheckBox("스윕 그래프에 이론 모드 표시")
        self.theory_show_check.setChecked(True)
        self.theory_show_check.toggled.connect(self.update_theory_lines)
        layout.addWidget(self.theory_show_check, row, 0, 1, 2)
        row += 1
        calc = QPushButton("이론 모드 계산")
        calc.clicked.connect(self.calculate_theory_modes)
        layout.addWidget(calc, row, 0, 1, 2)
        row += 1
        self.theory_mode_list = QListWidget()
        self.theory_mode_list.setMaximumHeight(120)
        layout.addWidget(self.theory_mode_list, row, 0, 1, 2)
        row += 1
        use = QPushButton("선택 모드 주변으로 스윕 범위 설정")
        use.clicked.connect(self.use_selected_theory_mode)
        layout.addWidget(use, row, 0, 1, 2)
        return g

    def _build_sweep_group(self):
        g = QGroupBox("2. Sweeper — 주파수 스윕")
        layout = QGridLayout(g)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("시작 주파수 (Hz):"), 0, 0)
        self.f_start_spin = QDoubleSpinBox()
        self.f_start_spin.setRange(1, 5e7)
        self.f_start_spin.setDecimals(1)
        self.f_start_spin.setValue(5000)
        layout.addWidget(self.f_start_spin, 0, 1)

        layout.addWidget(QLabel("끝 주파수 (Hz):"), 1, 0)
        self.f_stop_spin = QDoubleSpinBox()
        self.f_stop_spin.setRange(1, 5e7)
        self.f_stop_spin.setDecimals(1)
        self.f_stop_spin.setValue(35000)
        layout.addWidget(self.f_stop_spin, 1, 1)

        layout.addWidget(QLabel("포인트 수:"), 2, 0)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 20000)
        self.n_points_spin.setValue(500)
        layout.addWidget(self.n_points_spin, 2, 1)

        advanced = QWidget()
        advanced_layout = QGridLayout(advanced)
        advanced_layout.setContentsMargins(6, 4, 6, 4)
        advanced_layout.setColumnStretch(1, 1)

        advanced_layout.addWidget(QLabel("필터 모드:"), 0, 0)
        self.sweep_filter_mode_combo = QComboBox()
        self.sweep_filter_mode_combo.addItems(["Manual", "Fixed", "Auto"])
        self.sweep_filter_mode_combo.setCurrentText("Auto")
        self.sweep_filter_mode_combo.setToolTip("Manual은 현재 Lock-in 필터를 유지합니다. Fixed/Auto는 스윕 중에만 적용 후 복원합니다.")
        advanced_layout.addWidget(self.sweep_filter_mode_combo, 0, 1)

        advanced_layout.addWidget(QLabel("필터 차수 (1–8):"), 1, 0)
        self.sweep_filter_order_spin = QSpinBox()
        self.sweep_filter_order_spin.setRange(1, 8)
        self.sweep_filter_order_spin.setValue(4)
        advanced_layout.addWidget(self.sweep_filter_order_spin, 1, 1)

        advanced_layout.addWidget(QLabel("Fixed BW 3dB (Hz):"), 2, 0)
        self.sweep_fixed_bw_spin = QDoubleSpinBox()
        self.sweep_fixed_bw_spin.setRange(0.001, 200000)
        self.sweep_fixed_bw_spin.setValue(100.0)
        advanced_layout.addWidget(self.sweep_fixed_bw_spin, 2, 1)

        advanced_layout.addWidget(QLabel("Auto 최대 BW (Hz):"), 3, 0)
        self.sweep_max_bw_spin = QDoubleSpinBox()
        self.sweep_max_bw_spin.setRange(0.001, 1.25e6)
        self.sweep_max_bw_spin.setValue(1.25e6)
        advanced_layout.addWidget(self.sweep_max_bw_spin, 3, 1)

        advanced_layout.addWidget(QLabel("ω/2ω 억제 (dB):"), 4, 0)
        self.sweep_suppression_spin = QDoubleSpinBox()
        self.sweep_suppression_spin.setRange(0, 160)
        self.sweep_suppression_spin.setValue(40.0)
        advanced_layout.addWidget(self.sweep_suppression_spin, 4, 1)

        self.sweep_overlap_check = QCheckBox("BW overlap 허용")
        self.sweep_overlap_check.setChecked(False)
        advanced_layout.addWidget(self.sweep_overlap_check, 5, 0, 1, 2)

        advanced_layout.addWidget(QLabel("최소 정착시간 (s):"), 6, 0)
        self.sweep_min_settle_spin = QDoubleSpinBox()
        self.sweep_min_settle_spin.setRange(0.0, 100.0)
        self.sweep_min_settle_spin.setDecimals(4)
        self.sweep_min_settle_spin.setValue(0.0)
        advanced_layout.addWidget(self.sweep_min_settle_spin, 6, 1)

        advanced_layout.addWidget(QLabel("정착 잔류오차:"), 7, 0)
        self.sweep_residual_spin = QDoubleSpinBox()
        self.sweep_residual_spin.setRange(1e-10, 0.1)
        self.sweep_residual_spin.setDecimals(8)
        self.sweep_residual_spin.setValue(0.0001)
        advanced_layout.addWidget(self.sweep_residual_spin, 7, 1)

        advanced_layout.addWidget(QLabel("평균 최소 샘플 수:"), 8, 0)
        self.sweep_samples_spin = QSpinBox()
        self.sweep_samples_spin.setRange(1, 100000)
        self.sweep_samples_spin.setValue(1)
        advanced_layout.addWidget(self.sweep_samples_spin, 8, 1)

        advanced_layout.addWidget(QLabel("평균 최소 시간 (s):"), 9, 0)
        self.sweep_avg_time_spin = QDoubleSpinBox()
        self.sweep_avg_time_spin.setRange(0.0, 100.0)
        self.sweep_avg_time_spin.setDecimals(4)
        self.sweep_avg_time_spin.setValue(0.0)
        advanced_layout.addWidget(self.sweep_avg_time_spin, 9, 1)

        advanced_layout.addWidget(QLabel("평균 TC 수:"), 10, 0)
        self.sweep_avg_tc_spin = QDoubleSpinBox()
        self.sweep_avg_tc_spin.setRange(0.0, 1000.0)
        self.sweep_avg_tc_spin.setValue(0.0)
        advanced_layout.addWidget(self.sweep_avg_tc_spin, 10, 1)

        advanced_layout.addWidget(QLabel("샘플 rate (Sa/s):"), 11, 0)
        self.sweep_rate_spin = QDoubleSpinBox()
        self.sweep_rate_spin.setRange(1.0, 410000.0)
        self.sweep_rate_spin.setValue(224.9)
        advanced_layout.addWidget(self.sweep_rate_spin, 11, 1)

        layout.addWidget(CollapsibleSection(
            "고급 스윕 설정", advanced, expanded=False,
        ), 3, 0, 1, 2)

        self.sweep_progress = QProgressBar()
        layout.addWidget(self.sweep_progress, 4, 0, 1, 2)

        btn_row = QHBoxLayout()
        self.run_sweep_btn = QPushButton("스윕 시작")
        self.run_sweep_btn.clicked.connect(self.run_sweep)
        self.stop_sweep_btn = QPushButton("스윕 중지")
        self.stop_sweep_btn.setEnabled(False)
        self.stop_sweep_btn.clicked.connect(self.stop_sweep)
        btn_row.addWidget(self.run_sweep_btn)
        btn_row.addWidget(self.stop_sweep_btn)
        layout.addLayout(btn_row, 5, 0, 1, 2)

        return g

    def _build_peak_group(self):
        g = QGroupBox("3. Sweeper Math — 공진 피크 (Q / Center / Phase)")
        layout = QVBoxLayout(g)
        self.peak_list = QListWidget()
        self.peak_list.setMaximumHeight(160)
        self.peak_list.currentRowChanged.connect(self.on_peak_list_row_changed)
        layout.addWidget(self.peak_list)
        hint = QLabel("그래프의 빨간 점을 직접 클릭해도 선택됩니다.")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)
        self.lock_peak_btn = QPushButton("선택한 피크를 PLL로 적용")
        self.lock_peak_btn.clicked.connect(lambda: self.apply_peak(self.peak_list.currentRow(), show_msg=True))
        layout.addWidget(self.lock_peak_btn)
        fit_btn = QPushButton("선택 피크 복소 공진 fitting")
        fit_btn.clicked.connect(lambda: self.fit_selected_peak(self.peak_list.currentRow(), show_msg=True))
        layout.addWidget(fit_btn)
        self.fit_result_label = QLabel("Fitting: -")
        self.fit_result_label.setWordWrap(True)
        self.fit_result_label.setStyleSheet("color: #555;")
        layout.addWidget(self.fit_result_label)
        save_sweep_btn = QPushButton("스윕 데이터 + 피크 CSV로 저장")
        save_sweep_btn.clicked.connect(self.export_sweep_csv)
        layout.addWidget(save_sweep_btn)
        return g

    def _build_pll_group(self):
        g = QGroupBox("4. PLL — 파라미터")
        layout = QGridLayout(g)
        row = 0

        layout.addWidget(QLabel("중심 주파수 f0 (Hz):"), row, 0)
        self.pll_center_spin = QDoubleSpinBox()
        self.pll_center_spin.setRange(1, 5e7)
        self.pll_center_spin.setDecimals(1)
        layout.addWidget(self.pll_center_spin, row, 1)
        row += 1

        input_label = QLabel("PLL 입력:")
        input_label.setToolTip("HF2LI PLL 1의 내부 phase detector (demodulator 7)에 연결할 실제 검출 신호입니다.")
        layout.addWidget(input_label, row, 0)
        self.pll_input_combo = QComboBox()
        for label, value in [
            ("Signal Input 1 (sense piezo)", 0), ("Signal Input 2", 1),
            ("Aux Input 1", 2), ("Aux Input 2", 3), ("DIO 0", 4), ("DIO 1", 5),
        ]:
            self.pll_input_combo.addItem(label, value)
        self.pll_input_combo.setCurrentIndex(0)
        layout.addWidget(self.pll_input_combo, row, 1)
        row += 1

        q_label = QLabel("품질계수 Q:")
        q_label.setToolTip("Sweeper 피크를 클릭하면 자동 입력됩니다")
        layout.addWidget(q_label, row, 0)
        self.pll_q_spin = QDoubleSpinBox()
        self.pll_q_spin.setRange(1, 1e7)
        self.pll_q_spin.setDecimals(1)
        self.pll_q_spin.setValue(100)
        layout.addWidget(self.pll_q_spin, row, 1)
        row += 1

        sp_label = QLabel("Setpoint (deg):")
        sp_label.setToolTip("Sweeper 피크의 위상값에서 자동 입력됩니다")
        layout.addWidget(sp_label, row, 0)
        self.pll_setpoint_spin = QDoubleSpinBox()
        self.pll_setpoint_spin.setRange(-180, 180)
        self.pll_setpoint_spin.setValue(0.0)
        layout.addWidget(self.pll_setpoint_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("차수 (order):"), row, 0)
        self.pll_order_spin = QSpinBox()
        self.pll_order_spin.setRange(1, 8)
        self.pll_order_spin.setValue(2)
        layout.addWidget(self.pll_order_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("시상수 TC (s):"), row, 0)
        self.pll_tc_spin = QDoubleSpinBox()
        self.pll_tc_spin.setRange(1e-6, 1.0)
        self.pll_tc_spin.setDecimals(6)
        self.pll_tc_spin.setValue(0.002)
        layout.addWidget(self.pll_tc_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("캡처 범위 (Hz):"), row, 0)
        self.pll_range_spin = QDoubleSpinBox()
        self.pll_range_spin.setRange(1, 1e6)
        self.pll_range_spin.setValue(500)
        layout.addWidget(self.pll_range_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Target BW (Hz):"), row, 0)
        self.pll_target_bw_spin = QDoubleSpinBox()
        self.pll_target_bw_spin.setRange(0.01, 50000)
        self.pll_target_bw_spin.setValue(3.0)
        layout.addWidget(self.pll_target_bw_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("P (Hz/deg):"), row, 0)
        self.pll_p_spin = QDoubleSpinBox()
        self.pll_p_spin.setRange(-1e9, 1e9)
        self.pll_p_spin.setDecimals(4)
        layout.addWidget(self.pll_p_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("I (Hz²/deg):"), row, 0)
        self.pll_i_spin = QDoubleSpinBox()
        self.pll_i_spin.setRange(-1e12, 1e12)
        self.pll_i_spin.setDecimals(2)
        self.pll_i_spin.setValue(0.0)
        layout.addWidget(self.pll_i_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("D (1/deg):"), row, 0)
        self.pll_d_spin = QDoubleSpinBox()
        self.pll_d_spin.setRange(-1e6, 1e6)
        self.pll_d_spin.setDecimals(6)
        self.pll_d_spin.setValue(0.0)
        layout.addWidget(self.pll_d_spin, row, 1)
        row += 1

        self.pll_derivative_check = QCheckBox("D 항까지 수치 최적화 (고급)")
        self.pll_derivative_check.setToolTip("기본 PI 설계가 노이즈에 더 강합니다. D는 모델상 이득이 있을 때만 사용하세요.")
        layout.addWidget(self.pll_derivative_check, row, 0, 1, 2)
        row += 1

        self.pll_integral_check = QCheckBox("I 항 최적화 (P-only 잠금 확인 후)")
        self.pll_integral_check.setToolTip("기본값은 I=0입니다. 안정된 P-only lock을 먼저 확인한 뒤에만 켜세요.")
        layout.addWidget(self.pll_integral_check, row, 0, 1, 2)
        row += 1

        self.pll_invert_sign_check = QCheckBox("P/I/D 부호 반전 (실제 sign test 후에만)")
        self.pll_invert_sign_check.setToolTip("실제 PLL phase detector의 보정 결과와 lock 반응이 반대일 때만 사용하세요.")
        layout.addWidget(self.pll_invert_sign_check, row, 0, 1, 2)
        row += 1

        layout.addWidget(QLabel("최소 위상여유 (deg):"), row, 0)
        self.pll_pm_spin = QDoubleSpinBox()
        self.pll_pm_spin.setRange(30.0, 85.0)
        self.pll_pm_spin.setValue(60.0)
        layout.addWidget(self.pll_pm_spin, row, 1)
        row += 1

        estimate_btn = QPushButton("fitted 공진기 모델로 PID 최적화")
        estimate_btn.clicked.connect(self.estimate_pll_p)
        layout.addWidget(estimate_btn, row, 0, 1, 2)
        row += 1

        calibrate_btn = QPushButton("실제 PLL phase detector 보정 (f0 주변 스캔)")
        calibrate_btn.setToolTip("PLL의 ERROR 노드로 실제 내부 phase detector의 위상 기울기와 resonance setpoint를 측정합니다.")
        calibrate_btn.clicked.connect(self.calibrate_pll_phase_detector)
        layout.addWidget(calibrate_btn, row, 0, 1, 2)
        row += 1

        sign_test_btn = QPushButton("안전 P-only feedback sign test (+P / −P)")
        sign_test_btn.setToolTip("작은 PLL range 안에서 +P와 -P의 오차 RMS를 짧게 비교합니다. 구동이 켜진 상태에서만 실행됩니다.")
        sign_test_btn.clicked.connect(self.test_pll_feedback_sign)
        layout.addWidget(sign_test_btn, row, 0, 1, 2)
        row += 1

        hint = QLabel(
            "※ 선택 피크의 복소 2차 공진 fitting으로 plant를 만들고,\n"
            "   공진 ring-down·PLL demod 필터·loop rate·위상여유를 포함한\n"
            "   폐루프 모델을 수치 최적화합니다. 장비에 쓰기 전 결과를 확인하세요."
        )
        hint.setStyleSheet("color: gray; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint, row, 0, 1, 2)
        row += 1

        self.pll_design_label = QLabel("PID 설계: 아직 fitting된 공진기가 없습니다.")
        self.pll_design_label.setWordWrap(True)
        self.pll_design_label.setStyleSheet("color: #555;")
        layout.addWidget(self.pll_design_label, row, 0, 1, 2)
        row += 1

        self.pll_locked_label = QLabel("PLL: 비활성")
        self.pll_locked_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.pll_locked_label, row, 0, 1, 2)
        row += 1

        self.pll_enable_btn = QPushButton("PLL 활성화")
        self.pll_enable_btn.setCheckable(True)
        self.pll_enable_btn.toggled.connect(self.toggle_pll)
        layout.addWidget(self.pll_enable_btn, row, 0, 1, 2)

        return g

    def _build_experiment_metadata_group(self):
        g = QGroupBox("5. 실험조건 metadata")
        layout = QGridLayout(g)

        def line(row, label, attr, default=""):
            layout.addWidget(QLabel(label), row, 0)
            widget = QLineEdit(default)
            setattr(self, attr, widget)
            layout.addWidget(widget, row, 1)

        line(0, "Mounting:", "metadata_mounting_edit", "nylon-tip / direct piezo")
        line(1, "Clamp torque/contact:", "metadata_clamp_edit", "")
        line(2, "Fluid:", "metadata_fluid_edit", "DI water")
        line(3, "Triton concentration:", "metadata_triton_edit", "0.02")
        layout.addWidget(QLabel("Triton basis:"), 4, 0)
        self.metadata_triton_basis_combo = QComboBox()
        self.metadata_triton_basis_combo.addItems(["unspecified", "wt%", "vol%", "w/v%"])
        layout.addWidget(self.metadata_triton_basis_combo, 4, 1)
        line(5, "Particle material:", "metadata_particle_edit", "SS")
        line(6, "Particle size range (um):", "metadata_particle_size_edit", "130-155")
        line(7, "Vial set pressure A/B (kPa):", "metadata_set_pressure_edit", "")
        line(8, "Actual pressure A/B (kPa):", "metadata_actual_pressure_edit", "")
        layout.addWidget(QLabel("Pressure source:"), 9, 0)
        self.metadata_pressure_source_combo = QComboBox()
        self.metadata_pressure_source_combo.addItems(["manual input", "pressure sensor"])
        layout.addWidget(self.metadata_pressure_source_combo, 9, 1)
        line(10, "Temperature (C):", "metadata_temperature_edit", "")
        layout.addWidget(QLabel("Effective travel distance (mm):"), 11, 0)
        self.metadata_distance_spin = QDoubleSpinBox()
        self.metadata_distance_spin.setRange(0.001, 10000.0)
        self.metadata_distance_spin.setDecimals(3)
        self.metadata_distance_spin.setValue(49.0)
        layout.addWidget(self.metadata_distance_spin, 11, 1)
        note = QLabel("센서가 없으면 실제 압력과 온도는 수동 입력값으로 명시해 저장합니다.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note, 12, 0, 1, 2)
        return g

    @staticmethod
    def _parse_pair(text):
        values = str(text).replace(",", "/").split("/")
        if len(values) != 2:
            return np.nan, np.nan
        try:
            return float(values[0].strip()), float(values[1].strip())
        except ValueError:
            return np.nan, np.nan

    @staticmethod
    def _parse_size_range(text):
        values = str(text).replace("~", "-").split("-")
        if len(values) != 2:
            return np.nan, np.nan
        try:
            return float(values[0].strip()), float(values[1].strip())
        except ValueError:
            return np.nan, np.nan

    def _build_plotter_group(self):
        g = QGroupBox("6. Plotter — 데이터 스트림 & Math(Histogram/Std)")
        layout = QGridLayout(g)

        layout.addWidget(QLabel("저장 샘플 속도 (Sa/s):"), 0, 0)
        self.plotter_rate_spin = QDoubleSpinBox()
        self.plotter_rate_spin.setRange(1.0, 7200.0)
        self.plotter_rate_spin.setDecimals(1)
        self.plotter_rate_spin.setSingleStep(900.0)
        self.plotter_rate_spin.setValue(1800.0)
        self.plotter_rate_spin.setSuffix(" Sa/s")
        self.plotter_rate_spin.setToolTip(
            "HF2 demodulator USB streaming rate를 직접 설정합니다. "
            "권장 단계는 1800, 3600, 7200 Sa/s입니다. "
            "장비가 지원하는 가장 가까운 분주값이 실제로 적용됩니다. "
            "화면은 저장 속도와 무관하게 약 20 Hz로 갱신됩니다."
        )
        layout.addWidget(self.plotter_rate_spin, 0, 1)

        self.plotter_actual_rate_label = QLabel("실제 적용: 시작 후 표시")
        self.plotter_actual_rate_label.setToolTip(
            "HF2LI는 요청값을 지원하는 분주 속도로 근사합니다. "
            "CSV에는 요청 속도와 실제 속도가 모두 저장됩니다."
        )
        layout.addWidget(self.plotter_actual_rate_label, 1, 0, 1, 2)

        layout.addWidget(QLabel("시간축 폭 (s):"), 2, 0)
        self.plotter_window_spin = QDoubleSpinBox()
        self.plotter_window_spin.setRange(0.5, 3600)
        self.plotter_window_spin.setValue(10.0)
        layout.addWidget(self.plotter_window_spin, 2, 1)

        btn_row = QHBoxLayout()
        self.plotter_start_btn = QPushButton("Plotter 시작")
        self.plotter_start_btn.clicked.connect(self.start_plotter)
        self.plotter_stop_btn = QPushButton("Plotter 중지")
        self.plotter_stop_btn.setEnabled(False)
        self.plotter_stop_btn.clicked.connect(self.stop_plotter)
        btn_row.addWidget(self.plotter_start_btn)
        btn_row.addWidget(self.plotter_stop_btn)
        layout.addLayout(btn_row, 3, 0, 1, 2)

        self.baseline_start_btn = QPushButton("Baseline 테스트 시작 (입자/공압전환 비활성)")
        self.baseline_start_btn.setToolTip("연속 PLL tracking만 기록합니다. Ping-Pong 입자 감지와 자동 압력 전환은 실행되지 않습니다.")
        self.baseline_start_btn.clicked.connect(self.start_baseline_test)
        layout.addWidget(self.baseline_start_btn, 4, 0, 1, 2)

        self.baseline_stop_btn = QPushButton("Baseline 테스트 중지")
        self.baseline_stop_btn.setEnabled(False)
        self.baseline_stop_btn.clicked.connect(self.stop_baseline_test)
        layout.addWidget(self.baseline_stop_btn, 5, 0, 1, 2)

        baseline_analysis = QWidget()
        baseline_layout = QGridLayout(baseline_analysis)
        baseline_layout.setContentsMargins(6, 4, 6, 4)
        baseline_layout.setColumnStretch(1, 1)

        self.baseline_stats_label = QLabel("Baseline 분석: 아직 측정 데이터가 없습니다.")
        self.baseline_stats_label.setWordWrap(True)
        self.baseline_stats_label.setStyleSheet("font-weight: bold; color: #444;")
        baseline_layout.addWidget(self.baseline_stats_label, 0, 0, 1, 2)

        analysis_row = QHBoxLayout()
        self.baseline_analysis_btn = QPushButton("Baseline 상세 분석 실행 (최근 분석 구간)")
        self.baseline_analysis_btn.setToolTip("최근 최대 120초/10만 분석 샘플을 사용합니다. 전체 raw 기록은 CSV 저장으로 보존됩니다.")
        self.baseline_analysis_btn.clicked.connect(self.run_baseline_analysis)
        analysis_row.addWidget(self.baseline_analysis_btn)
        self.baseline_load_btn = QPushButton("평가/입자 CSV 분석")
        self.baseline_load_btn.setToolTip("원본 CSV는 수정하지 않고 메모리에서만 derived 분석을 수행합니다.")
        self.baseline_load_btn.clicked.connect(self.load_baseline_csv_for_analysis)
        analysis_row.addWidget(self.baseline_load_btn)
        baseline_layout.addLayout(analysis_row, 1, 0, 1, 2)

        blank_row = QHBoxLayout()
        self.calibration_blank_btn = QPushButton("Threshold calibration blank 선택")
        self.calibration_blank_btn.clicked.connect(self.select_calibration_blank)
        self.evaluation_blank_btn = QPushButton("Evaluation blank 분석")
        self.evaluation_blank_btn.clicked.connect(self.load_evaluation_blank)
        blank_row.addWidget(self.calibration_blank_btn)
        blank_row.addWidget(self.evaluation_blank_btn)
        baseline_layout.addLayout(blank_row, 2, 0, 1, 2)

        baseline_layout.addWidget(QLabel("파형 duration / method:"), 3, 0)
        detector_cfg = QHBoxLayout()
        detector_cfg.addWidget(QLabel("0.3 / 0.5 / 1.0 s"))
        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItems(["integral", "template"])
        detector_cfg.addWidget(self.detection_method_combo)
        baseline_layout.addLayout(detector_cfg, 3, 1)
        self.event_duration_spin = QDoubleSpinBox(); self.event_duration_spin.setRange(0.01, 10.0); self.event_duration_spin.setDecimals(3); self.event_duration_spin.setValue(0.3); self.event_duration_spin.setVisible(False)
        event_cfg = QHBoxLayout()
        self.event_baseline_spin = QDoubleSpinBox(); self.event_baseline_spin.setRange(0.05, 60.0); self.event_baseline_spin.setDecimals(3); self.event_baseline_spin.setValue(0.4)
        self.event_guard_spin = QDoubleSpinBox(); self.event_guard_spin.setRange(0.0, 10.0); self.event_guard_spin.setDecimals(3); self.event_guard_spin.setValue(0.05)
        event_cfg.addWidget(self.event_baseline_spin); event_cfg.addWidget(QLabel("/")); event_cfg.addWidget(self.event_guard_spin)
        baseline_layout.addLayout(event_cfg, 4, 1)
        baseline_layout.addWidget(QLabel("Local baseline / guard (s):"), 4, 0)
        baseline_layout.addWidget(QLabel("Manual peak threshold override (Hz, 0=calibration):"), 5, 0)
        self.event_threshold_spin = QDoubleSpinBox(); self.event_threshold_spin.setRange(0.0, 1e6); self.event_threshold_spin.setDecimals(4); self.event_threshold_spin.setValue(0.0)
        baseline_layout.addWidget(self.event_threshold_spin, 5, 1)

        baseline_layout.addWidget(QLabel("추가 rolling window (s, 0=기본값만):"), 6, 0)
        self.baseline_rolling_window_spin = QDoubleSpinBox(); self.baseline_rolling_window_spin.setRange(0.0, 36000.0); self.baseline_rolling_window_spin.setDecimals(2); self.baseline_rolling_window_spin.setValue(60.0)
        baseline_layout.addWidget(self.baseline_rolling_window_spin, 6, 1)

        analysis_export_row = QHBoxLayout()
        self.baseline_summary_export_btn = QPushButton("분석 Summary JSON 저장")
        self.baseline_summary_export_btn.setEnabled(False)
        self.baseline_summary_export_btn.clicked.connect(self.export_baseline_summary)
        self.baseline_derived_export_btn = QPushButton("Derived CSV 저장")
        self.baseline_derived_export_btn.setEnabled(False)
        self.baseline_derived_export_btn.clicked.connect(self.export_baseline_derived_csv)
        self.baseline_allan_export_btn = QPushButton("Allan 그래프·CSV 따로 저장")
        self.baseline_allan_export_btn.setEnabled(False)
        self.baseline_allan_export_btn.clicked.connect(self.export_baseline_allan)
        self.event_export_btn = QPushButton("Event/Threshold CSV 저장")
        self.event_export_btn.setEnabled(False)
        self.event_export_btn.clicked.connect(self.export_event_analysis_csv)
        analysis_export_row.addWidget(self.baseline_summary_export_btn)
        analysis_export_row.addWidget(self.baseline_derived_export_btn)
        analysis_export_row.addWidget(self.baseline_allan_export_btn)
        analysis_export_row.addWidget(self.event_export_btn)
        baseline_layout.addLayout(analysis_export_row, 7, 0, 1, 2)

        allan_calibration_row = QHBoxLayout()
        self.allan_sensitivity_spin = QDoubleSpinBox()
        self.allan_sensitivity_spin.setRange(0.0, 1e15)
        self.allan_sensitivity_spin.setDecimals(3)
        self.allan_sensitivity_spin.setValue(0.0)
        self.allan_sensitivity_spin.setToolTip(
            "해당 공진 모드의 측정된 질량 감도(pg/Hz)를 입력하면 µg로 환산합니다. "
            "0이면 주파수 Allan deviation(Hz)을 저장합니다.")
        allan_calibration_row.addWidget(QLabel("질량 감도 (pg/Hz, 0=Hz 그래프):"))
        allan_calibration_row.addWidget(self.allan_sensitivity_spin)
        baseline_layout.addLayout(allan_calibration_row, 8, 0, 1, 2)

        audit_row = QHBoxLayout()
        compare_btn = QPushButton("반복 baseline 비교")
        compare_btn.clicked.connect(self.compare_baseline_files)
        integration_btn = QPushButton("통합시험 리포트")
        integration_btn.clicked.connect(self.export_integration_report)
        bundle_btn = QPushButton("Session bundle 저장")
        bundle_btn.clicked.connect(self.export_session_bundle)
        audit_row.addWidget(compare_btn)
        audit_row.addWidget(integration_btn)
        audit_row.addWidget(bundle_btn)
        baseline_layout.addLayout(audit_row, 9, 0, 1, 2)

        layout.addWidget(CollapsibleSection(
            "Baseline 분석 및 내보내기", baseline_analysis, expanded=False,
        ), 5, 0, 1, 2)

        layout.addWidget(QLabel("Histogram 대상 채널:"), 6, 0)
        self.hist_channel_combo = QComboBox()
        self.hist_channel_combo.addItems(HIST_CHANNELS)
        self.hist_channel_combo.setCurrentText("Frequency (Hz)")
        layout.addWidget(self.hist_channel_combo, 6, 1)

        note = QLabel("※ 공진 주파수 노이즈를 보려면 Frequency를 선택하세요 (Amplitude 아님).")
        note.setStyleSheet("color: gray; font-size: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note, 7, 0, 1, 2)

        self.plotter_std_label = QLabel("Std: -")
        self.plotter_std_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.plotter_std_label, 8, 0, 1, 2)

        save_plotter_btn = QPushButton("전체 세션 CSV로 저장")
        save_plotter_btn.setToolTip(
            "위 그래프는 '시간축 폭'만큼만 보여주지만, 저장은 Plotter를\n"
            "시작한 뒤로 지금까지의 전체 기록을 저장합니다."
        )
        save_plotter_btn.clicked.connect(self.export_plotter_csv)
        layout.addWidget(save_plotter_btn, 9, 0, 1, 2)

        return g

    def _build_live_particle_group(self):
        g = QGroupBox("7. 실시간 파티클 검출 — 데모")
        layout = QGridLayout(g)

        self.live_particle_enabled = QCheckBox("PLL / Plotter 실행 중 자동 검출")
        self.live_particle_enabled.setChecked(True)
        self.live_particle_enabled.setToolTip(
            "첫 blank 구간의 국소 깊이 noise로 문턱을 고정한 뒤 "
            "오프라인 분석기와 같은 short(≤0.20초) 깊이 검출을 적용합니다."
        )
        self.live_particle_enabled.toggled.connect(self.reset_live_particle_detector)
        layout.addWidget(self.live_particle_enabled, 0, 0, 1, 2)

        layout.addWidget(QLabel("초기 blank 학습 (s):"), 1, 0)
        self.live_calibration_spin = QDoubleSpinBox()
        self.live_calibration_spin.setRange(6.0, 60.0)
        self.live_calibration_spin.setDecimals(1)
        self.live_calibration_spin.setValue(8.0)
        self.live_calibration_spin.setSuffix(" s")
        layout.addWidget(self.live_calibration_spin, 1, 1)

        layout.addWidget(QLabel("깊이 noise 배수 / 최소 깊이:"), 2, 0)
        sensitivity = QHBoxLayout()
        self.live_sigma_spin = QDoubleSpinBox()
        self.live_sigma_spin.setRange(1.0, 30.0)
        self.live_sigma_spin.setDecimals(1)
        self.live_sigma_spin.setValue(6.0)
        self.live_floor_spin = QDoubleSpinBox()
        self.live_floor_spin.setRange(0.001, 10000.0)
        self.live_floor_spin.setDecimals(3)
        self.live_floor_spin.setValue(0.15)
        self.live_floor_spin.setSuffix(" Hz")
        sensitivity.addWidget(self.live_sigma_spin)
        sensitivity.addWidget(QLabel("× /"))
        sensitivity.addWidget(self.live_floor_spin)
        layout.addLayout(sensitivity, 2, 1)

        self.live_savgol_enabled = QCheckBox("검출에 Savitzky–Golay 적용 · 필터선 실시간 표시")
        self.live_savgol_enabled.setChecked(False)
        self.live_savgol_enabled.setToolTip(
            "blank 학습과 후보 검출에 적용하고 실시간 파티클 그래프에 같은 필터 결과를 분홍색으로 표시합니다. "
            "원시 그래프와 저장 CSV는 변경하지 않습니다. 설정을 바꾸면 문턱을 다시 학습합니다."
        )
        layout.addWidget(self.live_savgol_enabled, 3, 0, 1, 2)
        sg_controls = QHBoxLayout()
        self.live_savgol_window_spin = QSpinBox()
        self.live_savgol_window_spin.setRange(3, 501)
        self.live_savgol_window_spin.setSingleStep(2)
        self.live_savgol_window_spin.setValue(27)
        self.live_savgol_window_spin.setSuffix(" 표본")
        self.live_savgol_window_spin.setToolTip(
            "오프라인 SG와 같은 표본 수입니다. 짝수는 다음 홀수로 바뀝니다. "
            "고속 획득에서는 실시간 검출 샘플이 최대 약 4 kSa/s로 줄어들 수 있어 "
            "같은 표본 수의 시간 폭이 원시 CSV와 다를 수 있습니다."
        )
        self.live_savgol_order_spin = QSpinBox()
        self.live_savgol_order_spin.setRange(1, 8)
        self.live_savgol_order_spin.setValue(4)
        sg_controls.addWidget(self.live_savgol_window_spin)
        sg_controls.addWidget(QLabel("/ 다항 차수"))
        sg_controls.addWidget(self.live_savgol_order_spin)
        layout.addWidget(QLabel("필터 Window / 차수:"), 4, 0)
        layout.addLayout(sg_controls, 4, 1)
        self.live_savgol_window_spin.setEnabled(False)
        self.live_savgol_order_spin.setEnabled(False)
        self.live_savgol_enabled.toggled.connect(self.live_savgol_window_spin.setEnabled)
        self.live_savgol_enabled.toggled.connect(self.live_savgol_order_spin.setEnabled)
        self.live_savgol_enabled.toggled.connect(self._set_live_filtered_trace_visible)
        self.live_savgol_enabled.toggled.connect(self.reset_live_particle_detector)
        self.live_savgol_window_spin.valueChanged.connect(self._normalize_live_savgol_window)
        self.live_savgol_order_spin.valueChanged.connect(self.reset_live_particle_detector)

        self.live_reset_btn = QPushButton("검출 초기화 · blank 재학습")
        self.live_reset_btn.clicked.connect(self.reset_live_particle_detector)
        layout.addWidget(self.live_reset_btn, 5, 0, 1, 2)

        self.live_detector_status = QLabel("대기: Plotter를 시작하면 자동으로 blank를 학습합니다.")
        self.live_detector_status.setWordWrap(True)
        self.live_detector_status.setStyleSheet("font-weight: 700; color: #147d64;")
        layout.addWidget(self.live_detector_status, 6, 0, 1, 2)

        self.live_export_btn = QPushButton("실시간 후보 CSV 저장")
        self.live_export_btn.setEnabled(False)
        self.live_export_btn.clicked.connect(self.export_live_particle_csv)
        layout.addWidget(self.live_export_btn, 7, 0, 1, 2)
        return g

    def _normalize_live_savgol_window(self, value):
        if value % 2 == 0:
            self.live_savgol_window_spin.setValue(min(self.live_savgol_window_spin.maximum(), value + 1))
            return
        self.reset_live_particle_detector()

    def _set_live_filtered_trace_visible(self, enabled):
        self.live_filtered_frequency_curve.setVisible(bool(enabled))
        if not enabled:
            self.live_filtered_frequency_curve.setData([], [])

    def _build_daq_group(self):
        g = QGroupBox("8. 에어레귤레이터 (NI-DAQ)")
        layout = QGridLayout(g)

        self.daq_spins = {}
        row = 0
        for label, ch in DAQ_CHANNELS.items():
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(MIN_DAQ_V, MAX_DAQ_V)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            layout.addWidget(spin, row, 1)
            set_btn = QPushButton("설정")
            set_btn.clicked.connect(lambda _, c=ch, s=spin: self.set_daq_voltage(c, s.value()))
            layout.addWidget(set_btn, row, 2)
            self.daq_spins[ch] = spin
            row += 1

        zero_btn = QPushButton("두 채널 모두 0V")
        zero_btn.clicked.connect(self.zero_daq_channels)
        layout.addWidget(zero_btn, row, 0, 1, 3)

        self.pressure_monitor_btn = QPushButton("실제 압력 읽기 시작 (AI0 / AI1)")
        self.pressure_monitor_btn.setCheckable(True)
        self.pressure_monitor_btn.toggled.connect(self._toggle_pressure_monitor)
        layout.addWidget(self.pressure_monitor_btn, row + 1, 0, 1, 3)
        self.pressure_readback_label = QLabel("A (AI0): —    B (AI1): —    A−B: —")
        self.pressure_readback_label.setWordWrap(True)
        layout.addWidget(self.pressure_readback_label, row + 2, 0, 1, 3)
        note = QLabel("ITV0050 · RSE · COM → AI GND · 약 5 Hz\n"
                      "1–5 V → 1–900 kPa 명목 환산 · 모니터 정확도 ±6% F.S.")
        note.setWordWrap(True)
        layout.addWidget(note, row + 3, 0, 1, 3)
        self.pressure_watchdog = QTimer(self)
        self.pressure_watchdog.timeout.connect(self._expire_pressure_sample)
        self.pressure_watchdog.start(250)

        return g

    def _toggle_pressure_monitor(self, enabled):
        self.measured_pressure_a = self.measured_pressure_b = None
        self.pressure_sample_time = None
        if enabled:
            self.pressure_readback_label.setText(f"{DAQ_DEVICE}/ai0:1 연결 중…")
            self.pressure_worker = PressureMonitor(DAQ_DEVICE, self)
            self.pressure_worker.sample.connect(self._pressure_sample)
            self.pressure_worker.failed.connect(self._pressure_failed)
            self.pressure_worker.finished.connect(self._pressure_finished)
            self.pressure_monitor_btn.setText("실제 압력 읽기 중지")
            self.pressure_worker.start()
        elif self.pressure_worker is not None:
            self.pressure_worker.requestInterruption()
            self.pressure_monitor_btn.setEnabled(False)
            self.pressure_readback_label.setText("압력 읽기 중지 중…")
        if self.measurement_metadata:
            self.measurement_metadata.setdefault("Pressure_monitor_events", []).append({
                "timestamp_local": datetime.now().astimezone().isoformat(),
                "enabled": bool(enabled), "channels": f"{DAQ_DEVICE}/ai0:1"})
        self.metadata_actual_pressure_edit.setEnabled(not enabled)
        self.metadata_pressure_source_combo.setEnabled(not enabled)

    def _pressure_sample(self, a, b, timestamp):
        if not self.pressure_monitor_btn.isChecked():
            return
        self.pressure_sample_time = timestamp
        self.measured_pressure_a, self.measured_pressure_b = pressure_kpa(a), pressure_kpa(b)
        def label(value, voltage):
            return (f"{value:.1f} kPa ({voltage:.4f} V)" if value is not None
                    else f"범위 밖 ({voltage:.4f} V)")
        pa, pb = self.measured_pressure_a, self.measured_pressure_b
        delta = f"{pa - pb:.1f} kPa" if pa is not None and pb is not None else "—"
        self.pressure_readback_label.setText(
            f"A (AI0): {label(pa, a)}\nB (AI1): {label(pb, b)}\nA−B: {delta}")
        self._expire_pressure_sample()

    def _expire_pressure_sample(self):
        if (self.pressure_monitor_btn.isChecked() and self.pressure_sample_time is not None
                and time.monotonic() - self.pressure_sample_time > 1.0):
            self.measured_pressure_a = self.measured_pressure_b = None
            self.pressure_readback_label.setText("압력 데이터 지연: 실측값 저장 중단")

    def _pressure_failed(self, message):
        self.measured_pressure_a = self.measured_pressure_b = None
        self.pressure_sample_time = None
        self.pressure_readback_label.setText(f"압력 읽기 오류: {message}")
        if self.measurement_metadata:
            self.measurement_metadata.setdefault("Pressure_monitor_events", []).append({
                "timestamp_local": datetime.now().astimezone().isoformat(),
                "error": message})

    def _pressure_finished(self):
        worker = self.pressure_worker
        self.pressure_worker = None
        if worker is not None:
            worker.deleteLater()
        self.pressure_monitor_btn.blockSignals(True)
        self.pressure_monitor_btn.setChecked(False)
        self.pressure_monitor_btn.blockSignals(False)
        self.pressure_monitor_btn.setEnabled(True)
        self.pressure_monitor_btn.setText("실제 압력 읽기 시작 (AI0 / AI1)")
        if self.pressure_readback_label.text() == "압력 읽기 중지 중…":
            self.pressure_readback_label.setText("압력 읽기 중지됨 · A: —  B: —")
        self.measured_pressure_a = self.measured_pressure_b = None
        self.metadata_actual_pressure_edit.setEnabled(True)
        self.metadata_pressure_source_combo.setEnabled(True)

    def _build_pingpong_group(self):
        g = QGroupBox("9. 파티클 반복측정 (Pneumatic Ping-Pong)")
        layout = QGridLayout(g)
        row = 0

        note = QLabel(
            "PLL로 잠긴 상태에서 Plotter의 Frequency 신호를 감시하다가,\n"
            "입자 통과로 인한 공진주파수 급강하(딥)를 감지하면 통과가\n"
            "끝난 뒤 자동으로 공압 방향을 반전시켜 같은 입자를 반복\n"
            "왕복시킵니다 (Katsikis et al. 방식). PLL 활성화 + Plotter\n"
            "실행 중이어야 합니다."
        )
        note.setStyleSheet("color: gray; font-size: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note, row, 0, 1, 2)
        row += 1

        base_label = QLabel("기본 전압 Base (V):")
        base_label.setToolTip("기포 억제·유체 응답성을 위해 양쪽 채널에 공통으로 인가하는 기본 압력")
        layout.addWidget(base_label, row, 0)
        self.pp_base_v_spin = QDoubleSpinBox()
        self.pp_base_v_spin.setRange(MIN_DAQ_V, MAX_DAQ_V)
        self.pp_base_v_spin.setDecimals(3)
        self.pp_base_v_spin.setValue(2.0)
        layout.addWidget(self.pp_base_v_spin, row, 1)
        row += 1

        delta_label = QLabel("차압 ΔV (V):")
        delta_label.setToolTip("두 채널 사이의 차압으로 유속/방향을 결정 (Base±ΔV, 0~10V로 클램프)")
        layout.addWidget(delta_label, row, 0)
        self.pp_delta_v_spin = QDoubleSpinBox()
        self.pp_delta_v_spin.setRange(0.0, MAX_DAQ_V)
        self.pp_delta_v_spin.setDecimals(3)
        self.pp_delta_v_spin.setValue(0.3)
        layout.addWidget(self.pp_delta_v_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("딥 감지 임계값 (Hz):"), row, 0)
        self.pp_threshold_spin = QDoubleSpinBox()
        self.pp_threshold_spin.setRange(0.01, 10000)
        self.pp_threshold_spin.setDecimals(3)
        self.pp_threshold_spin.setValue(3.0)
        layout.addWidget(self.pp_threshold_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("반전 후 정착 대기 (s):"), row, 0)
        self.pp_settle_spin = QDoubleSpinBox()
        self.pp_settle_spin.setRange(0.0, 60.0)
        self.pp_settle_spin.setValue(1.0)
        layout.addWidget(self.pp_settle_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("목표 반복 횟수 (0=무한):"), row, 0)
        self.pp_max_cycles_spin = QSpinBox()
        self.pp_max_cycles_spin.setRange(0, 100000)
        self.pp_max_cycles_spin.setValue(30)
        layout.addWidget(self.pp_max_cycles_spin, row, 1)
        row += 1

        btn_row = QHBoxLayout()
        self.pp_start_btn = QPushButton("Ping-Pong 시작")
        self.pp_start_btn.clicked.connect(self.start_pingpong)
        self.pp_stop_btn = QPushButton("Ping-Pong 정지")
        self.pp_stop_btn.setEnabled(False)
        self.pp_stop_btn.clicked.connect(self.stop_pingpong)
        btn_row.addWidget(self.pp_start_btn)
        btn_row.addWidget(self.pp_stop_btn)
        layout.addLayout(btn_row, row, 0, 1, 2)
        row += 1

        self.pp_status_label = QLabel("상태: 대기 중")
        self.pp_status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.pp_status_label, row, 0, 1, 2)
        row += 1

        self.pp_event_list = QListWidget()
        self.pp_event_list.setMaximumHeight(140)
        layout.addWidget(self.pp_event_list, row, 0, 1, 2)
        row += 1

        save_pp_btn = QPushButton("이벤트 목록 CSV로 저장")
        save_pp_btn.clicked.connect(self.export_pingpong_csv)
        layout.addWidget(save_pp_btn, row, 0, 1, 2)

        return g

    # ------------------------------------------------------------------
    def _connect_hardware(self):
        try:
            self.hf = HFController()
            hf_ok = True
        except Exception as e:
            hf_ok = False
            QMessageBox.warning(self, "HF2LI 연결 실패", str(e))

        daq_ok = nidaqmx is not None
        if nidaqmx is None:
            QMessageBox.warning(
                self,
                "NI-DAQ 초기화 실패",
                "내장 nidaqmx 모듈을 초기화할 수 없습니다.\n\n"
                f"원인: {NIDAQMX_IMPORT_ERROR or '알 수 없음'}\n\n"
                "이 PC에 NI-DAQmx 드라이버가 설치되어 있는지도 확인하세요.",
            )

        self.status_label.setText(
            f"HF2LI({HF2_DEVICE}): {'연결됨' if hf_ok else '실패'}  |  "
            f"NI-DAQ({DAQ_DEVICE}): {'사용 가능' if daq_ok else '실패'}"
        )

        if hf_ok:
            try:
                # Reference configuration copied from the supplied LabOne view.
                # Deliberately do not enable Output/Amp here: this application
                # drives a piezo directly and only a deliberate sweep/PLL action
                # may energize it.
                self.hf.set_output_on(False)
                self.hf.set_amp_enabled(False)
                self.hf.set_output_range(10.0)
                self.hf.set_drive_amplitude(10.0)
                for input_index, input_range in enumerate((1.3, 1.1)):
                    self.hf.set_input_range(input_index, input_range)
                    self.hf.set_input_ac_coupling(input_index, True)
                    self.hf.set_input_50_ohm(input_index, False)
                    self.hf.set_input_differential(input_index, False)
                self.hf.set_demod_order(4)
                self.hf.set_demod_tc(tc_from_bandwidth_3db(4.865, 4))
                self.hf.set_demod_rate(224.9)
            except Exception:
                # Individual optional nodes can differ on legacy firmware; the
                # controls below still show whatever values can be read.
                pass
            try:
                current_range = self.hf.get_output_range()
                closest_idx = min(range(len(OUTPUT_RANGE_OPTIONS)),
                                  key=lambda i: abs(OUTPUT_RANGE_OPTIONS[i][1] - current_range))
                self.range_combo.blockSignals(True)
                self.range_combo.setCurrentIndex(closest_idx)
                self.range_combo.blockSignals(False)
            except Exception:
                pass
            if not (STATE_DIR / "last_settings.json").exists():
                try:
                    pll_input = self.hf.get_pll_input()
                    combo_index = self.pll_input_combo.findData(pll_input)
                    if combo_index >= 0:
                        self.pll_input_combo.setCurrentIndex(combo_index)
                except Exception:
                    pass
            for input_index in range(2):
                try:
                    self._sync_input_controls(input_index)
                except Exception:
                    pass

    # --- drive output ---
    # "On", "Amp", and the Amp value are independent controls in LabOne, so
    # each is wired to its own node here with no implied side effects.
    def set_drive_amplitude_value(self, volts):
        if not self.hf:
            return
        try:
            self.hf.set_drive_amplitude(volts)
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))

    def on_range_changed(self, index):
        if not self.hf or index < 0:
            return
        _, volts = OUTPUT_RANGE_OPTIONS[index]
        try:
            self.hf.set_output_range(volts)
            # Keep the actual output voltage consistent with what the Amp
            # spinbox displays -- the AMPLITUDES fraction is relative to
            # Range, so changing Range alone would silently change the
            # real output volts unless we re-apply the intended Vpk here.
            self.hf.set_drive_amplitude(self.drive_v_spin.value())
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))

    def toggle_output(self, checked):
        if not self.hf:
            return
        try:
            self.hf.set_output_on(checked)
            self.output_on_btn.setText("On ✓" if checked else "On")
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))
            self.output_on_btn.setChecked(False)

    def toggle_amp_enabled(self, checked):
        if not self.hf:
            return
        try:
            self.hf.set_amp_enabled(checked)
            self.amp_enable_btn.setText("Amp ✓" if checked else "Amp")
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))
            self.amp_enable_btn.setChecked(False)

    def _sync_input_controls(self, input_index):
        range_spin = self.input_range_spins[input_index]
        range_spin.blockSignals(True)
        range_spin.setValue(self.hf.get_input_range(input_index))
        range_spin.blockSignals(False)
        for button, value, setter in (
            (self.input_ac_btns[input_index], self.hf.get_input_ac_coupling(input_index), self._set_input_ac_button_text),
            (self.input_50_btns[input_index], self.hf.get_input_50_ohm(input_index), self._set_input_50_button_text),
            (self.input_diff_btns[input_index], self.hf.get_input_differential(input_index), self._set_input_diff_button_text),
        ):
            button.blockSignals(True)
            button.setChecked(value)
            button.blockSignals(False)
            setter(input_index, value)

    def set_input_range_value(self, input_index, volts):
        if not self.hf:
            return
        try:
            self.hf.set_input_range(input_index, volts)
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))

    def _set_input_ac_button_text(self, input_index, checked):
        self.input_ac_btns[input_index].setText(f"AC{' ✓' if checked else ''}")

    def _set_input_50_button_text(self, input_index, checked):
        self.input_50_btns[input_index].setText(f"50 Ω{' ✓' if checked else ''}")

    def _set_input_diff_button_text(self, input_index, checked):
        self.input_diff_btns[input_index].setText(f"Diff{' ✓' if checked else ''}")

    def toggle_input_ac(self, input_index, checked):
        if not self.hf:
            return
        try:
            self.hf.set_input_ac_coupling(input_index, checked)
            self._set_input_ac_button_text(input_index, checked)
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))

    def toggle_input_50_ohm(self, input_index, checked):
        if not self.hf:
            return
        try:
            self.hf.set_input_50_ohm(input_index, checked)
            self._set_input_50_button_text(input_index, checked)
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))

    def toggle_input_differential(self, input_index, checked):
        """Mirror LabOne's Signal Inputs > Diff switch for one HF2LI input."""
        if not self.hf:
            return
        try:
            self.hf.set_input_differential(input_index, checked)
            self._set_input_diff_button_text(input_index, checked)
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))
            button = self.input_diff_btns[input_index]
            button.blockSignals(True)
            button.setChecked(not checked)
            button.blockSignals(False)
            self._set_input_diff_button_text(input_index, not checked)

    def _ensure_drive_active(self):
        """Sweep/PLL need an actually-driven signal to mean anything, so
        make sure both On and Amp are enabled (updating the buttons to
        match) -- the user can still flip either off independently after."""
        self.hf.set_drive_amplitude(self.drive_v_spin.value())
        self.hf.set_output_on(True)
        self.hf.set_amp_enabled(True)
        self.output_on_btn.setChecked(True)
        self.amp_enable_btn.setChecked(True)

    # --- theoretical doubly-clamped tube modes ---
    def calculate_theory_modes(self):
        try:
            beam = TubeBeam(
                self.theory_length_spin.value() * 1e-3,
                self.theory_outer_spin.value() * 1e-3,
                self.theory_inner_spin.value() * 1e-3,
                self.theory_e_spin.value() * 1e9,
                self.theory_density_spin.value(),
                self.theory_fluid_density_spin.value(),
            )
            self.theory_modes = calculate_modes(beam, self.theory_count_spin.value())
        except ValueError as e:
            QMessageBox.warning(self, "이론 모드 입력", str(e))
            return
        self.theory_mode_list.clear()
        for mode in self.theory_modes:
            self.theory_mode_list.addItem(
                f"Mode {mode.index}: {mode.frequency_hz:,.3f} Hz  (λ={mode.lambda_n:.5f})"
            )
        self.update_theory_lines()

    def update_theory_lines(self):
        for line in self.theory_lines:
            self.sweep_amp_plot.removeItem(line)
        self.theory_lines = []
        if not self.theory_show_check.isChecked():
            return
        for mode in self.theory_modes:
            line = pg.InfiniteLine(pos=mode.frequency_hz, angle=90,
                                   pen=pg.mkPen('#7f8c8d', width=1, style=Qt.DotLine))
            self.sweep_amp_plot.addItem(line)
            self.theory_lines.append(line)

    def use_selected_theory_mode(self):
        row = self.theory_mode_list.currentRow()
        if row < 0 or row >= len(self.theory_modes):
            QMessageBox.information(self, "이론 모드", "먼저 계산된 모드를 선택하세요.")
            return
        f0 = self.theory_modes[row].frequency_hz
        previous_f = self.theory_modes[row - 1].frequency_hz if row else 0.0
        next_f = self.theory_modes[row + 1].frequency_hz if row + 1 < len(self.theory_modes) else f0 * 1.8
        low = max(1.0, (previous_f + f0) / 2.0 if previous_f else f0 * 0.7)
        high = (f0 + next_f) / 2.0
        self.f_start_spin.setValue(low)
        self.f_stop_spin.setValue(high)

    # --- sweep ---
    def run_sweep(self):
        if not self.hf:
            return
        try:
            self._ensure_drive_active()
        except ZiAPIError as e:
            QMessageBox.critical(self, "HF2LI 오류", str(e))
            return

        self._save_last_settings()

        n = self.n_points_spin.value()
        self.sweep_freqs = np.full(n, np.nan)
        self.sweep_R = np.full(n, np.nan)
        self.sweep_theta = np.full(n, np.nan)
        self.sweep_X = np.full(n, np.nan)
        self.sweep_Y = np.full(n, np.nan)
        self.sweep_amp_curve.setData([], [])
        self.sweep_phase_curve.setData([], [])
        self.sweep_fit_amp_curve.setData([], [])
        self.sweep_fit_phase_curve.setData([], [])
        self.peak_markers.setData([], [])
        self.peak_list.clear()
        self._peak_candidates = []
        self.last_resonance_fit = None
        self.pll_calibrated_slope = None
        self.pll_calibrated_setpoint = None
        self.tabs.setCurrentIndex(0)

        settings = {
            "filter_mode": self.sweep_filter_mode_combo.currentText(),
            "order": self.sweep_filter_order_spin.value(),
            "fixed_bandwidth_hz": self.sweep_fixed_bw_spin.value(),
            "max_bandwidth_hz": self.sweep_max_bw_spin.value(),
            "omega_suppression_db": self.sweep_suppression_spin.value(),
            "bandwidth_overlap": self.sweep_overlap_check.isChecked(),
            "min_settling_s": self.sweep_min_settle_spin.value(),
            "settling_residual": self.sweep_residual_spin.value(),
            "sample_count": self.sweep_samples_spin.value(),
            "sample_rate_hz": self.sweep_rate_spin.value(),
            "averaging_time_s": self.sweep_avg_time_spin.value(),
            "averaging_tc": self.sweep_avg_tc_spin.value(),
        }
        self.sweep_worker = SweepWorker(
            self.f_start_spin.value(),
            self.f_stop_spin.value(),
            n,
            settings,
        )
        self.sweep_worker.point_ready.connect(self.on_sweep_point)
        self.sweep_worker.progress.connect(self.on_sweep_progress)
        self.sweep_worker.finished_ok.connect(self.on_sweep_finished)
        self.sweep_worker.failed.connect(self.on_sweep_failed)
        self.run_sweep_btn.setEnabled(False)
        self.stop_sweep_btn.setEnabled(True)
        self.sweep_worker.start()

    def stop_sweep(self):
        if self.sweep_worker:
            self.sweep_worker.stop()

    def on_sweep_point(self, i, f, r, theta, x, y):
        # Live-update the plot as each point is measured.
        self.sweep_freqs[i] = f
        self.sweep_R[i] = r
        self.sweep_theta[i] = theta
        self.sweep_X[i] = x
        self.sweep_Y[i] = y
        valid = ~np.isnan(self.sweep_freqs)
        self.sweep_amp_curve.setData(self.sweep_freqs[valid], self.sweep_R[valid])
        self.sweep_phase_curve.setData(self.sweep_freqs[valid], self.sweep_theta[valid])

    def on_sweep_progress(self, i, total):
        self.sweep_progress.setMaximum(total)
        self.sweep_progress.setValue(i)

    def on_sweep_finished(self, result):
        self.run_sweep_btn.setEnabled(True)
        self.stop_sweep_btn.setEnabled(False)
        freqs = result["freqs"]
        R = result["R"]
        theta = result["theta"]
        mask = np.isfinite(freqs) & np.isfinite(R) & (R > 0)
        for key in result:
            result[key] = result[key][mask]
        freqs, R, theta = result["freqs"], result["R"], result["theta"]
        if len(freqs) == 0:
            self._restore_last_sweep()
            return
        self.sweep_freqs, self.sweep_R, self.sweep_theta = freqs, R, theta
        self.sweep_X, self.sweep_Y = result["X"], result["Y"]
        self.sweep_X_sem, self.sweep_Y_sem = result["X_sem"], result["Y_sem"]
        self.sweep_metadata = result
        self.sweep_amp_curve.setData(freqs, R)
        self.sweep_phase_curve.setData(freqs, theta)
        self.detect_peaks(freqs, R, theta)
        self._save_last_sweep()

    def on_sweep_failed(self, msg):
        self.run_sweep_btn.setEnabled(True)
        self.stop_sweep_btn.setEnabled(False)
        self._restore_last_sweep()
        QMessageBox.critical(self, "스윕 오류", msg)

    # --- peak detection (Sweeper Math: Resonance Q / Center / Phase) ---
    def detect_peaks(self, freqs, R, theta):
        if len(R) < 5:
            return
        idx, fwhm_values, smooth, noise_sigma = detect_resonance_peaks(freqs, R)
        self.peak_list.clear()
        if len(idx) == 0:
            self.peak_list.addItem(
                f"유효 피크를 찾지 못했습니다 (진폭 noise σ≈{noise_sigma:.3g} V). "
                "평균/정착시간을 늘리거나 스윕 범위를 좁혀보세요."
            )
            self.peak_markers.setData([], [])
            self._peak_candidates = []
            return
        self._peak_candidates = []
        for k, i in enumerate(idx):
            f0 = freqs[i]
            bounds = three_db_bounds(freqs, smooth, i)
            fwhm = bounds[1] - bounds[0] if bounds is not None else fwhm_values[k]
            q = f0 / fwhm if fwhm > 0 else float("nan")
            phase0 = theta[i]
            self._peak_candidates.append((f0, R[i], q, phase0))
            item = QListWidgetItem(
                f"f0={f0:,.1f}Hz  Q≈{q:,.0f}  φ={phase0:.1f}°"
            )
            width_source = "−3 dB 교차점" if bounds is not None else "prominence 반폭 (−3 dB 교차점 미측정)"
            item.setToolTip(
                f"f0 = {f0:,.3f} Hz\nQ ≈ {q:,.1f}\n선폭 = {fwhm:,.2f} Hz ({width_source})\n"
                f"phase = {phase0:.2f} deg\nR = {R[i]:.4g} V"
            )
            self.peak_list.addItem(item)
        self.peak_markers.setData(
            freqs[idx], R[idx], data=np.arange(len(idx)), symbolBrush='r'
        )

    def export_sweep_csv(self):
        if len(self.sweep_freqs) == 0:
            QMessageBox.information(self, "저장할 데이터 없음", "먼저 스윕을 실행하세요.")
            return
        default_name = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "스윕 데이터 저장", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            session_id = self.measurement_metadata.get("SessionID") or (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
            particle_min, particle_max = self._parse_size_range(self.metadata_particle_size_edit.text())
            export_metadata = {
                "SessionID": session_id,
                "Mounting_condition": self.metadata_mounting_edit.text().strip(),
                "Clamp_torque_contact": self.metadata_clamp_edit.text().strip(),
                "Fluid": self.metadata_fluid_edit.text().strip(),
                "Triton_concentration": self.metadata_triton_edit.text().strip(),
                "Triton_basis": self.metadata_triton_basis_combo.currentText(),
                "Particle_material": self.metadata_particle_edit.text().strip(),
                "Particle_size_min_um": particle_min,
                "Particle_size_max_um": particle_max,
                "Effective_travel_distance_mm": self.metadata_distance_spin.value(),
            }
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                for key, value in export_metadata.items():
                    writer.writerow(["# Metadata", key, value])
                writer.writerow(["Frequency_Hz", "Amplitude_R_Vrms", "Phase_deg", "X_Vrms", "Y_Vrms", "X_SEM", "Y_SEM", "Bandwidth_3dB_Hz", "TC_s", "N_samples"])
                metadata = self.sweep_metadata
                for row in zip(self.sweep_freqs, self.sweep_R, self.sweep_theta, self.sweep_X, self.sweep_Y,
                               self.sweep_X_sem, self.sweep_Y_sem,
                               metadata.get("bandwidths", []), metadata.get("time_constants", []),
                               metadata.get("n_samples", [])):
                    writer.writerow(row)

            peaks_path = None
            if self._peak_candidates:
                peaks_path = path[:-4] + "_peaks.csv" if path.lower().endswith(".csv") else path + "_peaks.csv"
                with open(peaks_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    for key, value in export_metadata.items():
                        writer.writerow(["# Metadata", key, value])
                    writer.writerow(["f0_Hz", "R_V", "Q", "Phase_deg"])
                    for f0, r0, q, ph in self._peak_candidates:
                        writer.writerow([f0, r0, q, ph])

            msg = f"저장됨: {path}\n({len(self.sweep_freqs)} 포인트)"
            if peaks_path:
                msg += f"\n피크: {peaks_path} ({len(self._peak_candidates)}개)"
            QMessageBox.information(self, "저장 완료", msg)
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    def on_peak_marker_clicked(self, scatter, points):
        if not points:
            return
        idx = points[0].data()
        if idx is None:
            return
        self.peak_list.setCurrentRow(int(idx))
        self.apply_peak(int(idx), show_msg=False)

    def on_peak_list_row_changed(self, row):
        self._highlight_peak(row)

    def _highlight_peak(self, row):
        if not self._peak_candidates or row < 0 or row >= len(self._peak_candidates):
            return
        freqs_r = np.array([c[0] for c in self._peak_candidates])
        Rs = np.array([c[1] for c in self._peak_candidates])
        brushes = ['r'] * len(self._peak_candidates)
        sizes = [12] * len(self._peak_candidates)
        brushes[row] = 'y'
        sizes[row] = 18
        self.peak_markers.setData(
            freqs_r, Rs, data=np.arange(len(self._peak_candidates)),
            symbolBrush=brushes, size=sizes,
        )

    def apply_peak(self, row, show_msg=True):
        if row < 0 or row >= len(self._peak_candidates):
            if show_msg:
                QMessageBox.information(self, "피크 선택", "먼저 피크를 선택하세요 (목록 또는 그래프의 점을 클릭).")
            return
        fit = self.fit_selected_peak(row, show_msg=False)
        if fit is None:
            if show_msg:
                QMessageBox.warning(self, "PLL 파라미터 적용", "복소 공진 fitting이 성공한 피크만 PLL에 적용할 수 있습니다.")
            return
        self._highlight_peak(row)
        self.pll_center_spin.setValue(fit.f0_hz)
        self.pll_q_spin.setValue(fit.q)
        self.pll_setpoint_spin.setValue(fit.phase_deg)
        if show_msg:
            QMessageBox.information(
                self, "PLL 파라미터 적용됨",
                f"복소 fitting 결과를 적용했습니다.\n"
                f"f0 = {fit.f0_hz:,.4f} Hz, Q = {fit.q:,.2f}, Setpoint = {fit.phase_deg:.2f} deg\n"
                f"이제 'fitted 공진기 모델로 PID 최적화'를 실행해 확인한 뒤 PLL을 활성화하세요."
            )

    def fit_selected_peak(self, row, show_msg=True):
        if row < 0 or row >= len(self._peak_candidates) or len(self.sweep_X) < 9:
            if show_msg:
                QMessageBox.warning(self, "복소 fitting", "먼저 유효한 스윕과 피크를 선택하세요.")
            return None
        f0, _, coarse_q, _ = self._peak_candidates[row]
        df = abs(self.sweep_freqs[1] - self.sweep_freqs[0]) if len(self.sweep_freqs) > 1 else 1.0
        span = float(np.ptp(self.sweep_freqs))
        linewidth = f0 / coarse_q if np.isfinite(coarse_q) and coarse_q > 0 else span / 8.0
        # The coarse FWHM estimate can be badly inflated when neighboring
        # modes overlap.  Limit the selected peak's fitting window to the
        # midpoint toward its nearest detected neighbor, otherwise a much
        # larger distant mode can pull the optimizer away from the click.
        neighbors = []
        if row > 0:
            neighbors.append(f0 - self._peak_candidates[row - 1][0])
        if row + 1 < len(self._peak_candidates):
            neighbors.append(self._peak_candidates[row + 1][0] - f0)
        nearest_neighbor = min((distance for distance in neighbors if distance > 0), default=np.inf)
        unconstrained_half_window = max(3.0 * linewidth, 8.0 * df)
        isolation_half_window = 0.45 * nearest_neighbor if np.isfinite(nearest_neighbor) else np.inf
        half_window = min(unconstrained_half_window, isolation_half_window)
        half_window = max(half_window, 4.0 * df)
        peak_index = int(np.argmin(np.abs(self.sweep_freqs - f0)))
        _, _, smooth, _ = detect_resonance_peaks(self.sweep_freqs, self.sweep_R)
        bounds = three_db_bounds(self.sweep_freqs, smooth, peak_index)
        if bounds is not None:
            # The -3 dB crossings establish the core. Keep one more half-width
            # on each side to identify the complex background and phase slope.
            low = max(f0 - half_window, 2.0 * bounds[0] - f0)
            high = min(f0 + half_window, 2.0 * bounds[1] - f0)
            mask = (self.sweep_freqs >= low) & (self.sweep_freqs <= high)
        else:
            mask = np.abs(self.sweep_freqs - f0) <= half_window
        if np.count_nonzero(mask) < 9:
            order = np.argsort(np.abs(self.sweep_freqs - f0))[:max(9, min(41, len(self.sweep_freqs)))]
            mask = np.zeros(len(self.sweep_freqs), dtype=bool)
            mask[order] = True
        self.last_resonance_fit = None
        self.sweep_fit_amp_curve.setData([], [])
        self.sweep_fit_phase_curve.setData([], [])
        try:
            fit = fit_complex_resonance(
                self.sweep_freqs[mask], self.sweep_X[mask], self.sweep_Y[mask],
                self.sweep_X_sem[mask], self.sweep_Y_sem[mask],
                f0_guess=f0,
                # A selected peak may refine by several sweep points, but it
                # cannot silently migrate toward an adjacent resonance.
                f0_bounds=(f0 - 0.75 * half_window, f0 + 0.75 * half_window),
            )
        except Exception as e:
            self.fit_result_label.setText(f"Fitting 실패: {e}")
            if show_msg:
                QMessageBox.warning(self, "복소 fitting 실패", str(e))
            return None
        if not fit.success or not np.isfinite(fit.q) or fit.q <= 0:
            self.fit_result_label.setText(f"Fitting 실패: {fit.message}")
            if show_msg:
                QMessageBox.warning(self, "복소 fitting 실패", fit.message)
            return None
        display_f = self.sweep_freqs[mask]
        z_fit = model_response(display_f, fit.parameters, float(np.mean(display_f)),
                               fit.resonance_sign)
        quality = assess_fit_quality(
            display_f, self.sweep_X[mask], self.sweep_Y[mask], fit,
        )
        quality_text = (
            f"R²={quality.complex_r_squared:.3f}, "
            f"선폭점={quality.linewidth_bins:.2f}"
        )
        if not quality.accepted:
            self.last_resonance_fit = None
            reason_text = "\n".join(f"• {reason}" for reason in quality.reasons)
            self.fit_result_label.setText(
                f"Fitting 거부: f0={fit.f0_hz:,.5f} Hz, Q={fit.q:,.2f}, "
                f"Γ={fit.linewidth_hz:.4g} Hz, {quality_text}"
            )
            if show_msg:
                QMessageBox.warning(
                    self, "복소 fitting 품질 부족",
                    "수치 최적화는 끝났지만 PLL에 사용할 수 없는 결과입니다.\n\n" + reason_text,
                )
            return None
        self.last_resonance_fit = fit
        self.sweep_fit_amp_curve.setData(display_f, np.abs(z_fit))
        self.sweep_fit_phase_curve.setData(display_f, np.angle(z_fit, deg=True))
        self.fit_result_label.setText(
            f"Fitting: f0={fit.f0_hz:,.5f}±{fit.f0_std_hz:.3g} Hz, "
            f"Q={fit.q:,.2f}±{fit.q_std:.2g}, Γ={fit.linewidth_hz:.4g} Hz, "
            f"dφ/df={fit.phase_slope_deg_per_hz:.4g} deg/Hz, {quality_text} "
            f"(fit window {display_f[0]:,.1f}–{display_f[-1]:,.1f} Hz)"
        )
        if show_msg:
            QMessageBox.information(self, "복소 공진 fitting 완료", self.fit_result_label.text())
        return fit

    def calibrate_pll_phase_detector(self):
        """Calibrate the PLL internal detector through its public ERROR node.

        HF2LI's PLL phase detector is numbered 7 in LabOne, but it is an
        internal block, not a user-addressable ``/demods/6`` node.  ERROR is
        defined as setpoint minus that detector input, so an ERROR-versus-
        frequency scan gives both the detector phase slope and its phase at
        the fitted resonance without touching any nonexistent demod nodes.
        """
        if not self.hf:
            return
        if self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "PLL 보정", "PLL을 먼저 비활성화한 뒤 보정하세요.")
            return
        if not (self.output_on_btn.isChecked() and self.amp_enable_btn.isChecked()):
            QMessageBox.warning(self, "PLL 보정", "구동 Output On 및 Amp를 먼저 켜고 안정된 응답을 확인하세요.")
            return
        fit = self.last_resonance_fit
        if fit is None and self.peak_list.currentRow() >= 0:
            fit = self.fit_selected_peak(self.peak_list.currentRow(), show_msg=False)
        if fit is None:
            QMessageBox.warning(self, "PLL 보정", "먼저 선택 피크의 복소 fitting을 완료하세요.")
            return
        previous_freq = None
        try:
            previous_freq = self.hf.get_osc_freq()
            reference_setpoint = float(self.pll_setpoint_spin.value())
            input_index = int(self.pll_input_combo.currentData())
            input_name = self.pll_input_combo.currentText()
            self.hf.set_pll_input(input_index)
            half_span = max(3.0 * fit.linewidth_hz, 0.25)
            # HF2LI starts updating its internal PLL phase detector only
            # while the PLL is enabled.  P=I=D=0 leaves its frequency output
            # at the programmed center, so this is a safe detector-only scan.
            self.hf.configure_pll(
                fit.f0_hz, self.pll_order_spin.value(), self.pll_tc_spin.value(),
                max(self.pll_range_spin.value(), 2.0 * half_span),
                reference_setpoint, 0.0, 0.0, 0.0,
            )
            self.hf.set_pll_enabled(True)
            time.sleep(max(0.05, 16.0 * self.pll_tc_spin.value()))
            freqs = np.linspace(fit.f0_hz - half_span, fit.f0_hz + half_span, num=9)
            errors = []
            settle = max(0.02, 16.0 * self.pll_tc_spin.value())
            for frequency in freqs:
                self.hf.set_pll_center_frequency(float(frequency))
                time.sleep(settle)
                errors.append(self.hf.get_pll_error())
            # ERROR can be reported in the principal phase interval; unwrap
            # it before fitting across the resonance phase transition.
            unwrapped_errors = np.rad2deg(np.unwrap(np.deg2rad(errors)))
            error_slope, error_intercept = np.polyfit(freqs, unwrapped_errors, 1)
            # ERROR = setpoint - detector phase.
            slope = -error_slope
            detector_phase = reference_setpoint - (error_slope * fit.f0_hz + error_intercept)
            setpoint = (detector_phase + 180.0) % 360.0 - 180.0
            if abs(slope) < 1e-6:
                raise ValueError(
                    f"PLL ERROR가 주파수에 따라 변하지 않습니다. 현재 PLL 입력은 '{input_name}'입니다. "
                    "sense piezo가 연결된 입력을 선택하고, 해당 입력의 range/배선과 공진점을 확인하세요."
                )
            self.pll_calibrated_slope = float(slope)
            self.pll_calibrated_setpoint = float(setpoint)
            self.pll_setpoint_spin.setValue(setpoint)
            self.pll_design_label.setText(
                f"PLL ERROR 보정 ({input_name}): dφ/df={slope:.5g} deg/Hz, "
                f"resonance setpoint={setpoint:.3f}°. 이제 P-only PID 최적화를 실행하세요."
            )
        except Exception as e:
            QMessageBox.warning(self, "PLL phase detector 보정 실패", str(e))
        finally:
            try:
                self.hf.set_pll_enabled(False)
                self.hf.set_pll_center_frequency(fit.f0_hz)
                if previous_freq is not None:
                    self.hf.set_frequency(previous_freq)
            except Exception:
                pass
    def test_pll_feedback_sign(self):
        """Empirically select PID polarity with a bounded P-only phase step."""
        if not self.hf:
            return
        if self.pll_enable_btn.isChecked():
            QMessageBox.warning(self, "Feedback sign test", "PLL을 먼저 비활성화하세요.")
            return
        if not (self.output_on_btn.isChecked() and self.amp_enable_btn.isChecked()):
            QMessageBox.warning(self, "Feedback sign test", "구동 Output On 및 Amp를 먼저 켜세요.")
            return
        fit = self.last_resonance_fit
        if fit is None:
            QMessageBox.warning(self, "Feedback sign test", "먼저 피크 fitting과 PLL phase detector 보정을 완료하세요.")
            return
        slope = self.pll_calibrated_slope
        if slope is None or abs(slope) < 1e-6:
            QMessageBox.warning(self, "Feedback sign test", "먼저 실제 PLL phase detector 보정을 실행하세요.")
            return
        base_setpoint = self.pll_calibrated_setpoint
        if base_setpoint is None:
            base_setpoint = self.pll_setpoint_spin.value()
        step_deg = 5.0
        stepped_setpoint = (base_setpoint + step_deg + 180.0) % 360.0 - 180.0
        # About 0.2 open-loop DC gain: enough to distinguish direction, but
        # intentionally far below a normal acquisition gain.
        p_magnitude = max(1e-8, 0.2 / abs(slope))
        test_range = min(self.pll_range_spin.value(), max(5.0, 2.0 * self.pll_target_bw_spin.value()))
        scores = {}
        try:
            self.hf.set_pll_input(int(self.pll_input_combo.currentData()))
            for sign in (1.0, -1.0):
                self.hf.configure_pll(
                    fit.f0_hz, self.pll_order_spin.value(), self.pll_tc_spin.value(),
                    test_range, stepped_setpoint, sign * p_magnitude, 0.0, 0.0,
                )
                self.hf.set_pll_enabled(True)
                time.sleep(max(0.10, 16.0 * self.pll_tc_spin.value()))
                errors, deltas = [], []
                for _ in range(10):
                    errors.append(abs(self.hf.get_pll_error()))
                    deltas.append(abs(self.hf.get_pll_freq_delta()))
                    time.sleep(0.05)
                scores[sign] = (float(np.mean(errors[-5:])), max(deltas))
                self.hf.set_pll_enabled(False)
                self.hf.set_frequency(fit.f0_hz)
            preferred = min(scores, key=lambda key: scores[key][0])
            expected = 1.0 if slope > 0 else -1.0
            self.pll_invert_sign_check.setChecked(preferred != expected)
            self.pll_setpoint_spin.setValue(base_setpoint)
            self.pll_design_label.setText(
                f"P-only sign test: +P RMS={scores[1.0][0]:.3g}°, "
                f"−P RMS={scores[-1.0][0]:.3g}°. "
                f"선택 부호={'+' if preferred > 0 else '−'}P; 이후 PID 최적화에 반영됩니다."
            )
        except Exception as e:
            QMessageBox.warning(self, "Feedback sign test 실패", str(e))
        finally:
            try:
                self.hf.set_pll_enabled(False)
                self.hf.set_frequency(fit.f0_hz)
                self.pll_setpoint_spin.setValue(base_setpoint)
            except Exception:
                pass

    # --- PLL model-based P/PI/PID design ---
    def estimate_pll_p(self):
        fit = self.last_resonance_fit
        if fit is None and self.peak_list.currentRow() >= 0:
            fit = self.fit_selected_peak(self.peak_list.currentRow(), show_msg=False)
        if fit is None:
            QMessageBox.warning(self, "PID 최적화", "먼저 피크를 선택하고 복소 공진 fitting을 완료하세요.")
            return
        try:
            loop_rate = self.hf.get_pll_rate() if self.hf else 10000.0
            if loop_rate <= 0:
                loop_rate = 10000.0
            design = design_pll(
                phase_slope_deg_per_hz=(self.pll_calibrated_slope
                                         if self.pll_calibrated_slope is not None
                                         else fit.phase_slope_deg_per_hz),
                f0_hz=fit.f0_hz, q=fit.q,
                demod_tc_s=self.pll_tc_spin.value(),
                demod_order=self.pll_order_spin.value(),
                loop_rate_hz=loop_rate,
                target_bandwidth_hz=self.pll_target_bw_spin.value(),
                minimum_phase_margin_deg=self.pll_pm_spin.value(),
                allow_integral=self.pll_integral_check.isChecked(),
                allow_derivative=self.pll_derivative_check.isChecked(),
            )
        except Exception as e:
            QMessageBox.warning(self, "PID 최적화 실패", str(e))
            return
        slope_source = "실제 PLL detector 보정" if self.pll_calibrated_slope is not None else "DEMODS/0 fitting (보정 필요)"
        if not design.stable:
            self.pll_design_label.setText(
                f"PID 설계 거부: 계산 BW={design.achieved_bandwidth_hz:.3g} Hz, "
                f"PM={design.phase_margin_deg:.1f}°, GM={design.gain_margin_db:.1f} dB, "
                f"Ms={design.sensitivity_peak:.3g}; {slope_source}; {design.message}"
            )
            QMessageBox.warning(
                self, "불안정 PID 결과 적용 안 함",
                self.pll_design_label.text() +
                "\n\n기존 P/I/D 값은 변경하지 않았습니다. 현재처럼 Target BW=3 Hz, "
                "TC=0.002 s, order=2인 조합에서는 먼저 TC를 늘리거나 실제 PLL detector 보정과 "
                "P-only sign test를 수행하세요.",
            )
            return
        sign = -1.0 if self.pll_invert_sign_check.isChecked() else 1.0
        self.pll_p_spin.setValue(sign * design.p_hz_per_deg)
        self.pll_i_spin.setValue(sign * design.i_hz2_per_deg)
        self.pll_d_spin.setValue(sign * design.d_per_deg)
        status = "안정" if design.stable else "경고"
        self.pll_design_label.setText(
            f"PID 설계 ({status}): BW={design.achieved_bandwidth_hz:.3g} Hz, "
            f"PM={design.phase_margin_deg:.1f}°, GM={design.gain_margin_db:.1f} dB, "
            f"Ms={design.sensitivity_peak:.3g}; {slope_source}; {design.message}"
        )
        range_needed = 5.0 * self.pll_target_bw_spin.value()
        if self.pll_range_spin.value() < range_needed:
            self.pll_design_label.setText(self.pll_design_label.text() +
                                          f"  PLL Range는 최소 {range_needed:.3g} Hz를 권장합니다.")
        QMessageBox.information(
            self, "모델 기반 PID 설계 완료",
            self.pll_design_label.text() + "\n\n계수는 아직 장비에 적용되지 않았습니다. PLL 활성화 시 적용됩니다."
        )

    # --- PLL ---
    def toggle_pll(self, checked):
        if not self.hf:
            self.pll_enable_btn.setChecked(False)
            return
        if checked:
            try:
                self.hf.set_pll_input(int(self.pll_input_combo.currentData()))
                self.hf.configure_pll(
                    self.pll_center_spin.value(),
                    self.pll_order_spin.value(),
                    self.pll_tc_spin.value(),
                    self.pll_range_spin.value(),
                    self.pll_setpoint_spin.value(),
                    self.pll_p_spin.value(),
                    self.pll_i_spin.value(),
                    self.pll_d_spin.value(),
                )
                if not (self.output_on_btn.isChecked() and self.amp_enable_btn.isChecked()):
                    self._ensure_drive_active()
                self.hf.set_pll_enabled(True)
            except ZiAPIError as e:
                QMessageBox.critical(self, "PLL 설정 오류", str(e))
                self.pll_enable_btn.setChecked(False)
                return

            self.pll_enable_btn.setText("PLL 비활성화")
            self.pll_rail_count = 0
            self.pll_status_worker = PLLStatusWorker()
            self.pll_status_worker.sample_ready.connect(self.on_pll_status)
            self.pll_status_worker.failed.connect(self.on_pll_failed)
            self.pll_status_worker.start()

            # PLL only controls parameters; the actual frequency/amplitude
            # trend is watched on the Plotter, so make sure it's running.
            if self.plotter_worker is None or not self.plotter_worker.isRunning():
                self.start_plotter()
            self.tabs.setCurrentIndex(3 if self.live_particle_enabled.isChecked() else 2)
        else:
            if self.pll_status_worker:
                self.pll_status_worker.stop()
                self.pll_status_worker.wait(1000)
            self._latest_pll_locked = False
            self._latest_pll_error_deg = np.nan
            self._latest_pll_correction_hz = np.nan
            self._latest_pll_osc_frequency_hz = np.nan
            self._latest_pll_read_latency_s = np.nan
            try:
                self.hf.set_pll_enabled(False)
            except Exception:
                pass
            self.pll_enable_btn.setText("PLL 활성화")
            self.pll_locked_label.setText("PLL: 비활성")
            self.pll_locked_label.setStyleSheet("font-weight: bold; color: black;")
            self.pll_big_locked.setText("PLL: 비활성")
            self.pll_big_locked.setStyleSheet("font-size: 32px; font-weight: bold; color: black;")

    def on_pll_status(self, locked, error_deg, freq_delta, osc_f, read_latency_s):
        self._latest_pll_locked = bool(locked)
        self._latest_pll_error_deg = float(error_deg)
        self._latest_pll_correction_hz = float(freq_delta)
        self._latest_pll_osc_frequency_hz = float(osc_f)
        self._latest_pll_read_latency_s = float(read_latency_s)
        self._latest_pll_status_monotonic = time.monotonic()
        self.pll_big_error.setText(f"{error_deg:.4f}")
        self.pll_big_shift.setText(f"{freq_delta:+.4f}")
        self.pll_big_freq.setText(f"{osc_f:,.4f}")

        if locked:
            text = f"PLL: LOCKED  (f = {osc_f:,.3f} Hz)"
            color = "green"
        else:
            text = f"PLL: 잠금 안됨  (f = {osc_f:,.3f} Hz)"
            color = "#c0392b"
        self.pll_locked_label.setText(text)
        self.pll_locked_label.setStyleSheet(f"font-weight: bold; color: {color};")
        self.pll_big_locked.setText("LOCKED" if locked else "잠금 안됨")
        self.pll_big_locked.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color};")

        # A sustained rail condition is a reliable sign of wrong feedback
        # polarity or excessive integral action.  Disable before the loop can
        # keep the piezo at an unintended edge frequency.
        if self.pll_enable_btn.isChecked():
            if abs(freq_delta) >= 0.9 * self.pll_range_spin.value():
                self.pll_rail_count += 1
            else:
                self.pll_rail_count = 0
            if self.pll_rail_count >= 3:
                self.pll_enable_btn.setChecked(False)
                try:
                    # Disabling the HF2 PLL freezes the oscillator at its
                    # last value.  Explicitly return from the capture-range
                    # edge to the fitted/selected center for safe recovery.
                    self.hf.set_frequency(self.pll_center_spin.value())
                except Exception:
                    pass
                QMessageBox.warning(
                    self, "PLL 자동 해제",
                    "PLL 주파수 보정이 capture range 끝에 3회 연속 도달했습니다.\n"
                    "P 부호를 반전하거나 Target BW/P를 낮추고, I=0으로 다시 확인하세요."
                )

    def on_pll_failed(self, msg):
        self.pll_enable_btn.setChecked(False)
        QMessageBox.critical(self, "PLL 모니터링 오류", msg)

    # --- Plotter (Sample Frequency + Sample R together, vs time) ---
    def _capture_measurement_metadata(self, baseline):
        """Snapshot all settings that contextualize a raw tracking record."""
        pll_rate = "Unavailable"
        if self.hf:
            try:
                pll_rate = self.hf.get_pll_rate()
            except Exception:
                pass
        set_a, set_b = self._parse_pair(self.metadata_set_pressure_edit.text())
        actual_a, actual_b = self._parse_pair(self.metadata_actual_pressure_edit.text())
        particle_min, particle_max = self._parse_size_range(self.metadata_particle_size_edit.text())
        pressure_source = self.metadata_pressure_source_combo.currentText()
        if self.pressure_monitor_btn.isChecked():
            self._expire_pressure_sample()
            actual_a = self.measured_pressure_a if self.measured_pressure_a is not None else np.nan
            actual_b = self.measured_pressure_b if self.measured_pressure_b is not None else np.nan
            pressure_source = "ITV0050 monitor / NI AI / nominal conversion"
        try:
            temperature = float(self.metadata_temperature_edit.text())
        except ValueError:
            temperature = np.nan
        self.measured_pressure_a = actual_a if np.isfinite(actual_a) else None
        self.measured_pressure_b = actual_b if np.isfinite(actual_b) else None
        self.temperature_c = temperature if np.isfinite(temperature) else None
        return {
            "SessionID": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
            "Measurement_mode": "Baseline (PLL tracking only)" if baseline else "Tracking",
            "Measurement_start_timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "Plotter_requested_interval_ms": 1000.0 / self.plotter_rate_spin.value(),
            "Acquisition_mode": "Buffered demod subscribe/poll; device timestamps",
            "Demod_requested_rate_Hz": self.plotter_rate_spin.value(),
            "Plotter_GUI_refresh_rate_Hz": 20.0,
            "PLL_status_logging": "Cached low-rate PLLStatusWorker values; not read per demod sample",
            "PLL_update_rate_Hz": pll_rate,
            "PLL_Kp_Hz_per_deg": self.pll_p_spin.value(),
            "PLL_Ki_Hz2_per_deg": self.pll_i_spin.value(),
            "PLL_Kd_per_deg": self.pll_d_spin.value(),
            "PLL_target_bandwidth_Hz": self.pll_target_bw_spin.value(),
            "PLL_filter_order": self.pll_order_spin.value(),
            "PLL_filter_time_constant_s": self.pll_tc_spin.value(),
            "PLL_capture_range_Hz": self.pll_range_spin.value(),
            "PLL_target_phase_deg": self.pll_setpoint_spin.value(),
            "Excitation_amplitude_Vpk": self.drive_v_spin.value(),
            "Software_smoothing": "None; raw samples are retained",
            "Plot_display_window_s": self.plotter_window_spin.value(),
            "Mounting_condition": self.metadata_mounting_edit.text().strip(),
            "Clamp_torque_contact": self.metadata_clamp_edit.text().strip(),
            "Fluid": self.metadata_fluid_edit.text().strip(),
            "Triton_concentration": self.metadata_triton_edit.text().strip(),
            "Triton_basis": self.metadata_triton_basis_combo.currentText(),
            "Particle_material": self.metadata_particle_edit.text().strip(),
            "Particle_size_min_um": particle_min,
            "Particle_size_max_um": particle_max,
            "Vial_set_pressure_A_kPa": set_a,
            "Vial_set_pressure_B_kPa": set_b,
            "Measured_pressure_A_kPa": actual_a,
            "Measured_pressure_B_kPa": actual_b,
            "Measured_differential_pressure_kPa": (actual_a - actual_b if np.isfinite(actual_a) and np.isfinite(actual_b) else np.nan),
            "Pressure_source": pressure_source,
            "Pressure_monitor_channels": f"{DAQ_DEVICE}/ai0:1 (A, B); RSE; +/-10 V",
            "Pressure_monitor_conversion": "kPa = 1 + (V - 1) * 899 / 4; valid 1..5 V; uncalibrated",
            "Pressure_monitor_logging": "Latest host reading (~5 Hz), repeated per received demod batch; not hardware synchronized; blank on error or age >1 s",
            "Pressure_monitor_events": [],
            "Temperature_C": temperature,
            "Temperature_source": "manual input" if np.isfinite(temperature) else "not recorded",
            "Effective_travel_distance_mm": self.metadata_distance_spin.value(),
        }

    def _reset_plotter_logs(self):
        if self.plotter_spool is not None:
            self.plotter_spool.discard()
        self.plotter_spool = PlotterSpool()
        self.calibration_t, self.calibration_f = [], []
        self._calibration_sample_index = 0
        self.analysis_t, self.analysis_f = [], []
        self._analysis_sample_index = 0
        self._display_sample_index = 0
        self._log_sample_index = 0
        self._recent_raw_dt_s = []
        self._last_raw_t = None
        self.plot_t, self.plot_R, self.plot_theta, self.plot_f, self.plot_x, self.plot_y = \
            [], [], [], [], [], []
        self.log_t, self.log_R, self.log_theta, self.log_f, self.log_x, self.log_y = \
            [], [], [], [], [], []
        self.log_pll_target_phase, self.log_pll_phase_error, self.log_pll_frequency_command = [], [], []
        self.log_pll_correction, self.log_pll_locked = [], []
        self.log_pressure_a, self.log_pressure_b = [], []
        self.log_flow_direction, self.log_switch_state, self.log_switch_event = [], [], []
        self.log_measured_pressure_a, self.log_measured_pressure_b, self.log_temperature_c = [], [], []
        self.log_demod_read_latency_s, self.log_pll_read_latency_s, self.log_pll_status_age_s = [], [], []
        self.received_sample_count = 0
        self.ui_refresh_durations_s = []

    def reset_live_particle_detector(self, *_args):
        """Clear live results and start a fresh particle-free calibration."""
        self.calibration_t, self.calibration_f = [], []
        self._calibration_sample_index = 0
        self.live_particle_generation += 1
        self.live_particle_thresholds = None
        self.live_particle_events = []
        self.live_particle_last_analyzed_s = 0.0
        self._last_live_analysis_t = -np.inf
        self.live_event_list.clear()
        self.live_event_list.addItem("검출된 후보가 여기에 실시간으로 표시됩니다.")
        self.live_particle_markers.setData([], [])
        self.live_filtered_frequency_curve.setData([], [])
        self.live_count_value.setText("0")
        self.live_latest_value.setText("—")
        self.live_export_btn.setEnabled(False)
        if self.baseline_active:
            self.live_state_value.setText("BASELINE")
            self.live_calibration_progress.setValue(0)
            self.live_calibration_progress.setFormat("Baseline 모드에서는 검출 중지")
            self.live_detector_status.setStyleSheet("font-weight: 700; color: #60708a;")
            self.live_detector_status.setText("Baseline 측정 보호를 위해 파티클 검출을 실행하지 않습니다.")
        elif self.live_particle_enabled.isChecked():
            self.live_detector_status.setStyleSheet("font-weight: 700; color: #147d64;")
            self.live_state_value.setText("학습 대기")
            self.live_calibration_progress.setValue(0)
            self.live_calibration_progress.setFormat("blank 학습 대기")
            self.live_detector_status.setText(
                "준비: Plotter 시작 후 처음 구간은 입자 없이 유지하세요."
            )
        else:
            self.live_detector_status.setStyleSheet("font-weight: 700; color: #60708a;")
            self.live_state_value.setText("꺼짐")
            self.live_calibration_progress.setValue(0)
            self.live_calibration_progress.setFormat("실시간 검출 꺼짐")
            self.live_detector_status.setText("실시간 파티클 검출이 꺼져 있습니다.")

    def _schedule_live_particle_analysis(self, latest_t):
        if not self.live_particle_enabled.isChecked() or self.baseline_active or len(self.log_t) < 30:
            return
        warmup = self.live_calibration_spin.value()
        start_t = self.calibration_t[0] if self.calibration_t else float(self.log_t[0])
        elapsed = max(0.0, latest_t - start_t)
        if self.live_particle_thresholds is None:
            progress = min(100, int(round(100.0 * elapsed / warmup)))
            self.live_calibration_progress.setValue(progress)
            self.live_calibration_progress.setFormat(f"blank 학습 {min(elapsed, warmup):.1f} / {warmup:.1f} s")
            self.live_state_value.setText("BLANK 학습")
            self.live_detector_status.setText(
                "초기 noise를 학습 중입니다. 이 구간에는 입자를 통과시키지 마세요."
            )
            if elapsed < warmup:
                return
        if self.live_particle_worker is not None and self.live_particle_worker.isRunning():
            return
        if latest_t - self._last_live_analysis_t < 0.25:
            return
        self._last_live_analysis_t = latest_t
        widths = (.03, .05, .10, .20)
        # Local depth fits the baseline from samples 0.25-0.55 s away.
        margin = .60
        if self.live_particle_thresholds is None:
            right_t = start_t + warmup
            time_s, frequency_hz = self._read_initial_calibration(right_t)
            if len(time_s) < 30:
                return
            settings = ParticleSettings(
                widths=widths, reference_start=start_t + margin,
                reference_end=right_t - margin,
                sigma_multiplier=self.live_sigma_spin.value(),
                floor_hz=self.live_floor_spin.value(),
            )
        else:
            segment_start = max(start_t, latest_t - 20.0)
            left = bisect_left(self.analysis_t, segment_start)
            time_s = np.asarray(self.analysis_t[left:], dtype=float)
            frequency_hz = np.asarray(self.analysis_f[left:], dtype=float)
            if len(time_s) < 30:
                return
            analyze_end = latest_t - margin
            analyze_start = max(float(time_s[0]) + margin,
                                self.live_particle_last_analyzed_s - 0.35)
            if analyze_end <= analyze_start:
                return
            settings = ParticleSettings(
                widths=widths, analyze_start=analyze_start, analyze_end=analyze_end,
                sigma_multiplier=self.live_sigma_spin.value(),
                floor_hz=self.live_floor_spin.value(),
            )
        worker = LiveParticleAnalysisWorker(
            time_s, frequency_hz, settings, self.live_particle_thresholds,
            savgol={"enabled": self.live_savgol_enabled.isChecked(),
                    "window_length": self.live_savgol_window_spin.value(),
                    "polyorder": self.live_savgol_order_spin.value()})
        worker.generation = self.live_particle_generation
        worker.completed.connect(self.on_live_particle_analysis)
        worker.failed.connect(self.on_live_particle_failed)
        self.live_particle_worker = worker
        worker.start()

    def _read_initial_calibration(self, right_t):
        """Use the bounded first blank interval after the live logs roll."""
        right = bisect_left(self.calibration_t, right_t)
        return (np.asarray(self.calibration_t[:right], float),
                np.asarray(self.calibration_f[:right], float))

    def on_live_particle_analysis(self, payload):
        worker = self.sender()
        if getattr(worker, 'generation', -1) != self.live_particle_generation:
            return
        if payload["mode"] == "calibration":
            self.live_particle_thresholds = payload["thresholds"]
            self.live_calibration_progress.setValue(100)
            self.live_calibration_progress.setFormat("blank 학습 완료 · 실시간 감시 중")
            self.live_state_value.setText("감시 중")
            threshold_text = min(row['threshold_hz'] for row in self.live_particle_thresholds)
            sg_window = payload.get("savgol_window_samples", 0)
            sg_ms = payload.get("savgol_effective_window_ms", 0)
            sg_text = (f" · SG {sg_window}표본 ≈ {sg_ms:.2f} ms" if sg_window else
                       " · SG 미적용" if self.live_savgol_enabled.isChecked() else "")
            self.live_detector_status.setText(
                f"검출 준비 완료 · short 깊이 기준 · 적용 문턱 {threshold_text:.3g} Hz{sg_text}"
            )
            return

        self.live_particle_last_analyzed_s = max(
            self.live_particle_last_analyzed_s, float(payload["analyze_end"]))
        new_events = []
        for event in payload.get("events", []):
            separation = .30 if event['family'] == 'short' else .80
            duplicate = any(
                saved['family'] == event['family'] and
                abs(saved['time_s'] - event['time_s']) < separation
                for saved in self.live_particle_events
            )
            if duplicate:
                continue
            index = bisect_left(self.analysis_t, event['time_s'])
            index = min(max(index, 0), len(self.analysis_f) - 1)
            event = dict(event, frequency_hz=float(self.analysis_f[index]),
                         savgol_enabled=int(self.live_savgol_enabled.isChecked()),
                         savgol_window_length=self.live_savgol_window_spin.value(),
                         savgol_polyorder=self.live_savgol_order_spin.value(),
                         savgol_window_samples=payload.get("savgol_window_samples", 0),
                         savgol_effective_window_ms=payload.get("savgol_effective_window_ms", 0),
                         detected_at_local=datetime.now().astimezone().isoformat(timespec='milliseconds'))
            self.live_particle_events.append(event)
            new_events.append(event)
        self.live_particle_events.sort(key=lambda event: event['time_s'])
        if new_events:
            if self.live_event_list.count() == 1 and "여기에" in self.live_event_list.item(0).text():
                self.live_event_list.clear()
            for event in new_events:
                number = self.live_particle_events.index(event) + 1
                family = "SHORT" if event['family'] == 'short' else "WIDE"
                self.live_event_list.addItem(
                    f"● #{number:02d}  {event['time_s']:8.3f} s   "
                    f"깊이 {event['depth_hz']:.4g} Hz   구간 {1000*event.get('duration_s',0):.0f} ms   {family}   "
                    f"S/N {event['reference_ratio']:.1f}×"
                )
            self.live_event_list.scrollToBottom()
            latest = max(new_events, key=lambda event: event['time_s'])
            self.live_latest_value.setText(f"{latest['depth_hz']:.4g} Hz")
            self.live_state_value.setText("검출!")
        else:
            self.live_state_value.setText("감시 중")
        self.live_count_value.setText(str(len(self.live_particle_events)))
        self.live_export_btn.setEnabled(bool(self.live_particle_events))
        self._update_live_particle_markers()

    def on_live_particle_failed(self, message):
        worker = self.sender()
        if getattr(worker, 'generation', -1) != self.live_particle_generation:
            return
        self.live_state_value.setText("확인 필요")
        self.live_detector_status.setText(f"실시간 검출 일시 중단: {message}")
        self.live_detector_status.setStyleSheet("font-weight: 700; color: #b42318;")

    def _update_live_particle_markers(self):
        if not self.live_particle_events:
            self.live_particle_markers.setData([], [])
            return
        cutoff = self.plot_t[0] if self.plot_t else -np.inf
        shown = [event for event in self.live_particle_events if event['time_s'] >= cutoff]
        self.live_particle_markers.setData(
            [event['time_s'] for event in shown],
            [event['frequency_hz'] for event in shown],
        )

    def export_live_particle_csv(self):
        if not self.live_particle_events:
            QMessageBox.information(self, "저장할 후보 없음", "아직 검출된 파티클 후보가 없습니다.")
            return
        default_name = f"live_particles_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "실시간 파티클 후보 저장", default_name, "CSV Files (*.csv)")
        if not path:
            return
        fields = ["event_index", "time_s", "peak_time_s", "event_start_s", "event_end_s",
                  "duration_s", "selection_source", "frequency_hz", "depth_hz", "contrast_hz",
                  "width_s", "family", "reference_ratio", "detection_threshold_hz",
                  "local_max_jump_hz", "quality", "savgol_enabled", "savgol_window_length",
                  "savgol_polyorder", "savgol_window_samples", "savgol_effective_window_ms",
                  "detected_at_local"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for index, event in enumerate(self.live_particle_events, 1):
                    writer.writerow({"event_index": index, **event})
            QMessageBox.information(self, "저장 완료", f"{len(self.live_particle_events)}개 후보를 저장했습니다.\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def start_plotter(self, baseline=False):
        if not self.hf:
            return
        self.baseline_active = bool(baseline)
        try:
            self._reset_plotter_logs()
        except OSError as exc:
            QMessageBox.critical(self, "기록 파일 오류", str(exc))
            return
        self.baseline_analysis_result = None
        self.baseline_analysis_source = None
        self.plotter_actual_rate_hz = np.nan
        self.plotter_actual_rate_label.setText("실제 적용: 연결 중...")
        self._last_plotter_ui_update_t = -np.inf
        self._last_baseline_ui_update_t = -np.inf
        self._last_filtered_ui_update_t = -np.inf
        self.measurement_metadata = self._capture_measurement_metadata(self.baseline_active)
        self.plotter_freq_curve.setData([], [])
        self.plotter_r_curve.setData([], [])
        self.live_filtered_frequency_curve.setData([], [])
        self.baseline_raw_curve.setData([], [])
        self.baseline_trend_curve.setData([], [])
        self.baseline_residual_curve.setData([], [])
        self.baseline_amplitude_curve.setData([], [])
        self.baseline_error_curve.setData([], [])
        self.baseline_stats_label.setText("Baseline 분석: 수집 중... (raw data, no software smoothing)")
        self.reset_live_particle_detector()
        self.tabs.setCurrentIndex(4 if self.baseline_active else 3)

        requested_rate_hz = self.plotter_rate_spin.value()
        self.plotter_worker = PlotterWorker(
            requested_rate_hz, emit_period_s=0.05, timing_enabled=self.baseline_active
        )
        self.plotter_worker.batch_ready.connect(self.on_plotter_batch)
        self.plotter_worker.stream_started.connect(self.on_plotter_stream_started)
        self.plotter_worker.failed.connect(self.on_plotter_failed)
        self.plotter_rate_spin.setEnabled(False)
        self.plotter_start_btn.setEnabled(False)
        self.plotter_stop_btn.setEnabled(True)
        self.baseline_start_btn.setEnabled(not self.baseline_active)
        self.baseline_stop_btn.setEnabled(self.baseline_active)
        self.plotter_worker.start()

    def stop_plotter(self):
        if self.plotter_worker:
            self.plotter_worker.stop()
            self.plotter_worker.wait(1000)
        if self.measurement_metadata:
            self.measurement_metadata["Measurement_end_timestamp_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self.measurement_metadata["Received_sample_count"] = int(self.received_sample_count)
            self.measurement_metadata["Logged_row_count"] = int(self.plotter_spool.rows if self.plotter_spool else 0)
        self.plotter_rate_spin.setEnabled(True)
        self.plotter_start_btn.setEnabled(True)
        self.plotter_stop_btn.setEnabled(False)
        self.baseline_start_btn.setEnabled(True)
        self.baseline_stop_btn.setEnabled(False)

    def start_baseline_test(self):
        """Record tracking noise only; automatic particle/pressure logic is off."""
        if self.pingpong_active:
            self.stop_pingpong()
        if self.plotter_worker and self.plotter_worker.isRunning():
            self.stop_plotter()
        self.start_plotter(baseline=True)

    def stop_baseline_test(self):
        self.stop_plotter()
        self.baseline_active = False

    def _pressure_log_state(self):
        self._expire_pressure_sample()
        channels = list(DAQ_CHANNELS.values())
        a = float(self.pressure_commands.get(channels[0], 0.0))
        b = float(self.pressure_commands.get(channels[1], 0.0))
        if self.pingpong_active:
            direction = "A_to_B" if self.pingpong_forward else "B_to_A"
            state = self.pingpong_state
        elif abs(a - b) < 1e-12:
            direction, state = "None", "Idle"
        else:
            direction = "A_to_B" if a > b else "B_to_A"
            state = "Manual"
        return a, b, direction, ("Baseline" if self.baseline_active else state)

    def on_plotter_stream_started(self, requested_rate_hz, actual_rate_hz):
        self.plotter_actual_rate_hz = float(actual_rate_hz)
        self.measurement_metadata["Demod_stream_rate_Hz_actual"] = float(actual_rate_hz)
        self.measurement_metadata["Demod_stream_rate_Hz_requested"] = float(requested_rate_hz)
        self.plotter_actual_rate_label.setText(
            f"실제 적용: {actual_rate_hz:,.3f} Sa/s "
            f"(요청 {requested_rate_hz:,.1f} Sa/s)"
        )

    def on_plotter_batch(self, batch):
        """Record every streamed sample, but redraw the GUI only at 20 Hz."""
        if not batch:
            return
        self.received_sample_count += len(batch)
        locked = self._latest_pll_locked
        phase_error = self._latest_pll_error_deg
        correction = self._latest_pll_correction_hz
        pll_latency = self._latest_pll_read_latency_s
        rows = []
        for t, r, theta, f, x, y, poll_latency in batch:
            rows.append(self._record_plotter_sample(
                t, r, theta, f, x, y, locked, phase_error, correction,
                poll_latency, pll_latency,
            ))
        try:
            self.plotter_spool.append(rows)
        except OSError as exc:
            self.on_plotter_failed(f"임시 기록 파일에 쓸 수 없습니다: {exc}")
            return
        batch_times = np.fromiter((row[1] for row in rows), dtype=float)
        if self._last_raw_t is not None:
            self._recent_raw_dt_s.append(float(batch_times[0] - self._last_raw_t))
        self._recent_raw_dt_s.extend(np.diff(batch_times).tolist())
        if len(self._recent_raw_dt_s) > 10000:
            del self._recent_raw_dt_s[:-10000]
        self._last_raw_t = float(batch_times[-1])
        latest_t = batch[-1][0]
        self._trim_live_logs(latest_t)
        self._schedule_live_particle_analysis(latest_t)
        if latest_t - self._last_plotter_ui_update_t >= 0.05:
            self._last_plotter_ui_update_t = latest_t
            self._refresh_plotter_ui(latest_t)

    def _record_plotter_sample(self, t, r, theta, f, x, y, locked, phase_error,
                               correction, demod_latency, pll_latency):
        if self.pingpong_active and not self.baseline_active:
            self._pingpong_step(t, f)
        pressure_a, pressure_b, flow_direction, switch_state = self._pressure_log_state()
        switch_event = int(self._switch_event_pending)
        self._switch_event_pending = False

        display_stride = max(1, int(np.ceil(self.plotter_actual_rate_hz *
                                            self.plotter_window_spin.value() / LIVE_LOG_MAX_ROWS))) if np.isfinite(self.plotter_actual_rate_hz) else 1
        if self._display_sample_index % display_stride == 0:
            self.plot_t.append(t)
            self.plot_R.append(r)
            self.plot_theta.append(theta)
            self.plot_f.append(f)
            self.plot_x.append(x)
            self.plot_y.append(y)
        self._display_sample_index += 1

        analysis_stride = max(1, int(np.ceil(self.plotter_actual_rate_hz / 4000))) if np.isfinite(self.plotter_actual_rate_hz) else 1
        if self._analysis_sample_index % analysis_stride == 0:
            self.analysis_t.append(t)
            self.analysis_f.append(f)
        self._analysis_sample_index += 1

        pll_status_age = (
            time.monotonic() - self._latest_pll_status_monotonic
            if np.isfinite(self._latest_pll_status_monotonic) else np.nan
        )
        log_stride = max(1, int(np.ceil(self.plotter_actual_rate_hz *
                                        LIVE_LOG_WINDOW_S / LIVE_LOG_MAX_ROWS))) if np.isfinite(self.plotter_actual_rate_hz) else 1
        if self._log_sample_index % log_stride == 0:
            self.log_t.append(t)
            self.log_R.append(r)
            self.log_theta.append(theta)
            self.log_f.append(f)
            self.log_x.append(x)
            self.log_y.append(y)
            self.log_pll_target_phase.append(self.pll_setpoint_spin.value())
            self.log_pll_phase_error.append(phase_error)
            self.log_pll_frequency_command.append(f)
            self.log_pll_correction.append(correction)
            self.log_pll_locked.append(int(locked))
            self.log_pressure_a.append(pressure_a)
            self.log_pressure_b.append(pressure_b)
            self.log_flow_direction.append(flow_direction)
            self.log_switch_state.append(switch_state)
            self.log_switch_event.append(switch_event)
            self.log_measured_pressure_a.append(self.measured_pressure_a)
            self.log_measured_pressure_b.append(self.measured_pressure_b)
            self.log_temperature_c.append(self.temperature_c)
            self.log_demod_read_latency_s.append(demod_latency)
            self.log_pll_read_latency_s.append(pll_latency)
            self.log_pll_status_age_s.append(pll_status_age)
        self._log_sample_index += 1
        if not self.baseline_active and self.live_particle_thresholds is None:
            # Match the live detection stream's effective sample rate so one
            # SG window_length means the same thing for blank and events.
            if self._calibration_sample_index % analysis_stride == 0 and len(self.calibration_t) < 250000:
                self.calibration_t.append(t)
                self.calibration_f.append(f)
            self._calibration_sample_index += 1
        pa, pb = self.measured_pressure_a, self.measured_pressure_b
        dp = float(pa) - float(pb) if pa is not None and pb is not None else ""
        return [
            self.measurement_metadata.get("SessionID", ""), t, f, r, theta, x, y,
            self.pll_setpoint_spin.value(), phase_error, f, correction, int(locked),
            pll_status_age, pressure_a, pressure_b, flow_direction,
            switch_state, switch_event, pa, pb, dp, self.temperature_c,
            demod_latency, pll_latency,
        ]

    def _trim_live_logs(self, latest_t):
        # Keep enough recent raw samples for live calibration and diagnostics.
        expired = max(bisect_left(self.log_t, latest_t - LIVE_LOG_WINDOW_S),
                      len(self.log_t) - LIVE_LOG_MAX_ROWS)
        if expired > 0:
            for name in (
                "log_t", "log_R", "log_theta", "log_f", "log_x", "log_y",
                "log_pll_target_phase", "log_pll_phase_error", "log_pll_frequency_command",
                "log_pll_correction", "log_pll_locked", "log_pressure_a", "log_pressure_b",
                "log_flow_direction", "log_switch_state", "log_switch_event",
                "log_measured_pressure_a", "log_measured_pressure_b", "log_temperature_c",
                "log_demod_read_latency_s", "log_pll_read_latency_s", "log_pll_status_age_s",
            ):
                del getattr(self, name)[:expired]
        analysis_expired = max(bisect_left(self.analysis_t, latest_t - 25.0),
                               len(self.analysis_t) - 100000)
        if analysis_expired > 0:
            del self.analysis_t[:analysis_expired]
            del self.analysis_f[:analysis_expired]

    def _refresh_plotter_ui(self, t):
        refresh_started = time.perf_counter()
        window = self.plotter_window_spin.value()
        cutoff = t - window
        # At multi-kSa/s rates, repeated pop(0) shifts the full list once per
        # expired sample and can starve the GUI thread. Find the boundary once
        # and remove the expired prefix from all display buffers in bulk.
        expired = max(bisect_left(self.plot_t, cutoff), len(self.plot_t) - LIVE_LOG_MAX_ROWS)
        if expired:
            for values in (
                self.plot_t, self.plot_R, self.plot_theta,
                self.plot_f, self.plot_x, self.plot_y,
            ):
                del values[:expired]

        # Rendering hundreds of thousands of points at 20 Hz can dominate the
        # GUI thread. Keep a bounded sample of the requested display window,
        # then decimate its visual representation further.
        display_stride = max(1, len(self.plot_t) // 5000)
        self.plotter_freq_curve.setData(self.plot_t[::display_stride], self.plot_f[::display_stride])
        self.plotter_r_curve.setData(self.plot_t[::display_stride], self.plot_R[::display_stride])
        self.live_frequency_curve.setData(self.plot_t[::display_stride], self.plot_f[::display_stride])
        if (self.live_savgol_enabled.isChecked() and
                t - self._last_filtered_ui_update_t >= 0.20):
            self._last_filtered_ui_update_t = t
            # Use the detector stream (up to 4 kSa/s), rather than the more
            # aggressively decimated drawing buffer, so this has the same
            # sample-count meaning as the signal passed to detection. Keep
            # the complete requested time window so the trace scrolls
            # continuously instead of replacing a short 5,000-sample tail.
            preview_start = bisect_left(self.analysis_t, cutoff)
            preview_t = np.asarray(self.analysis_t[preview_start:], dtype=float)
            preview_f = np.asarray(self.analysis_f[preview_start:], dtype=float)
            if len(preview_t) >= self.live_savgol_window_spin.value():
                try:
                    filtered, _ = filter_frequency(
                        preview_t, preview_f, enabled=True,
                        window_length=self.live_savgol_window_spin.value(),
                        polyorder=self.live_savgol_order_spin.value())
                    preview_stride = max(1, len(preview_t) // 2500)
                    self.live_filtered_frequency_curve.setData(
                        preview_t[::preview_stride], filtered[::preview_stride])
                except ValueError:
                    self.live_filtered_frequency_curve.setData([], [])
        if self.plot_f:
            self.live_frequency_value.setText(f"{self.plot_f[-1]:,.3f} Hz")
        self._update_live_particle_markers()

        channel_map = {
            "Frequency (Hz)": self.plot_f,
            "Amplitude R (V)": self.plot_R,
            "Phase (deg)": self.plot_theta,
            "X (V)": self.plot_x,
            "Y (V)": self.plot_y,
        }
        source = channel_map[self.hist_channel_combo.currentText()]
        if len(source) >= 5:
            arr = np.asarray(source)
            std = float(np.std(arr))
            mean = float(np.mean(arr))
            counts, edges = np.histogram(arr, bins=30)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = (edges[1] - edges[0]) * 0.9 if len(edges) > 1 else 1.0
            self.plotter_hist_item.setOpts(x=centers, height=counts, width=width)
            self.plotter_hist_plot.setLabel('bottom', self.hist_channel_combo.currentText())
            self.plotter_std_label.setText(
                f"Std ({self.hist_channel_combo.currentText()}): {std:.6g}   Mean: {mean:.6g}   N={len(arr)}"
            )
        # Recent-window fitting is intentionally slower than the live Plotter.
        # Disk logging remains at the device stream rate.
        if self.baseline_active and t - self._last_baseline_ui_update_t >= 1.0:
            self._last_baseline_ui_update_t = t
            self._update_baseline_analysis()
        self.ui_refresh_durations_s.append(time.perf_counter() - refresh_started)
        if len(self.ui_refresh_durations_s) > 10000:
            del self.ui_refresh_durations_s[:5000]

    def _update_baseline_analysis(self):
        n = len(self.log_t)
        if n == 0:
            return
        t = np.asarray(self.log_t, dtype=float)
        frequency = np.asarray(self.log_f, dtype=float)
        amplitude = np.asarray(self.log_R, dtype=float)
        phase_error = np.asarray(self.log_pll_phase_error, dtype=float)
        baseline = linear_baseline(t, frequency)
        if not baseline.get("valid"):
            return
        trend = baseline["trend_hz"]
        residual = baseline["residual_hz"]
        raw_std = baseline["raw_std_hz"]
        detrended_std = baseline["detrended_std_hz"]
        nominal_interval_s = (
            1.0 / self.plotter_actual_rate_hz
            if np.isfinite(self.plotter_actual_rate_hz) and self.plotter_actual_rate_hz > 0
            else 1.0 / self.plotter_rate_spin.value()
        )
        timing = timing_quality(t, nominal_interval_s)
        display_stride = max(1, n // 5000)
        show = slice(None, None, display_stride)
        self.baseline_raw_curve.setData(t[show], frequency[show])
        self.baseline_trend_curve.setData(t[show], trend[show])
        self.baseline_residual_curve.setData(t[show], residual[show])
        self.baseline_amplitude_curve.setData(t[show], amplitude[show])
        self.baseline_error_curve.setData(t[show], phase_error[show])
        self.baseline_stats_label.setText(
            f"Baseline (N={n}): mean={np.mean(frequency):.8g} Hz | raw σ={raw_std:.5g} Hz | "
            f"p-p={baseline['peak_to_peak_hz']:.5g} Hz | drift={baseline['drift_hz_per_min']:.5g} Hz/min | "
            f"detrended σ={detrended_std:.5g} Hz | 3σ={3*detrended_std:.5g} Hz | "
            f"5σ={5*detrended_std:.5g} Hz | effective rate={timing.get('effective_rate_hz', np.nan):.4g} Hz"
        )

    def run_baseline_analysis(self):
        if len(self.log_t) < 2:
            QMessageBox.information(self, "Baseline 분석", "먼저 Plotter/Baseline 데이터를 2개 이상 기록하세요.")
            return
        nominal_interval_s = (
            1.0 / self.plotter_actual_rate_hz
            if np.isfinite(self.plotter_actual_rate_hz) and self.plotter_actual_rate_hz > 0
            else 1.0 / self.plotter_rate_spin.value()
        )
        self._run_baseline_analysis_source({
            "name": "recent bounded Plotter analysis window", "metadata": dict(self.measurement_metadata),
            "time_s": np.asarray(self.log_t, float), "frequency_hz": np.asarray(self.log_f, float),
            "amplitude_v": np.asarray(self.log_R, float), "phase_deg": np.asarray(self.log_theta, float),
            "phase_error_deg": np.asarray(self.log_pll_phase_error, float),
            "demod_latency_s": np.asarray(self.log_demod_read_latency_s, float),
            "pll_latency_s": np.asarray(self.log_pll_read_latency_s, float),
            "nominal_interval_s": nominal_interval_s,
        })

    def load_baseline_csv_for_analysis(self):
        path, _ = QFileDialog.getOpenFileName(self, "Plotter CSV 불러오기 (분석 전용)", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.baseline_source_role = "particle"
            self._run_baseline_analysis_source(self._load_analysis_source(path))
        except Exception as exc:
            QMessageBox.warning(self, "CSV 불러오기 실패", str(exc))

    def _load_analysis_source(self, path):
        loaded = load_plotter_csv(path)
        metadata = loaded["metadata"]
        try:
            requested_ms = float(metadata.get("Plotter_requested_interval_ms", "nan"))
        except ValueError:
            requested_ms = np.nan
        nominal = requested_ms / 1000.0 if np.isfinite(requested_ms) and requested_ms > 0 else np.nan
        if not np.isfinite(nominal):
            nominal = timing_quality(loaded["time_s"]).get("dt_median_s", np.nan)
        return {
            "name": os.path.basename(path), "path": os.path.abspath(path), "metadata": metadata,
            "time_s": loaded["time_s"], "frequency_hz": loaded["frequency_hz"],
            "amplitude_v": loaded["amplitude_v"], "phase_deg": loaded["phase_deg"],
            "phase_error_deg": loaded.get("phase_error_deg", loaded["phase_deg"]),
            "demod_latency_s": np.asarray([], float), "pll_latency_s": np.asarray([], float),
            "nominal_interval_s": nominal,
        }

    def select_calibration_blank(self):
        path, _ = QFileDialog.getOpenFileName(self, "Threshold calibration blank 선택", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.threshold_calibration_source = self._load_analysis_source(path)
            self.calibration_blank_btn.setText("Calibration: " + os.path.basename(path))
            self.baseline_stats_label.setText(
                "Calibration blank 선택됨: " + os.path.basename(path) +
                "\n문턱은 이 파일에서 확정되며 evaluation/particle 분석 중 변경되지 않습니다."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Calibration blank 오류", str(exc))

    def load_evaluation_blank(self):
        if self.threshold_calibration_source is None and self.event_threshold_spin.value() <= 0:
            QMessageBox.warning(self, "Calibration 필요", "먼저 독립 Threshold calibration blank를 선택하세요.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Evaluation blank 선택", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            absolute = os.path.abspath(path)
            calibration_path = (self.threshold_calibration_source or {}).get("path", "")
            if calibration_path and os.path.normcase(absolute) == os.path.normcase(calibration_path):
                raise ValueError("Calibration blank와 Evaluation blank는 서로 다른 파일이어야 합니다.")
            self.baseline_source_role = "evaluation_blank"
            self._run_baseline_analysis_source(self._load_analysis_source(path))
        except Exception as exc:
            QMessageBox.warning(self, "Evaluation blank 오류", str(exc))

    def _run_baseline_analysis_source(self, source):
        if self.baseline_analysis_worker and self.baseline_analysis_worker.isRunning():
            return
        self.baseline_analysis_source = source
        self.baseline_analysis_btn.setEnabled(False)
        self.baseline_load_btn.setEnabled(False)
        self.baseline_analysis_btn.setText("Baseline 상세 분석 중...")
        self.baseline_analysis_worker = BaselineAnalysisWorker(
            source["time_s"], source["frequency_hz"], source["amplitude_v"], source["phase_deg"], source["nominal_interval_s"],
            self.event_duration_spin.value(), self.event_baseline_spin.value(), self.event_guard_spin.value(),
            self.event_threshold_spin.value() if self.event_threshold_spin.value() > 0 else None,
            source.get("demod_latency_s"), source.get("pll_latency_s"),
            (self.baseline_rolling_window_spin.value(),) if self.baseline_rolling_window_spin.value() > 0 else (),
            self.threshold_calibration_source, self.baseline_source_role,
            self.detection_method_combo.currentText(), self.metadata_distance_spin.value(),
        )
        self.baseline_analysis_worker.completed.connect(self.on_baseline_analysis_ready)
        self.baseline_analysis_worker.failed.connect(self.on_baseline_analysis_failed)
        self.baseline_analysis_worker.start()

    def on_baseline_analysis_ready(self, result):
        self.baseline_analysis_result = result
        self.baseline_analysis_btn.setEnabled(True)
        self.baseline_load_btn.setEnabled(True)
        self.baseline_analysis_btn.setText("Baseline 상세 분석 실행 (현재 raw 기록)")
        self.baseline_summary_export_btn.setEnabled(True)
        self.baseline_derived_export_btn.setEnabled(True)
        self.baseline_allan_export_btn.setEnabled(bool(result["allan"]))
        self.event_export_btn.setEnabled(True)
        timing = result["timing"]
        candidates = result["particle_candidates"].get("candidates", [])
        global_result = result["global"]
        if global_result.get("valid"):
            self.baseline_raw_curve.setData(global_result["time_s"], global_result["trend_hz"] + global_result["residual_hz"])
            self.baseline_trend_curve.setData(global_result["time_s"], global_result["trend_hz"])
            self.baseline_residual_curve.setData(global_result["time_s"], global_result["residual_hz"])
        source = self.baseline_analysis_source or {}
        self.baseline_amplitude_curve.setData(source.get("time_s", []), source.get("amplitude_v", []))
        # Imported CSV only supplies phase; its phase-error column is not required for analysis.
        self.baseline_error_curve.setData(source.get("time_s", []), source.get("phase_error_deg", source.get("phase_deg", [])))
        rolling_key = str(self.baseline_rolling_window_spin.value())
        rolling = result["rolling"].get(rolling_key) or result["rolling"].get("60.0")
        if rolling:
            self.baseline_rolling_curve.setData(rolling["time_s"], rolling["metrics"]["detrended_std_hz"])
        allan = result["allan"]
        if allan:
            tau = np.asarray([row["tau_s"] for row in allan], float); adev = np.asarray([row["allan_deviation_hz"] for row in allan], float)
            valid = (tau > 0) & (adev > 0) & np.isfinite(tau) & np.isfinite(adev)
            self.baseline_allan_curve.setData(tau[valid], adev[valid])
        adequacy = result["event_sampling_adequacy"]
        latency = result["read_latency"]["demod_read"]
        calibration = result.get("threshold_calibration")
        calibration_text = (f"calibration={calibration.get('source_name', 'unknown')}"
                            if calibration else "calibration 미지정: 자동 검출 비활성")
        evaluation = result.get("blank_evaluation")
        if evaluation:
            rate = evaluation["false_event_rate"]
            detection_text = (
                f"evaluation false events={rate['event_count']}건, "
                f"{rate['false_events_per_h']:.4g}건/h, "
                f"Poisson 95% 상한={rate['poisson_upper_events_per_h']:.4g}건/h"
            )
        else:
            excluded = len(result["particle_candidates"].get("excluded_transient", []))
            detection_text = f"자동 후보={len(candidates)}개, transient 분리={excluded}개"
        self.baseline_stats_label.setText(
            f"분석 source: {source.get('name', 'unknown')} ({result.get('source_role', 'particle')})\n"
            f"median dt={timing.get('dt_median_s', np.nan):.5g}s, p99={timing.get('dt_p99_s', np.nan):.5g}s, "
            f"{calibration_text}, {detection_text}. "
            f"이벤트 표본={adequacy['median_samples_per_event']:.2g} ({adequacy['assessment']}). "
            f"HF2 demod read median/p99={latency.get('median_s', np.nan)*1e3:.3g}/{latency.get('p99_s', np.nan)*1e3:.3g} ms.\n"
            "raw / derived / events는 동일 SessionID의 별도 파일로 저장됩니다."
        )

    def on_baseline_analysis_failed(self, message):
        self.baseline_analysis_btn.setEnabled(True)
        self.baseline_load_btn.setEnabled(True)
        self.baseline_analysis_btn.setText("Baseline 상세 분석 실행 (현재 raw 기록)")
        QMessageBox.warning(self, "Baseline 분석 실패", message)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict): return {str(k): MainWindow._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [MainWindow._json_safe(v) for v in value]
        if isinstance(value, np.ndarray): return [MainWindow._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic): return value.item()
        return value

    def export_baseline_summary(self):
        if not self.baseline_analysis_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Baseline summary 저장", "baseline_summary.json", "JSON Files (*.json)")
        if not path: return
        source = self.baseline_analysis_source or {}
        payload = {"source": source.get("name", "current in-memory raw Plotter log"), "source_path": source.get("path", ""), "raw_data_modified": False,
                   "analysis_timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "measurement_metadata": source.get("metadata", self.measurement_metadata), "analysis": self.baseline_analysis_result}
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=True)
            QMessageBox.information(self, "저장 완료", f"Summary 저장됨: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def export_baseline_allan(self):
        result = self.baseline_analysis_result
        if not result or not result.get("allan"):
            QMessageBox.information(self, "Allan 저장", "먼저 충분한 baseline 데이터를 분석하세요.")
            return
        source = self.baseline_analysis_source or {}
        session_id = source.get("metadata", self.measurement_metadata).get("SessionID", "unknown")
        path, _ = QFileDialog.getSaveFileName(
            self, "Allan deviation 그래프·수치 저장", f"allan_{session_id}.csv", "CSV Files (*.csv)")
        if not path:
            return
        sensitivity = self.allan_sensitivity_spin.value() or None
        try:
            csv_path, png_path = save_allan_export(path, result["allan"], sensitivity)
            unit = "µg (입력한 질량 감도 적용)" if sensitivity is not None else "Hz"
            QMessageBox.information(self, "저장 완료", f"Allan y축: {unit}\nCSV: {csv_path}\nPNG: {png_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Allan 저장 오류", str(exc))

    def export_baseline_derived_csv(self):
        result = self.baseline_analysis_result
        if not result or not result["global"].get("valid"):
            return
        source = self.baseline_analysis_source or {}
        session_id = source.get("metadata", self.measurement_metadata).get("SessionID", "unknown")
        path, _ = QFileDialog.getSaveFileName(self, "Derived time-series 저장", f"derived_{session_id}.csv", "CSV Files (*.csv)")
        if not path: return
        global_result=result["global"]
        rolling_key = str(self.baseline_rolling_window_spin.value())
        rolling=result["rolling"].get(rolling_key) or result["rolling"].get("60.0")
        rolling_window = rolling["window_s"] if rolling else np.nan
        transient=result["transients"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer=csv.writer(handle)
                writer.writerow(["# Metadata", "SessionID", session_id])
                writer.writerow(["# Derived", "Raw_data_modified", "False"])
                writer.writerow(["# Derived", "Processing", "linear detrend uses actual timestamps; rolling is trailing/causal; no interpolation/resampling"])
                suffix = f"_{rolling_window:g}s" if np.isfinite(rolling_window) else ""
                writer.writerow(["SessionID", "Time_s", "Frequency_Trend_Hz", "Frequency_Detrended_Hz", f"RollingMean{suffix}_Hz", f"RollingStd{suffix}_Hz", f"RollingDetrendedStd{suffix}_Hz", f"LocalDrift{suffix}_HzPerMin", "TransientFlag", "TransientType", "TransientMagnitude"])
                metrics=rolling["metrics"] if rolling else {}
                for i, time_s in enumerate(global_result["time_s"]):
                    writer.writerow([session_id, time_s, global_result["trend_hz"][i], global_result["residual_hz"][i],
                                     metrics.get("mean_hz", [np.nan]*len(global_result["time_s"]))[i],
                                     metrics.get("std_hz", [np.nan]*len(global_result["time_s"]))[i],
                                     metrics.get("detrended_std_hz", [np.nan]*len(global_result["time_s"]))[i],
                                     metrics.get("drift_hz_per_min", [np.nan]*len(global_result["time_s"]))[i],
                                     int(transient["flag"][i]), transient["type"][i], transient["magnitude"][i]])
            QMessageBox.information(self, "저장 완료", f"Derived CSV 저장됨: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def export_event_analysis_csv(self):
        if not self.baseline_analysis_result:
            return
        source = self.baseline_analysis_source or {}
        session_id = source.get("metadata", self.measurement_metadata).get("SessionID", "unknown")
        path, _ = QFileDialog.getSaveFileName(self, "Event analysis 저장", f"events_{session_id}.csv", "CSV Files (*.csv)")
        if not path: return
        result=self.baseline_analysis_result
        try:
            event_rows=result["particle_candidates"].get("all_candidates", [])
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer=csv.writer(handle)
                writer.writerow(["# Metadata", "SessionID", session_id])
                writer.writerow(["# Analysis", "Processing", result["particle_candidates"]["processing"]])
                fields=["SessionID","EventID","EventStart_s","PeakTime_s","EventEnd_s","WindowDuration_s","ActualTransitStart_s","ActualTransitEnd_s","ActualTransitTime_s","Velocity_mm_s","EventSampleCount","BaselineSampleCount","LocalBaselineFrequency_Hz","LocalBaselineDetrendedStd_Hz","PeakDepth_Hz","IntegratedDip_Hz_s","TemplateAmplitude_Hz","TemplateCorrelation","PeakSNR","IntegralSNR","AmplitudeChange_V","PhaseChange_deg","DetectionMethod","DetectionScore","ScoreThreshold","PeakThreshold_Hz","NormalizedScore","MatchedDurationCount","Detected","Decision","TransientOverlap","TimestampGapFlag"]
                writer.writerow(fields)
                for row in event_rows:
                    writer.writerow([session_id if key=="SessionID" else row.get(key,"") for key in fields])
            threshold_path=path[:-4]+"_thresholds.csv" if path.lower().endswith(".csv") else path+"_thresholds.csv"
            thresholds = (result.get("threshold_calibration") or {}).get("thresholds", [])
            with open(threshold_path,"w",newline="",encoding="utf-8-sig") as handle:
                writer=csv.writer(handle)
                writer.writerow(["# Metadata", "SessionID", session_id])
                threshold_fields=["Duration_s","PeakThreshold_Hz","IntegralThreshold_Hz_s","TemplateThreshold_Hz","CalibrationWindowCount","FamilywiseFalseAlarm","PerDurationAlpha","Percentile","ThresholdMethod"]
                writer.writerow(threshold_fields)
                for row in thresholds: writer.writerow([row.get(key,"") for key in threshold_fields])
            evaluation = result.get("blank_evaluation")
            evaluation_text = ""
            if evaluation:
                rate = evaluation["false_event_rate"]
                evaluation_path=path[:-4]+"_evaluation_summary.csv"
                with open(evaluation_path,"w",newline="",encoding="utf-8-sig") as handle:
                    writer=csv.writer(handle); writer.writerow(rate.keys()); writer.writerow(rate.values())
                evaluation_text = f"\nEvaluation summary: {evaluation_path}"
            QMessageBox.information(self,"저장 완료",f"Event CSV: {path}\nThreshold CSV: {threshold_path}{evaluation_text}")
        except Exception as exc:
            QMessageBox.critical(self,"저장 오류",str(exc))

    def on_plotter_failed(self, msg):
        self.stop_plotter()
        QMessageBox.critical(self, "Plotter 오류", msg)

    def export_plotter_csv(self):
        if self.plotter_spool is None or self.plotter_spool.rows == 0:
            QMessageBox.information(self, "저장할 데이터 없음", "먼저 Plotter를 실행하세요.")
            return
        session_id = self.measurement_metadata.get("SessionID", time.strftime('%Y%m%d_%H%M%S'))
        default_name = f"raw_{session_id}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Plotter 데이터 저장", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            self._write_raw_csv(path)
            QMessageBox.information(self, "저장 완료", f"저장됨: {path}\n({self.plotter_spool.rows} 포인트)")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    def _write_raw_csv(self, path):
        metadata = dict(self.measurement_metadata)
        metadata["Received_sample_count"] = int(self.received_sample_count)
        metadata["Logged_row_count"] = int(self.plotter_spool.rows if self.plotter_spool else 0)
        if self._recent_raw_dt_s:
            elapsed = np.asarray(self._recent_raw_dt_s, dtype=float)
            elapsed = elapsed[elapsed > 0]
            if len(elapsed):
                metadata["Actual_sampling_rate_Hz"] = 1.0 / float(np.median(elapsed))
        if self.plotter_spool is None:
            raise OSError("Plotter 임시 기록 파일이 없습니다.")
        self.plotter_spool.export(path, metadata)

    def compare_baseline_files(self):
        baseline_paths, _ = QFileDialog.getOpenFileNames(
            self, "비교할 baseline raw CSV 선택", "", "CSV Files (*.csv)")
        if not baseline_paths:
            return
        sweep_paths, _ = QFileDialog.getOpenFileNames(
            self, "연결할 sweep/peak CSV 선택 (선택 사항)", "", "CSV Files (*.csv)")
        output, _ = QFileDialog.getSaveFileName(
            self, "Baseline 비교 결과 저장", "baseline_comparison.csv", "CSV Files (*.csv)")
        if not output:
            return
        try:
            result = compare_baselines(baseline_paths, sweep_paths)
            csv_path, json_path = write_comparison(result, output)
            warning = "\n" + "\n".join(result["warnings"]) if result["warnings"] else "\n실험조건 metadata가 일치합니다."
            QMessageBox.information(
                self, "Baseline 비교 완료",
                f"CSV: {csv_path}\nJSON: {json_path}{warning}")
        except Exception as exc:
            QMessageBox.critical(self, "Baseline 비교 오류", str(exc))

    def _integration_result(self):
        pll_rate = self.measurement_metadata.get("PLL_update_rate_Hz", np.nan)
        try:
            expected_pll_update_s = 1.0 / float(pll_rate) if float(pll_rate) > 0 else 0.25
        except (TypeError, ValueError):
            expected_pll_update_s = 0.25
        result = integration_diagnostics(
            self.log_t, len(self.log_t), self.log_pll_status_age_s,
            self.log_demod_read_latency_s, self.log_pressure_a, self.log_pressure_b,
            self.log_measured_pressure_a, self.log_measured_pressure_b,
            expected_pll_update_s, self.ui_refresh_durations_s, 0.05,
        )
        spooled = self.plotter_spool.rows if self.plotter_spool else 0
        result["raw_recording"] = {
            "received_samples": self.received_sample_count,
            "temporary_file_rows": spooled,
            "pass": self.received_sample_count == spooled,
        }
        result["analysis_window"] = "Recent sampled window; full acquisition is in the temporary CSV."
        result["checks"]["overall_pass"] &= result["raw_recording"]["pass"]
        return result

    def export_integration_report(self):
        if not self.log_t:
            QMessageBox.information(self, "통합시험", "먼저 실제 장비 기록을 수집하세요.")
            return
        session_id = self.measurement_metadata.get("SessionID", "unknown")
        path, _ = QFileDialog.getSaveFileName(
            self, "통합시험 리포트 저장", f"integration_{session_id}.json", "JSON Files (*.json)")
        if not path:
            return
        result = {
            "SessionID": session_id,
            "measurement_metadata": self.measurement_metadata,
            "integration": self._integration_result(),
            "limitations": "장시간 UI 지연은 demod read latency와 PLL status age로 진단하며 실제 장비 실행 중 검증해야 합니다.",
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._json_safe(result), handle, ensure_ascii=False, indent=2, allow_nan=True)
            overall = result["integration"]["checks"]["overall_pass"]
            QMessageBox.information(self, "통합시험 리포트", f"저장됨: {path}\n전체 판정: {'PASS' if overall else 'CHECK'}")
        except Exception as exc:
            QMessageBox.critical(self, "통합시험 저장 오류", str(exc))

    def export_session_bundle(self):
        if not self.log_t:
            QMessageBox.information(self, "Session bundle", "먼저 세션 데이터를 수집하세요.")
            return
        if not self.baseline_analysis_result:
            QMessageBox.information(self, "Session bundle", "먼저 Baseline 상세 분석을 실행하세요.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Session bundle 저장 폴더 선택")
        if not directory:
            return
        session_id = self.measurement_metadata.get("SessionID", "unknown")
        raw_path = os.path.join(directory, f"raw_{session_id}.csv")
        derived_path = os.path.join(directory, f"derived_{session_id}.csv")
        events_path = os.path.join(directory, f"events_{session_id}.csv")
        summary_path = os.path.join(directory, f"summary_{session_id}.json")
        thresholds_path = os.path.join(directory, f"thresholds_{session_id}.csv")
        try:
            self._write_raw_csv(raw_path)
            result = self.baseline_analysis_result
            global_result = result["global"]
            rolling = result["rolling"].get(str(self.baseline_rolling_window_spin.value())) or result["rolling"].get("60.0")
            metrics = rolling["metrics"] if rolling else {}
            transient = result["transients"]
            with open(derived_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# Metadata", "SessionID", session_id])
                writer.writerow(["SessionID", "Time_s", "Frequency_Trend_Hz", "Frequency_Detrended_Hz", "RollingDetrendedStd_Hz", "TransientFlag", "TransientType", "TransientMagnitude"])
                for i, timestamp in enumerate(global_result["time_s"]):
                    rolling_std = metrics.get("detrended_std_hz", [np.nan] * len(global_result["time_s"]))[i]
                    writer.writerow([session_id, timestamp, global_result["trend_hz"][i], global_result["residual_hz"][i], rolling_std, int(transient["flag"][i]), transient["type"][i], transient["magnitude"][i]])
            event_fields = ["SessionID", "EventID", "EventStart_s", "PeakTime_s", "EventEnd_s", "WindowDuration_s", "ActualTransitTime_s", "Velocity_mm_s", "PeakDepth_Hz", "IntegratedDip_Hz_s", "TemplateAmplitude_Hz", "PeakSNR", "IntegralSNR", "AmplitudeChange_V", "PhaseChange_deg", "Decision", "TransientOverlap", "TimestampGapFlag"]
            with open(events_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle); writer.writerow(event_fields)
                for row in result["particle_candidates"].get("all_candidates", []):
                    writer.writerow([session_id if key == "SessionID" else row.get(key, "") for key in event_fields])
            threshold_fields = ["Duration_s", "PeakThreshold_Hz", "IntegralThreshold_Hz_s", "TemplateThreshold_Hz", "CalibrationWindowCount", "FamilywiseFalseAlarm", "PerDurationAlpha", "Percentile", "ThresholdMethod"]
            with open(thresholds_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle); writer.writerow(["# Metadata", "SessionID", session_id]); writer.writerow(threshold_fields)
                for row in (result.get("threshold_calibration") or {}).get("thresholds", []):
                    writer.writerow([row.get(key, "") for key in threshold_fields])
            payload = {
                "SessionID": session_id, "measurement_metadata": self.measurement_metadata,
                "analysis": result, "integration": self._integration_result(),
                "files": {"raw": raw_path, "derived": derived_path, "events": events_path, "thresholds": thresholds_path},
            }
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(self._json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=True)
            QMessageBox.information(self, "Session bundle", f"SessionID: {session_id}\nraw/derived/events/thresholds/summary 저장 완료")
        except Exception as exc:
            QMessageBox.critical(self, "Session bundle 저장 오류", str(exc))

    # --- DAQ ---
    def set_daq_voltage(self, channel, voltage):
        voltage = max(MIN_DAQ_V, min(MAX_DAQ_V, float(voltage)))
        # Record the requested output even if the driver is temporarily
        # unavailable; this is a command log, not a fabricated pressure readback.
        self.pressure_commands[channel] = voltage
        if nidaqmx is None:
            return
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(channel, min_val=MIN_DAQ_V, max_val=MAX_DAQ_V)
                task.write(voltage, auto_start=True)
        except Exception as e:
            QMessageBox.critical(self, "DAQ 오류", f"{channel}: {e}")

    def zero_daq_channels(self):
        for ch in DAQ_CHANNELS.values():
            self.set_daq_voltage(ch, 0.0)
            if ch in self.daq_spins:
                self.daq_spins[ch].setValue(0.0)

    # --- Ping-Pong particle repeat-measurement ---
    # Driven off the Plotter's own live Frequency stream
    # (_record_plotter_sample calls _pingpong_step for every buffered sample).
    # State machine:
    #   baseline -> transit (dip detected) -> settling (dwell) -> baseline
    # On each baseline->settling->baseline cycle the DAQ pressure direction
    # is reversed, sending the same particle back through the tube -- the
    # same "detect at tube end, reverse pressure" method as Katsikis et al.
    def start_pingpong(self):
        if not self.hf:
            return
        if self.baseline_active:
            QMessageBox.warning(
                self, "Baseline 모드",
                "Baseline 테스트 중에는 입자 감지와 자동 공압 전환을 실행할 수 없습니다. 먼저 Baseline 테스트를 중지하세요.",
            )
            return
        if self.plotter_worker is None or not self.plotter_worker.isRunning():
            self.start_plotter()
        if not self.pll_enable_btn.isChecked():
            QMessageBox.warning(
                self, "PLL 확인",
                "PLL이 잠겨있지 않으면 Frequency가 고정되어 딥을 감지할 수 없습니다.\n"
                "PLL을 먼저 활성화하는 것을 권장합니다. 일단 계속 진행합니다."
            )

        self.pingpong_state = "baseline"
        self.pingpong_baseline_f = None
        self.pingpong_transit_start_t = None
        self.pingpong_transit_min_f = None
        self.pingpong_forward = True
        self.pingpong_cycle_count = 0
        self.pingpong_settle_until_t = None
        self.pingpong_events = []
        self.pp_event_list.clear()
        self.pingpong_markers.setData([], [])

        self._pingpong_apply_direction(forward=True)
        self.pingpong_active = True
        self.pp_start_btn.setEnabled(False)
        self.pp_stop_btn.setEnabled(True)
        self.pp_status_label.setText("상태: 기준선 수집 중...")
        self.tabs.setCurrentIndex(3 if self.live_particle_enabled.isChecked() else 2)

    def stop_pingpong(self):
        self.pingpong_active = False
        self.pp_start_btn.setEnabled(True)
        self.pp_stop_btn.setEnabled(False)
        self.pp_status_label.setText(f"상태: 정지됨 (총 {self.pingpong_cycle_count}회 반복)")
        self.zero_daq_channels()

    def _pingpong_apply_direction(self, forward):
        base = self.pp_base_v_spin.value()
        delta = self.pp_delta_v_spin.value()
        ch0, ch1 = list(DAQ_CHANNELS.values())  # Dev3/ao0, Dev3/ao1
        v0, v1 = (base + delta, base - delta) if forward else (base - delta, base + delta)
        self.set_daq_voltage(ch0, v0)
        self.set_daq_voltage(ch1, v1)
        if ch0 in self.daq_spins:
            self.daq_spins[ch0].setValue(max(MIN_DAQ_V, min(MAX_DAQ_V, v0)))
        if ch1 in self.daq_spins:
            self.daq_spins[ch1].setValue(max(MIN_DAQ_V, min(MAX_DAQ_V, v1)))
        self.pingpong_forward = forward
        self._last_flow_direction = "A_to_B" if forward else "B_to_A"
        self._switch_event_pending = True

    def _pingpong_step(self, t, f):
        threshold = self.pp_threshold_spin.value()
        settle_time = self.pp_settle_spin.value()
        max_cycles = self.pp_max_cycles_spin.value()

        if self.pingpong_state == "baseline":
            if self.pingpong_baseline_f is None:
                self.pingpong_baseline_f = f
            else:
                # Slow-moving average so a fast dip can't drag the baseline
                # down with it (baseline is held frozen during transit/settling).
                self.pingpong_baseline_f = 0.98 * self.pingpong_baseline_f + 0.02 * f

            if self.pingpong_baseline_f - f > threshold:
                self.pingpong_state = "transit"
                self.pingpong_transit_start_t = t
                self.pingpong_transit_min_f = f
                self.pp_status_label.setText(f"상태: 입자 통과 감지됨 (사이클 {self.pingpong_cycle_count})")

        elif self.pingpong_state == "transit":
            self.pingpong_transit_min_f = min(self.pingpong_transit_min_f, f)
            if self.pingpong_baseline_f - f < threshold * 0.5:
                duration = t - self.pingpong_transit_start_t
                depth = self.pingpong_baseline_f - self.pingpong_transit_min_f
                event_t = (self.pingpong_transit_start_t + t) / 2.0
                self.pingpong_events.append((event_t, self.pingpong_transit_min_f, depth, duration))
                self._add_pingpong_event(event_t, self.pingpong_transit_min_f, depth, duration)
                self.pingpong_state = "settling"
                self.pingpong_settle_until_t = t + settle_time
                self.pp_status_label.setText(
                    f"상태: 통과 완료 (깊이 {depth:.3g} Hz, {duration:.2f}s) — 반전 대기 중"
                )

        elif self.pingpong_state == "settling":
            if t >= self.pingpong_settle_until_t:
                self._pingpong_apply_direction(forward=not self.pingpong_forward)
                self.pingpong_cycle_count += 1
                self.pingpong_state = "baseline"
                direction_txt = "→" if self.pingpong_forward else "←"
                self.pp_status_label.setText(f"상태: 반전 완료 ({direction_txt}), 총 {self.pingpong_cycle_count}회")
                if max_cycles > 0 and self.pingpong_cycle_count >= max_cycles:
                    self.stop_pingpong()
                    self.pp_status_label.setText(f"상태: 목표 반복 횟수({max_cycles}) 도달, 정지됨")

    def _add_pingpong_event(self, t, min_f, depth, duration):
        n = len(self.pingpong_events)
        item = QListWidgetItem(f"#{n}  t={t:.2f}s  깊이={depth:.4g}Hz  통과시간={duration:.2f}s")
        self.pp_event_list.addItem(item)
        self.pp_event_list.scrollToBottom()
        xs = [e[0] for e in self.pingpong_events]
        ys = [e[1] for e in self.pingpong_events]
        self.pingpong_markers.setData(xs, ys)

    def export_pingpong_csv(self):
        if not self.pingpong_events:
            QMessageBox.information(self, "저장할 데이터 없음", "먼저 Ping-Pong을 실행하세요.")
            return
        default_name = f"pingpong_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Ping-Pong 이벤트 저장", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Event_index", "Time_s", "Min_Frequency_Hz", "Depth_Hz", "Duration_s"])
                for i, (t, min_f, depth, dur) in enumerate(self.pingpong_events, 1):
                    writer.writerow([i, t, min_f, depth, dur])
            QMessageBox.information(self, "저장 완료", f"저장됨: {path}\n({len(self.pingpong_events)} 이벤트)")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    # --- emergency stop ---
    def emergency_stop(self):
        self.pingpong_active = False
        self.pp_start_btn.setEnabled(True)
        self.pp_stop_btn.setEnabled(False)
        if self.sweep_worker:
            self.sweep_worker.stop()
        if self.pll_status_worker:
            self.pll_status_worker.stop()
            self.pll_status_worker.wait(1000)
        if self.plotter_worker:
            self.plotter_worker.stop()
            self.plotter_worker.wait(1000)
        if self.live_particle_worker and self.live_particle_worker.isRunning():
            self.live_particle_worker.wait(2000)
        self.pll_enable_btn.setChecked(False)
        self.output_on_btn.setChecked(False)
        self.amp_enable_btn.setChecked(False)
        self.plotter_start_btn.setEnabled(True)
        self.plotter_stop_btn.setEnabled(False)
        if self.hf:
            try:
                self.hf.set_output_on(False)
                self.hf.set_amp_enabled(False)
                self.hf.set_pll_enabled(False)
            except Exception:
                pass
        self.zero_daq_channels()
        self.status_label.setText(self.status_label.text() + "  [긴급 정지 실행됨]")

    # --- cleanup ---
    def closeEvent(self, event):
        if self.pressure_worker is not None:
            self.pressure_worker.requestInterruption()
            if not self.pressure_worker.wait(1000):
                self.pressure_readback_label.setText("DAQ 응답 대기 중: 압력 읽기 종료 후 다시 닫아주세요.")
                event.ignore()
                return
        self.pressure_watchdog.stop()
        try:
            self._save_last_settings()
        except OSError as exc:
            print(f"Last settings could not be saved: {exc}", file=sys.stderr)
        try:
            self.emergency_stop()
        except Exception:
            pass
        if self.plotter_spool is not None:
            self.plotter_spool.discard()
            self.plotter_spool = None
        if self.hf:
            self.hf.close()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Auto-detect hardware BEFORE building the UI, so DAQ_CHANNELS (used
    # while constructing the DAQ panel) already reflects this PC's actual
    # device alias rather than whatever PC the code was written on.
    global HF2_DEVICE, DAQ_DEVICE, DAQ_CHANNELS

    detected_hf2 = discover_hf2_serial()
    if detected_hf2:
        HF2_DEVICE = detected_hf2

    if AUTO_DETECT_DAQ:
        detected_daq, daq_candidates = discover_daq_device()
        if detected_daq:
            DAQ_DEVICE = detected_daq
            DAQ_CHANNELS = _build_daq_channels(DAQ_DEVICE)
            if len(daq_candidates) > 1:
                QMessageBox.information(
                    None, "NI-DAQ 장치 여러 개 발견",
                    f"AO 채널이 있는 NI-DAQ 장치가 {len(daq_candidates)}개 발견되어 "
                    f"첫 번째({DAQ_DEVICE})를 사용합니다.\n발견된 장치: {', '.join(daq_candidates)}\n\n"
                    f"다른 장치를 쓰려면 코드 상단의 DAQ_DEVICE를 직접 지정하세요."
                )

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
