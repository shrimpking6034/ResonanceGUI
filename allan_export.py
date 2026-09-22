"""Export baseline Allan deviation as a standalone graph and numeric table."""

import csv
from pathlib import Path

import numpy as np


def allan_export_rows(allan, sensitivity_pg_hz=None):
    """Keep the analysis values in Hz; convert to µg only with a valid calibration."""
    if sensitivity_pg_hz is not None:
        sensitivity_pg_hz = float(sensitivity_pg_hz)
        if not np.isfinite(sensitivity_pg_hz) or sensitivity_pg_hz <= 0:
            raise ValueError("질량 감도는 0보다 큰 유한한 pg/Hz 값이어야 합니다.")
    rows = []
    for point in allan:
        tau = float(point["tau_s"])
        adev_hz = float(point["allan_deviation_hz"])
        if not np.isfinite(tau) or not np.isfinite(adev_hz) or tau <= 0 or adev_hz < 0:
            continue
        rows.append({
            "tau_s": tau,
            "allan_deviation_hz": adev_hz,
            "mass_equivalent_noise_ug": (adev_hz * sensitivity_pg_hz / 1e6
                                         if sensitivity_pg_hz is not None else None),
            "cluster_count": int(point["cluster_count"]),
            "method": point["method"],
        })
    if not rows:
        raise ValueError("저장할 유효한 Allan deviation 값이 없습니다.")
    return rows


def save_allan_export(csv_path, allan, sensitivity_pg_hz=None):
    """Write a CSV and a sibling PNG for the same Allan analysis result."""
    rows = allan_export_rows(allan, sensitivity_pg_hz)
    csv_path = Path(csv_path)
    png_path = csv_path.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tau_s", "allan_deviation_hz", "mass_equivalent_noise_ug",
                         "sensitivity_pg_hz", "cluster_count", "method"])
        for row in rows:
            writer.writerow([row["tau_s"], row["allan_deviation_hz"],
                             row["mass_equivalent_noise_ug"] if sensitivity_pg_hz is not None else "",
                             sensitivity_pg_hz if sensitivity_pg_hz is not None else "",
                             row["cluster_count"], row["method"]])

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(5.4, 4.2), dpi=180)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    positive = [row for row in rows if row["allan_deviation_hz"] > 0]
    if positive:
        y_key = ("mass_equivalent_noise_ug" if sensitivity_pg_hz is not None
                 else "allan_deviation_hz")
        ax.loglog([row["tau_s"] for row in positive],
                  [row[y_key] for row in positive], "o-", color="#5968a6",
                  linewidth=1.6, markersize=4)
    ax.set_xlabel("Averaging time, τ (s)")
    ax.set_ylabel("Mass-equivalent noise (µg)" if sensitivity_pg_hz is not None
                  else "Allan deviation (Hz)")
    ax.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(png_path)
    return csv_path, png_path
