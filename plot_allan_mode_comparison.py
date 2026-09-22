"""Plot frequency Allan deviation for resonant modes from exported CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "Noise Data" / "processed"
SOURCES = [
    ("4th mode", PROCESSED / "allan_20260920_211818_dd0e3f5b_4thmode.csv"),
    ("5th mode", PROCESSED / "allan_20260920_210408_f94f0e6d_5thmode.csv"),
    ("5th mode (SR560)", PROCESSED / "sr560" /
     "allan_20260921_133740_a9190a49_5thmode_SR560.csv"),
    ("6th mode", PROCESSED / "allan_20260918_180332_06f1fad5_6thmode_without_sensitivity.csv"),
]
OUTPUT = ROOT / "allan_deviation_4th_5th_SR560_6th_modes.png"


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    styles = [
        ("#3b82c4", "o"),
        ("#d65f5f", "s"),
        ("#8a5bb8", "D"),
        ("#4c9f70", "^"),
    ]
    for (label, path), (color, marker) in zip(SOURCES, styles):
        data = pd.read_csv(path, encoding="utf-8-sig")
        required = {"tau_s", "allan_deviation_hz"}
        if not required.issubset(data.columns):
            raise ValueError(f"{path.name}: required columns are missing")
        valid = data["tau_s"].gt(0) & data["allan_deviation_hz"].gt(0)
        ax.loglog(
            data.loc[valid, "tau_s"],
            data.loc[valid, "allan_deviation_hz"],
            marker=marker,
            markersize=5,
            linewidth=1.8,
            color=color,
            label=label,
        )

    ax.set_title("Allan Deviation by Resonant Mode")
    ax.set_xlabel("Averaging time, τ (s)")
    ax.set_ylabel("Allan deviation (Hz)")
    ax.grid(True, which="major", color="#b8bec7", alpha=0.55, linewidth=0.7)
    ax.grid(True, which="minor", color="#d9dde3", alpha=0.4, linewidth=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#c8cdd4")
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
