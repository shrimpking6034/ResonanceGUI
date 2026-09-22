import tempfile
import unittest
from pathlib import Path
import csv

import numpy as np

from baseline_analysis import load_plotter_csv
from fpga_resonance_gui import write_tracking_csv


class FPGAGUIDataTests(unittest.TestCase):
    def test_amplitude_stream_keeps_counts_separate_from_voltage(self):
        row = (0.0, np.nan, np.nan, np.nan, np.nan, np.nan,
               0, "header_unverified", 0, 0, 1, 123456.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amplitude.csv"
            write_tracking_csv(path, [row], backend="legacy-smr-fpga",
                               time_source="host", stream_rate_hz=3051,
                               stream_signal="amplitude_counts")
            with path.open(newline="", encoding="utf-8-sig") as handle:
                lines = list(csv.reader(handle))
        self.assertEqual(lines[5][-1], "Amplitude_counts")
        self.assertEqual(lines[6][-1], "123456.0")
        self.assertEqual(lines[6][2].lower(), "nan")

    def test_tracking_export_opens_in_existing_analysis_loader(self):
        rows = [
            (0.0, 29400.0, np.nan, np.nan, np.nan, np.nan,
             10, "first", 0, 0, 1),
            (0.001, 29399.9, np.nan, np.nan, np.nan, np.nan,
             10, "payload", 0, 0, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.csv"
            write_tracking_csv(
                path, rows, backend="legacy-smr-fpga",
                time_source="reconstructed_from_user_stream_rate",
                stream_rate_hz=1000, local_queue_dropped_batches=0)
            loaded = load_plotter_csv(path)
        np.testing.assert_allclose(loaded["time_s"], [0.0, 0.001])
        np.testing.assert_allclose(loaded["frequency_hz"], [29400.0, 29399.9])
        self.assertTrue(np.all(np.isnan(loaded["amplitude_v"])))
        self.assertEqual(loaded["metadata"]["Backend"], "legacy-smr-fpga")


if __name__ == "__main__":
    unittest.main()
