import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from session_storage import PlotterSpool, RAW_COLUMNS, atomic_json, atomic_sweep


class SessionStorageTests(unittest.TestCase):
    def test_spool_exports_every_row_then_discards_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = PlotterSpool()
            temp_path = spool.path
            rows = [["session", index / 100, 5000 + index] + [""] * (len(RAW_COLUMNS) - 3)
                    for index in range(5000)]
            for offset in range(0, len(rows), 500):
                spool.append(rows[offset:offset + 500])
            output = Path(directory) / "raw.csv"
            spool.export(output, {"SessionID": "session", "Logged_row_count": spool.rows})
            with output.open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.reader(handle))
            self.assertEqual(exported[0], ["# Metadata", "SessionID", "session"])
            self.assertEqual(exported[2], RAW_COLUMNS)
            self.assertEqual(exported[3:], [[str(value) for value in row] for row in rows])
            spool.discard()
            self.assertFalse(temp_path.exists())
            self.assertTrue(output.exists())

    def test_last_settings_and_sweep_are_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "last_settings.json"
            sweep = Path(directory) / "last_sweep.npz"
            atomic_json(settings, {"values": {"pll_center_spin": 12345.0}})
            atomic_json(settings, {"values": {"pll_center_spin": 23456.0}})
            self.assertEqual(json.loads(settings.read_text())["values"]["pll_center_spin"], 23456.0)
            atomic_sweep(sweep, {"freqs": np.array([1., 2.]), "R": np.array([.1, .2])})
            atomic_sweep(sweep, {"freqs": np.array([3., 4.]), "R": np.array([.3, .4])})
            with np.load(sweep, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["freqs"], [3., 4.])


if __name__ == "__main__":
    unittest.main()
