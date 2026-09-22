import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from allan_export import allan_export_rows, save_allan_export
from baseline_analysis import allan_deviation


class AllanExportTests(unittest.TestCase):
    def test_export_preserves_frequency_adev_and_converts_calibrated_mass(self):
        # Alternating cluster means differ by 2 Hz: sqrt(1/2 * 2²) = sqrt(2) Hz.
        time_s = np.arange(0, 8, .25)
        frequency_hz = 100. + 2. * (np.floor(time_s).astype(int) % 2)
        allan = allan_deviation(time_s, frequency_hz, taus=(1.,))
        self.assertAlmostEqual(allan[0]["allan_deviation_hz"], np.sqrt(2.))
        rows = allan_export_rows(allan, sensitivity_pg_hz=2e6)
        self.assertAlmostEqual(rows[0]["mass_equivalent_noise_ug"], 2 * np.sqrt(2.))

        with tempfile.TemporaryDirectory() as directory:
            csv_path, png_path = save_allan_export(Path(directory) / "allan.csv", allan, 2e6)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 1)
            self.assertAlmostEqual(float(saved[0]["allan_deviation_hz"]), np.sqrt(2.))
            self.assertAlmostEqual(float(saved[0]["mass_equivalent_noise_ug"]), 2 * np.sqrt(2.))
            self.assertGreater(png_path.stat().st_size, 1000)

    def test_uncalibrated_export_has_no_mass_value(self):
        rows = allan_export_rows([{"tau_s": 1., "allan_deviation_hz": .2,
                                   "cluster_count": 3, "method": "test"}])
        self.assertIsNone(rows[0]["mass_equivalent_noise_ug"])
        with self.assertRaises(ValueError):
            allan_export_rows([{"tau_s": 1., "allan_deviation_hz": .2,
                                "cluster_count": 3, "method": "test"}], 0)


if __name__ == "__main__":
    unittest.main()
