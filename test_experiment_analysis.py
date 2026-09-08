import unittest

import numpy as np

from experiment_analysis import integration_diagnostics


class IntegrationDiagnosticsTests(unittest.TestCase):
    def test_received_rows_timestamp_pll_and_pressure_checks(self):
        t = np.arange(0., 10., .01)
        n = len(t)
        result = integration_diagnostics(
            t, n, pll_status_age_s=np.full(n, .05), demod_latency_s=np.full(n, .001),
            pressure_command_a=np.ones(n), pressure_command_b=np.zeros(n),
            measured_pressure_a=np.full(n, 2.), measured_pressure_b=np.full(n, 1.),
            expected_pll_update_s=.1, ui_refresh_latency_s=np.full(100, .005),
        )
        self.assertTrue(result["checks"]["received_vs_logged"]["pass"])
        self.assertTrue(result["checks"]["timestamps"]["pass"])
        self.assertTrue(result["checks"]["pll_status_update"]["pass"])
        self.assertTrue(result["checks"]["pressure_recording"]["pass"])
        self.assertTrue(result["checks"]["overall_pass"])

    def test_missing_rows_fails(self):
        result = integration_diagnostics(np.arange(10.), 11)
        self.assertFalse(result["checks"]["received_vs_logged"]["pass"])
        self.assertFalse(result["checks"]["overall_pass"])


if __name__ == "__main__":
    unittest.main()
