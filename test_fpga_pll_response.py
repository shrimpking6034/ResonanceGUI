import tempfile
import unittest
from pathlib import Path

import numpy as np

from fpga_pll_response import analyze_step_response, save_step_response


class PLLStepResponseTests(unittest.TestCase):
    def test_drift_and_underresolved_transition(self):
        t=np.arange(0,3,.001)
        y=39000+.12*t
        mask=t>=.5
        y[mask]+=2*(1-np.exp(-(t[mask]-.5)/.02))
        result=analyze_step_response(t,y,step_time_s=.5,restore_time_s=2.2,delay_step_turns=.001)
        self.assertTrue(result.accepted,result.reasons)
        self.assertAlmostEqual(result.tau_s,.02,delta=.001)
        self.assertAlmostEqual(result.drift_hz_per_s,.12,delta=.01)
        fast=y.copy()
        fast[mask]=39000+.12*t[mask]+2*(1-np.exp(-(t[mask]-.5)/.0002))
        result=analyze_step_response(t,fast,step_time_s=.5,restore_time_s=2.2,delay_step_turns=.001)
        self.assertFalse(result.accepted)

    def test_recovers_first_order_bandwidth_and_saves_artifacts(self):
        rng = np.random.default_rng(12)
        rate = 1000.0
        t = np.arange(0.0, 3.0, 1.0 / rate)
        step_time, restore_time = 0.5, 2.2
        tau = 0.025
        y = np.full_like(t, 39000.0)
        active = (t >= step_time) & (t < restore_time)
        y[active] += 3.0 * (1.0 - np.exp(-(t[active] - step_time) / tau))
        y[t >= restore_time] += 3.0 * np.exp(-(t[t >= restore_time] - restore_time) / tau)
        y += rng.normal(0.0, 0.02, len(t))
        result = analyze_step_response(
            t, y, step_time_s=step_time, restore_time_s=restore_time,
            delay_step_turns=1 / 2048,
        )
        self.assertTrue(result.accepted, result.reasons)
        self.assertAlmostEqual(result.tau_s, tau, delta=0.002)
        self.assertAlmostEqual(result.bandwidth_hz, 1 / (2 * np.pi * tau), delta=0.5)
        self.assertGreater(result.r_squared, 0.99)
        with tempfile.TemporaryDirectory() as directory:
            segment = np.where(t < step_time, "baseline",
                               np.where(t < restore_time, "step", "recovery"))
            paths = save_step_response(
                directory, t, y, segment, result,
                {"step_time_s": step_time, "restore_time_s": restore_time})
            self.assertTrue(all(Path(path).exists() for path in paths))

    def test_flags_response_that_is_not_first_order(self):
        t = np.linspace(0.0, 3.0, 3001)
        y = np.full_like(t, 10000.0)
        active = (t >= 0.5) & (t < 2.2)
        y[active] += 2.0 * np.sin(2 * np.pi * 15 * (t[active] - 0.5))
        result = analyze_step_response(
            t, y, step_time_s=0.5, restore_time_s=2.2,
            delay_step_turns=1 / 2048)
        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()
