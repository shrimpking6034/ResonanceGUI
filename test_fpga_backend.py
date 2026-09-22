import unittest

import numpy as np

from fpga_backend import FPGAController, SimulatedFPGAController
from fpga_registers import DEFAULT_REGISTER_ADDRESSES, ModeFields, PLLUserParameters
from fpga_transport import NetworkProfile


class FakeTCP:
    def __init__(self):
        self.sock = object()
        self.writes = []
        self.values = {}

    def write_register(self, address, value):
        self.writes.append((address, value))
        self.values[address] = value
        return "ok"

    def read_register(self, address):
        return self.values[address]

    def close(self):
        self.sock = None


class SimulatedBackendTests(unittest.TestCase):
    def setUp(self):
        self.controller = SimulatedFPGAController(seed=7)
        self.controller.connect()
        self.parameters = PLLUserParameters(
            frequency_hz=29400, minimum_frequency_hz=29350,
            maximum_frequency_hz=29450, resonator_q=1200,
            loop_bandwidth_hz=3, loop_order=1, cic_rate=256,
            drive_amplitude_fraction=0.02,
            mode=ModeFields(send_data_to_pc=True,
                            run_nco_at_fixed_frequency=False))

    def test_sweep_response_peaks_near_resonance(self):
        frequencies = np.linspace(29300, 29500, 401)
        amplitude = np.asarray([abs(self.controller.complex_response(f)[0])
                                for f in frequencies])
        self.assertAlmostEqual(frequencies[np.argmax(amplitude)], 29400, delta=1)

    def test_tracking_requires_configuration(self):
        with self.assertRaises(RuntimeError):
            self.controller.start_tracking()
        self.controller.apply_parameters(self.parameters)
        self.controller.start_tracking()
        t, f, z = self.controller.sample_tracking(100, 1000)
        self.assertEqual((len(t), len(f), len(z)), (100, 100, 100))
        self.assertTrue(np.all(np.isfinite(f)))
        self.controller.stop_output()
        self.assertFalse(self.controller.tracking)

    def test_disconnect_does_not_leave_simulated_output_running(self):
        self.controller.apply_parameters(self.parameters, start=True)
        self.controller.disconnect()
        self.assertFalse(self.controller.connected)
        self.assertFalse(self.controller.running)


class HardwareBackendOrderingTests(unittest.TestCase):
    def test_stop_attempts_drive_zero_if_mode_write_fails(self):
        class FailModeTCP(FakeTCP):
            def write_register(self, address, value):
                if address == 0:
                    raise OSError("mode write failed")
                return super().write_register(address, value)
        controller = FPGAController(NetworkProfile())
        controller.tcp = FailModeTCP()
        with self.assertRaisesRegex(RuntimeError, "Run=0"):
            controller.stop_output()
        self.assertIn((19, 0), controller.tcp.writes)

    def test_captured_sweep_preset_matches_working_driver(self):
        controller = FPGAController(NetworkProfile())
        controller.tcp = FakeTCP()
        controller.apply_verified_sweep_preset(117000, 119000, 0.02)
        writes = controller.tcp.writes
        self.assertEqual(writes[0], (0, 0))
        self.assertEqual(writes[1], (19, 0))
        self.assertIn((14, 1605631), writes)
        self.assertIn((17, 77213), writes)
        self.assertIn((18, 0xFFFFFFFF), writes)
        self.assertEqual(writes[-1], (0, 0x8040E))
        controller.start_fixed(stream=True)
        self.assertEqual(writes[-1], (0, 0x8040F))
        controller.stop_output()
        self.assertEqual(writes[-2], (0, 0))
        self.assertEqual(writes[-1], (19, 0))

    def test_order_one_tracking_control_and_restart(self):
        controller = FPGAController(NetworkProfile())
        controller.tcp = FakeTCP()
        parameters = PLLUserParameters(
            frequency_hz=118000, minimum_frequency_hz=110000,
            maximum_frequency_hz=120000, resonator_q=140,
            loop_bandwidth_hz=30, loop_order=1, cic_rate=32767,
            cic_bit_shift=21, drive_amplitude_fraction=0.02,
            mode=ModeFields(enable_agc=True, send_data_to_pc=True,
                            dac_a_output=1, run_nco_at_fixed_frequency=False))
        controller.apply_parameters(parameters)
        from fpga_registers import user_parameters_to_registers
        words = user_parameters_to_registers(parameters)
        self.assertIn((17, words.gain_proportional), controller.tcp.writes)
        self.assertIn((18, words.gain_integral), controller.tcp.writes)
        controller.start_tracking(stream=True)
        self.assertEqual(controller.tcp.writes[-1], (0, 0x80007))
        controller.restart_all_plls()
        self.assertEqual(controller.tcp.writes[-2:],
                         [(0xFF00, 0), (0xFF00, 0x80007)])

    def test_configuration_is_stopped_and_mode_is_written_last(self):
        controller = FPGAController(NetworkProfile(), channel=2, channel_stride=0x20)
        controller.tcp = FakeTCP()
        requested_mode = ModeFields(run=True, send_data_to_pc=True,
                                    run_nco_at_fixed_frequency=False)
        parameters = PLLUserParameters(
            frequency_hz=29400, minimum_frequency_hz=29350,
            maximum_frequency_hz=29450, resonator_q=1200,
            loop_bandwidth_hz=3, loop_order=1, cic_rate=256,
            drive_amplitude_fraction=0.02, mode=requested_mode)
        values = controller.apply_parameters(parameters, verify_readback=True)
        mode_address = DEFAULT_REGISTER_ADDRESSES["mode"] + 2*0x20
        self.assertEqual(controller.tcp.writes[-1][0], mode_address)
        self.assertEqual(controller.tcp.writes[-1][1] & 0b101, 0)
        self.assertFalse(controller.last_parameters.mode.run)
        self.assertFalse(controller.last_parameters.mode.send_data_to_pc)
        controller.start_tracking(stream=True)
        final_mode = controller.tcp.writes[-1][1]
        self.assertEqual(final_mode & 0b101, 0b101)
        self.assertEqual(final_mode & (1 << 3), 0)

    def test_fixed_frequency_writes_channel_offset_address_three(self):
        controller = FPGAController(NetworkProfile(), channel=3, channel_stride=0x20)
        controller.tcp = FakeTCP()
        controller.set_fixed_frequency(1000)
        self.assertEqual(controller.tcp.writes[-1][0], 3 + 3*0x20)


if __name__ == "__main__":
    unittest.main()
