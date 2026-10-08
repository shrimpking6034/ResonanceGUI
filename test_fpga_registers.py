import math
import unittest

from fpga_registers import (
    DEFAULT_REGISTER_ADDRESSES, NCO_CLOCK_HZ, ModeFields, PLLUserParameters,
    RegisterValueError, frequency_to_word, register_write_map,
    pll_update_rate_hz, signed_to_wire, user_parameters_to_registers,
    word_to_frequency,
)


class FPGARegisterTests(unittest.TestCase):
    def parameters(self, **changes):
        base = dict(
            frequency_hz=29400.0, minimum_frequency_hz=29200.0,
            maximum_frequency_hz=29600.0, resonator_q=1200.0,
            loop_bandwidth_hz=3.0, loop_order=2, cic_rate=256,
            cic_bit_shift=4, pll_delay_turns=0.125,
            drive_amplitude_fraction=0.02,
        )
        base.update(changes)
        return PLLUserParameters(**base)

    def test_frequency_word_round_trip_is_within_half_lsb(self):
        value = 29400.12345
        result = word_to_frequency(frequency_to_word(value))
        self.assertLessEqual(abs(result-value), 0.5*NCO_CLOCK_HZ/2**32)

    def test_pll_update_rate_follows_100mhz_cic_decimation(self):
        self.assertAlmostEqual(pll_update_rate_hz(32767), 100e6 / 32767)
        self.assertAlmostEqual(pll_update_rate_hz(16384), 100e6 / 16384)
        with self.assertRaises(RegisterValueError):
            pll_update_rate_hz(15)

    def test_nco_clock_itself_is_rejected_instead_of_wrapping_to_zero(self):
        with self.assertRaises(RegisterValueError):
            frequency_to_word(NCO_CLOCK_HZ)

    def test_mode_formula_matches_labview_comment(self):
        mode = ModeFields(
            run=True, enable_agc=True, send_data_to_pc=True,
            run_nco_at_fixed_frequency=True, impulse=True, input_source=2,
            signal_of_interest=5, dac_a_output=3, dac_b_output=4,
            pll_datarate_decimation=2)
        expected = (1 + 2 + 2**2 + 2**3 + 2**5 + 2*2**6 + 5*2**8
                    + 3*2**19 + 4*2**22 + 2*2**25)
        self.assertEqual(mode.encode(), expected)

    def test_decimator_and_fractional_words_match_formula(self):
        result = user_parameters_to_registers(self.parameters())
        self.assertEqual(result.decimator_control, 256 + 4*2**16)
        self.assertEqual(result.delay, round(0.125*2**32))
        self.assertEqual(result.nco_gain, round(0.02*(2**32-1)))

    def test_order_one_has_two_identity_q30_filters(self):
        result = user_parameters_to_registers(self.parameters(loop_order=1))
        self.assertEqual(result.sos_filter_0_b0, 2**30)
        self.assertEqual(result.sos_filter_1_b0, 2**30)
        for name, value in result.as_dict().items():
            if name.startswith("sos_") and not name.endswith("b0"):
                self.assertEqual(value, 0)

    def test_order_five_coefficients_are_finite_signed_words(self):
        result = user_parameters_to_registers(self.parameters(loop_order=5))
        coefficients = [value for name, value in result.as_dict().items()
                        if name.startswith("sos_")]
        self.assertEqual(len(coefficients), 10)
        self.assertTrue(all(-(2**31) <= value < 2**31 for value in coefficients))

    def test_negative_sos_is_serialized_as_twos_complement(self):
        self.assertEqual(signed_to_wire(-1), 0xFFFFFFFF)
        writes = register_write_map(user_parameters_to_registers(self.parameters()))
        self.assertIn(DEFAULT_REGISTER_ADDRESSES["phase_increment_upon_reset"], writes)
        self.assertTrue(all(0 <= word <= 0xFFFFFFFF for word in writes.values()))

    def test_known_register_addresses_follow_saved_vi_write_order(self):
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["phase_increment_upon_reset"], 3)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["phase_increment_minimum"], 1)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["phase_increment_maximum"], 2)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["decimator_control"], 0x0E)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["delay"], 0x05)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["gain_proportional"], 0x11)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["gain_integral"], 0x12)
        self.assertEqual(DEFAULT_REGISTER_ADDRESSES["nco_gain"], 0x13)

    def test_first_order_pi_words_follow_vi_frequency_and_cic_formula(self):
        for frequency, cic in ((29400, 32767), (118000, 32767), (29400, 256)):
            with self.subTest(frequency=frequency, cic=cic):
                p = self.parameters(frequency_hz=frequency,
                                    minimum_frequency_hz=frequency-200,
                                    maximum_frequency_hz=frequency+200,
                                    cic_rate=cic, loop_order=1,
                                    loop_bandwidth_hz=20, resonator_q=47)
                result = user_parameters_to_registers(p)
                beta = 2 * math.pi * p.loop_bandwidth_hz
                tau = p.resonator_q / (p.frequency_hz * math.pi)
                compensation = 2 / .89 * 8 / (2 * math.pi)
                self.assertEqual(result.gain_proportional,
                                 round(beta * 1e-8 * compensation * 2**32))
                self.assertEqual(result.gain_integral,
                                 round(beta * 1e-16 * cic / tau *
                                       compensation * 2**8 * 2**32))

    def test_invalid_parameters_fail_before_hardware_write(self):
        for changes in (
            {"cic_rate": 15}, {"cic_bit_shift": 32}, {"loop_order": 6},
            {"drive_amplitude_fraction": 1.1},
            {"minimum_frequency_hz": 29500, "maximum_frequency_hz": 29600},
        ):
            with self.subTest(changes=changes), self.assertRaises(RegisterValueError):
                user_parameters_to_registers(self.parameters(**changes))


if __name__ == "__main__":
    unittest.main()
