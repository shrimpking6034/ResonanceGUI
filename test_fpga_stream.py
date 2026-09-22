import unittest

import numpy as np

from fpga_registers import frequency_to_word
from fpga_stream import PacketSequenceTracker, decode_packet, make_test_packet, parse_header


class FPGAStreamTests(unittest.TestCase):
    def test_big_endian_frequency_packet(self):
        packet = make_test_packet(7, 0x1234,
                                  [frequency_to_word(1000), frequency_to_word(2000)])
        decoded = decode_packet(packet)
        self.assertEqual(decoded.header.channel_id, 7)
        self.assertEqual(decoded.header.packet_index, 0x1234)
        np.testing.assert_allclose(decoded.values, [1000, 2000], atol=0.002)

    def test_little_endian_packet_is_explicit(self):
        packet = make_test_packet(3, 9, [1, 0xFFFFFFFF], byte_order="little")
        decoded = decode_packet(packet, signal="unsigned", byte_order="little")
        self.assertEqual(decoded.header.channel_id, 3)
        np.testing.assert_array_equal(decoded.values, [1, 0xFFFFFFFF])

    def test_signed_payload(self):
        packet = make_test_packet(0, 1, [0xFFFFFFFF, 0x80000000, 0x7FFFFFFF])
        decoded = decode_packet(packet, signal="signed")
        np.testing.assert_array_equal(decoded.values, [-1, -(2**31), 2**31-1])

    def test_malformed_packet_is_rejected(self):
        for packet in (b"", b"1234", b"123456789"):
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                decode_packet(packet)

    def test_sequence_gap_duplicate_reorder_wrap_and_restart(self):
        tracker = PacketSequenceTracker(restart_backstep=10)
        def observe(index):
            return tracker.observe(parse_header((index << 8) | 2))
        self.assertEqual(observe(10).classification, "first")
        self.assertEqual(observe(11).classification, "continuous")
        gap = observe(14)
        self.assertEqual((gap.classification, gap.missing_packets), ("gap", 2))
        self.assertEqual(observe(14).classification, "duplicate")
        self.assertEqual(observe(13).classification, "reordered")
        restart = observe(0)
        self.assertEqual((restart.classification, restart.segment_id), ("restart", 1))

        tracker = PacketSequenceTracker()
        tracker.observe(parse_header((0xFFFF << 8) | 1))
        self.assertEqual(tracker.observe(parse_header(1)).classification, "wrap")

    def test_channels_have_independent_sequences(self):
        tracker = PacketSequenceTracker()
        self.assertEqual(tracker.observe(parse_header((4 << 8) | 1)).classification, "first")
        self.assertEqual(tracker.observe(parse_header((99 << 8) | 2)).classification, "first")
        self.assertEqual(tracker.observe(parse_header((5 << 8) | 1)).classification, "continuous")


if __name__ == "__main__":
    unittest.main()
