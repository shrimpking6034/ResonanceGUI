import ctypes as ct
import unittest

from zi_hf2_api import (
    MAX_PATH_LEN, ZIEvent, ZIDemodSample, ZiHF2,
    ZI_VALUE_TYPE_DEMOD_SAMPLE,
)


class _FakeDLL:
    @staticmethod
    def ziAPIPollDataEx(_conn, _event, _timeout_ms):
        return 0


class ZiHF2StreamingTests(unittest.TestCase):
    def test_event_header_matches_64_bit_c_layout(self):
        self.assertEqual(ZIEvent.path.offset, 8)
        self.assertEqual(ZIEvent.value.offset, 8 + MAX_PATH_LEN)

    def test_poll_event_copies_all_demod_samples(self):
        samples = (ZIDemodSample * 2)()
        samples[0].timeStamp = 100
        samples[0].x = 1.0
        samples[0].y = 2.0
        samples[0].frequency = 1234.5
        samples[1].timeStamp = 110
        samples[1].x = 3.0
        samples[1].y = 4.0
        samples[1].frequency = 1235.5

        event = ZIEvent()
        event.valueType = ZI_VALUE_TYPE_DEMOD_SAMPLE
        event.count = 2
        path = b"/DEV1213/DEMODS/0/SAMPLE"
        for index, value in enumerate(path):
            event.path[index] = value
        event.value = ct.cast(samples, ct.c_void_p).value

        api = object.__new__(ZiHF2)
        api.dll = _FakeDLL()
        api.conn = None
        returned_path, copied = api.poll_event(ct.pointer(event), 5)

        self.assertEqual(returned_path, path.decode())
        self.assertEqual(len(copied), 2)
        self.assertEqual(copied[0][:4], (100, 1.0, 2.0, 1234.5))
        self.assertEqual(copied[1][:4], (110, 3.0, 4.0, 1235.5))


if __name__ == "__main__":
    unittest.main()
