"""Minimal ctypes wrapper around the legacy ziAPI (C API) bundled with
LabOne 19.05, used to talk to the HF2 Data Server (ziServer) on port 8005.

The modern `zhinst` PyPI package (LabOne API level 6) is not protocol
compatible with this HF2's old Data Server (reports itself as version
'0.0.6'), so we drive the native C DLL directly instead.
"""
import ctypes as ct
import sys
from pathlib import Path

# Prefer the DLL bundled next to a PyInstaller executable, then fall back to
# the normal LabOne installation for direct Python execution.
_BUNDLED_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_BUNDLED_DLL = _BUNDLED_DIR / "ziAPI-win64.dll"
DLL_PATH = str(_BUNDLED_DLL) if _BUNDLED_DLL.is_file() else (
    r"C:\Program Files\Zurich Instruments\LabOne\API\C\lib\ziAPI-win64.dll"
)

ZIDoubleData = ct.c_double
ZIIntegerData = ct.c_int64
ZIResult = ct.c_int
ZIConnection = ct.c_void_p
ZITimeStamp = ct.c_uint64

ZI_INFO_SUCCESS = 0x0000
ZI_VALUE_TYPE_DEMOD_SAMPLE = 3
MAX_PATH_LEN = 256


class ZIDemodSample(ct.Structure):
    _fields_ = [
        ("timeStamp", ZITimeStamp),
        ("x", ct.c_double),
        ("y", ct.c_double),
        ("frequency", ct.c_double),
        ("phase", ct.c_double),
        ("dioBits", ct.c_uint32),
        ("trigger", ct.c_uint32),
        ("auxIn0", ct.c_double),
        ("auxIn1", ct.c_double),
    ]


class ZIEvent(ct.Structure):
    """Header of the variable-sized event allocated by the LabOne C API.

    The DLL owns the trailing data buffer.  ``value`` points at the first
    typed item in that buffer, so the Python declaration only needs the
    fixed header fields to read a polled event safely.
    """
    _fields_ = [
        ("valueType", ct.c_uint32),
        ("count", ct.c_uint32),
        ("path", ct.c_uint8 * MAX_PATH_LEN),
        ("value", ct.c_void_p),
    ]


class ZiAPIError(RuntimeError):
    pass


class ZiHF2:
    def __init__(self, host="localhost", port=8005):
        self.dll = ct.WinDLL(DLL_PATH)
        self._setup_signatures()
        self.conn = ZIConnection()
        self._check(self.dll.ziAPIInit(ct.byref(self.conn)))
        try:
            self._check(self.dll.ziAPIConnect(self.conn, host.encode(), port))
        except Exception:
            self.dll.ziAPIDestroy(self.conn)
            raise

    def _setup_signatures(self):
        d = self.dll
        d.ziAPIInit.argtypes = [ct.POINTER(ZIConnection)]
        d.ziAPIInit.restype = ZIResult
        d.ziAPIConnect.argtypes = [ZIConnection, ct.c_char_p, ct.c_uint16]
        d.ziAPIConnect.restype = ZIResult
        d.ziAPIDisconnect.argtypes = [ZIConnection]
        d.ziAPIDisconnect.restype = ZIResult
        d.ziAPIDestroy.argtypes = [ZIConnection]
        d.ziAPIDestroy.restype = ZIResult
        d.ziAPIGetValueD.argtypes = [ZIConnection, ct.c_char_p, ct.POINTER(ZIDoubleData)]
        d.ziAPIGetValueD.restype = ZIResult
        d.ziAPISetValueD.argtypes = [ZIConnection, ct.c_char_p, ZIDoubleData]
        d.ziAPISetValueD.restype = ZIResult
        d.ziAPIGetValueI.argtypes = [ZIConnection, ct.c_char_p, ct.POINTER(ZIIntegerData)]
        d.ziAPIGetValueI.restype = ZIResult
        d.ziAPISetValueI.argtypes = [ZIConnection, ct.c_char_p, ZIIntegerData]
        d.ziAPISetValueI.restype = ZIResult
        d.ziAPIGetValueString.argtypes = [ZIConnection, ct.c_char_p, ct.c_char_p, ct.c_uint32]
        d.ziAPIGetValueString.restype = ZIResult
        d.ziAPIListNodes.argtypes = [ZIConnection, ct.c_char_p, ct.c_char_p, ct.c_uint32, ct.c_int32]
        d.ziAPIListNodes.restype = ZIResult
        d.ziAPIGetLastError.argtypes = [ZIConnection, ct.c_char_p, ct.c_uint32]
        d.ziAPIGetLastError.restype = ZIResult
        d.ziAPIGetDemodSample.argtypes = [ZIConnection, ct.c_char_p, ct.POINTER(ZIDemodSample)]
        d.ziAPIGetDemodSample.restype = ZIResult
        d.ziAPISubscribe.argtypes = [ZIConnection, ct.c_char_p]
        d.ziAPISubscribe.restype = ZIResult
        d.ziAPIUnSubscribe.argtypes = [ZIConnection, ct.c_char_p]
        d.ziAPIUnSubscribe.restype = ZIResult
        d.ziAPIAllocateEventEx.argtypes = []
        d.ziAPIAllocateEventEx.restype = ct.POINTER(ZIEvent)
        d.ziAPIPollDataEx.argtypes = [ZIConnection, ct.POINTER(ZIEvent), ct.c_uint32]
        d.ziAPIPollDataEx.restype = ZIResult
        d.ziAPIDeallocateEventEx.argtypes = [ct.POINTER(ZIEvent)]
        d.ziAPIDeallocateEventEx.restype = None
        d.ziAPISync.argtypes = [ZIConnection]
        d.ziAPISync.restype = ZIResult

    def _check(self, result):
        if result != ZI_INFO_SUCCESS:
            buf = ct.create_string_buffer(1024)
            try:
                self.dll.ziAPIGetLastError(self.conn, buf, 1024)
                msg = buf.value.decode(errors="replace")
            except Exception:
                msg = ""
            raise ZiAPIError(f"ziAPI call failed, code=0x{result:x} {msg}")
        return result

    def list_nodes(self, path, flags=0):
        buf = ct.create_string_buffer(65536)
        self._check(self.dll.ziAPIListNodes(self.conn, path.encode(), buf, len(buf), flags))
        text = buf.value.decode(errors="replace")
        return [n for n in text.split("\n") if n]

    def get_string(self, path):
        buf = ct.create_string_buffer(4096)
        self._check(self.dll.ziAPIGetValueString(self.conn, path.encode(), buf, len(buf)))
        return buf.value.decode(errors="replace")

    def get_double(self, path):
        val = ZIDoubleData()
        self._check(self.dll.ziAPIGetValueD(self.conn, path.encode(), ct.byref(val)))
        return val.value

    def set_double(self, path, value):
        self._check(self.dll.ziAPISetValueD(self.conn, path.encode(), ZIDoubleData(value)))

    def get_int(self, path):
        val = ZIIntegerData()
        self._check(self.dll.ziAPIGetValueI(self.conn, path.encode(), ct.byref(val)))
        return val.value

    def set_int(self, path, value):
        self._check(self.dll.ziAPISetValueI(self.conn, path.encode(), ZIIntegerData(value)))

    def get_demod_sample(self, path):
        sample = ZIDemodSample()
        self._check(self.dll.ziAPIGetDemodSample(self.conn, path.encode(), ct.byref(sample)))
        return sample

    def subscribe(self, path):
        self._check(self.dll.ziAPISubscribe(self.conn, path.encode()))

    def unsubscribe(self, path):
        self._check(self.dll.ziAPIUnSubscribe(self.conn, path.encode()))

    def allocate_event(self):
        event = self.dll.ziAPIAllocateEventEx()
        if not event:
            raise ZiAPIError("ziAPIAllocateEventEx returned a null event pointer")
        return event

    def deallocate_event(self, event):
        if event:
            self.dll.ziAPIDeallocateEventEx(event)

    def poll_event(self, event, timeout_ms):
        """Poll one buffered event and copy its demod samples into Python tuples.

        Returned sample tuples are ``(timestamp, x, y, frequency, phase,
        dio_bits, trigger, aux_in_0, aux_in_1)``.  Copying is essential because
        the C event buffer is reused by the next poll call.
        """
        self._check(self.dll.ziAPIPollDataEx(
            self.conn, event, ct.c_uint32(max(0, int(timeout_ms)))
        ))
        header = event.contents
        path = bytes(header.path).split(b"\0", 1)[0].decode(errors="replace")
        if header.count == 0 or header.valueType != ZI_VALUE_TYPE_DEMOD_SAMPLE:
            return path, []
        if not header.value:
            raise ZiAPIError("Polled demod event has data count but a null value pointer")
        samples = ct.cast(header.value, ct.POINTER(ZIDemodSample))
        copied = []
        for index in range(int(header.count)):
            sample = samples[index]
            copied.append((
                int(sample.timeStamp), float(sample.x), float(sample.y),
                float(sample.frequency), float(sample.phase),
                int(sample.dioBits), int(sample.trigger),
                float(sample.auxIn0), float(sample.auxIn1),
            ))
        return path, copied

    def sync(self):
        self._check(self.dll.ziAPISync(self.conn))

    def close(self):
        try:
            self.dll.ziAPIDisconnect(self.conn)
        finally:
            self.dll.ziAPIDestroy(self.conn)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
