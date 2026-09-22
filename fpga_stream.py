"""Loss-aware decoder for the legacy SMR multicast sample stream."""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Literal

import numpy as np

from fpga_registers import NCO_CLOCK_HZ


@dataclass(frozen=True)
class PacketHeader:
    channel_id: int
    packet_index: int
    reserved: int
    raw_word: int


@dataclass(frozen=True)
class DecodedPacket:
    header: PacketHeader
    values: np.ndarray
    raw_words: np.ndarray


@dataclass(frozen=True)
class SequenceResult:
    classification: str
    missing_packets: int
    segment_id: int


def parse_header(word: int) -> PacketHeader:
    word = int(word) & 0xFFFFFFFF
    return PacketHeader(channel_id=word & 0xFF,
                        packet_index=(word >> 8) & 0xFFFF,
                        reserved=(word >> 24) & 0xFF,
                        raw_word=word)


def decode_packet(data: bytes, *, signal: Literal["frequency", "unsigned", "signed"] = "frequency",
                  byte_order: Literal["big", "little"] = "big") -> DecodedPacket:
    if len(data) < 8 or len(data) % 4:
        raise ValueError("SMR packet must contain a header and payload of 32-bit words")
    dtype = ">u4" if byte_order == "big" else "<u4"
    words = np.frombuffer(data, dtype=dtype).astype(np.uint32, copy=True)
    header = parse_header(int(words[0]))
    payload = words[1:]
    if signal == "frequency":
        values = payload.astype(np.float64) * NCO_CLOCK_HZ / 2**32
    elif signal == "signed":
        values = payload.view(np.int32).astype(np.float64)
    elif signal == "unsigned":
        values = payload.astype(np.float64)
    else:
        raise ValueError(f"unsupported signal decoder: {signal}")
    return DecodedPacket(header, values, words)


class PacketSequenceTracker:
    def __init__(self, restart_backstep: int = 1024):
        self.last: dict[int, int] = {}
        self.segments: dict[int, int] = {}
        self.restart_backstep = int(restart_backstep)

    def observe(self, header: PacketHeader) -> SequenceResult:
        channel, current = header.channel_id, header.packet_index
        segment = self.segments.get(channel, 0)
        if channel not in self.last:
            classification, missing = "first", 0
        else:
            previous = self.last[channel]
            delta = (current - previous) & 0xFFFF
            if delta == 0:
                return SequenceResult("duplicate", 0, segment)
            if delta == 1:
                classification, missing = ("wrap" if current < previous else "continuous"), 0
            elif 1 < delta < 0x8000:
                classification, missing = "gap", delta - 1
            else:
                backstep = previous - current
                if backstep >= self.restart_backstep:
                    segment += 1
                    self.segments[channel] = segment
                    classification, missing = "restart", 0
                else:
                    return SequenceResult("reordered", 0, segment)
        self.last[channel] = current
        self.segments[channel] = segment
        return SequenceResult(classification, missing, segment)


def make_test_packet(channel: int, packet_index: int, words: list[int],
                     byte_order: Literal["big", "little"] = "big") -> bytes:
    header = (int(packet_index) & 0xFFFF) << 8 | (int(channel) & 0xFF)
    fmt = ">" if byte_order == "big" else "<"
    return struct.pack(fmt + "I" * (1 + len(words)), header, *[w & 0xFFFFFFFF for w in words])
