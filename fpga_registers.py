"""Register conversion for the legacy DE2-115 SMR PLL firmware.

PI and higher-order equations translate the Formula Node in
``UserParametersToRegisterValues.vi``. The register address mapping follows
the independently working ``FPGA_Python/fpga_gui.py``.
Transport and register addresses remain separate from the conversion math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


UINT32_MAX = 2**32 - 1
NCO_CLOCK_HZ = 12_500_000.0
NCO_MAX_HZ = UINT32_MAX * NCO_CLOCK_HZ / 2**32
PLL_SAMPLE_PERIOD_S = 1e-8
Q30 = 2**30


def pll_update_rate_hz(cic_rate: int) -> float:
    """Return the PLL/CIC update rate implied by the saved VI formula.

    The LabVIEW coefficient code uses ``Ts * CIC_rate`` as the decimated
    sample period, with Ts=10 ns.  This is also consistent with the working
    32767 setting producing about 3052 samples/s.  It does not include the
    separate optional UDP ``PLL_datarate_decimation`` mode field.
    """
    rate = int(cic_rate)
    if not 16 <= rate <= 32767:
        raise RegisterValueError("cic_rate must be 16..32767")
    return 1.0 / (PLL_SAMPLE_PERIOD_S * rate)


class RegisterValueError(ValueError):
    pass


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RegisterValueError(f"{name} must be finite")
    return value


def _range(name: str, value: float, low: float, high: float) -> float:
    value = _finite(name, value)
    if not low <= value <= high:
        raise RegisterValueError(f"{name} must be in [{low}, {high}], got {value}")
    return value


def _u32(value: float, name: str) -> int:
    rounded = int(round(_finite(name, value)))
    if not 0 <= rounded <= UINT32_MAX:
        raise RegisterValueError(f"{name} overflows uint32: {rounded}")
    return rounded


def _i32(value: float, name: str) -> int:
    rounded = int(round(_finite(name, value)))
    if not -(2**31) <= rounded <= 2**31 - 1:
        raise RegisterValueError(f"{name} overflows int32: {rounded}")
    return rounded


def signed_to_wire(value: int) -> int:
    """Return the uint32 two's-complement word for a signed coefficient."""
    if not -(2**31) <= int(value) <= 2**31 - 1:
        raise RegisterValueError(f"signed coefficient out of range: {value}")
    return int(value) & UINT32_MAX


def frequency_to_word(frequency_hz: float) -> int:
    frequency_hz = _range("frequency_hz", frequency_hz, 0.0, NCO_MAX_HZ)
    return _u32(frequency_hz / NCO_CLOCK_HZ * 2**32, "frequency word")


def word_to_frequency(word: int) -> float:
    if not 0 <= int(word) <= UINT32_MAX:
        raise RegisterValueError(f"frequency word outside uint32: {word}")
    return float(int(word)) * NCO_CLOCK_HZ / 2**32


@dataclass(frozen=True)
class ModeFields:
    run: bool = False
    enable_agc: bool = False
    send_data_to_pc: bool = False
    run_nco_at_fixed_frequency: bool = True
    impulse: bool = False
    input_source: int = 0
    signal_of_interest: int = 0
    dac_a_output: int = 0
    dac_b_output: int = 0
    pll_datarate_decimation: int = 0

    def encode(self) -> int:
        fields = {
            "input_source": (self.input_source, 2),
            "signal_of_interest": (self.signal_of_interest, 11),
            "dac_a_output": (self.dac_a_output, 7),
            "dac_b_output": (self.dac_b_output, 7),
            "pll_datarate_decimation": (self.pll_datarate_decimation, 7),
        }
        for name, (value, maximum) in fields.items():
            if not 0 <= int(value) <= maximum:
                raise RegisterValueError(f"{name} outside encoded range 0..{maximum}")
        return (
            int(self.run)
            | (int(self.enable_agc) << 1)
            | (int(self.send_data_to_pc) << 2)
            | (int(self.run_nco_at_fixed_frequency) << 3)
            | (int(self.impulse) << 5)
            | (int(self.input_source) << 6)
            | (int(self.signal_of_interest) << 8)
            | (int(self.dac_a_output) << 19)
            | (int(self.dac_b_output) << 22)
            | (int(self.pll_datarate_decimation) << 25)
        )


@dataclass(frozen=True)
class PLLUserParameters:
    frequency_hz: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    resonator_q: float
    loop_bandwidth_hz: float
    loop_order: int = 2
    cic_rate: int = 256
    cic_bit_shift: int = 0
    pll_delay_turns: float = 0.0
    drive_amplitude_fraction: float = 0.05
    feedback_delay: float = 0.0
    feedback_gain: float = 0.0
    mode: ModeFields = field(default_factory=ModeFields)

    def validate(self) -> None:
        f = _range("frequency_hz", self.frequency_hz, 0.0, NCO_MAX_HZ)
        fmin = _range("minimum_frequency_hz", self.minimum_frequency_hz, 0.0, NCO_MAX_HZ)
        fmax = _range("maximum_frequency_hz", self.maximum_frequency_hz, 0.0, NCO_MAX_HZ)
        if not fmin <= f <= fmax:
            raise RegisterValueError("frequency must lie between minimum and maximum")
        _range("resonator_q", self.resonator_q, 1e-12, 1e12)
        _range("loop_bandwidth_hz", self.loop_bandwidth_hz, 1e-12, 0.45 / (PLL_SAMPLE_PERIOD_S * self.cic_rate))
        if int(self.loop_order) not in range(1, 6):
            raise RegisterValueError("loop_order must be 1..5")
        if not 16 <= int(self.cic_rate) <= 32767:
            raise RegisterValueError("cic_rate must be 16..32767")
        if not 0 <= int(self.cic_bit_shift) <= 31:
            raise RegisterValueError("cic_bit_shift must be 0..31")
        _range("pll_delay_turns", self.pll_delay_turns, 0.0, 1.0)
        _range("drive_amplitude_fraction", self.drive_amplitude_fraction, 0.0, 1.0)
        _range("feedback_delay", self.feedback_delay, 0.0, 65535.0)
        _range("feedback_gain", self.feedback_gain, 0.0, 255.999)
        self.mode.encode()


@dataclass(frozen=True)
class PLLRegisterValues:
    mode: int
    phase_increment_upon_reset: int
    phase_increment_minimum: int
    phase_increment_maximum: int
    decimator_control: int
    gain_proportional: int
    gain_integral: int
    delay: int
    nco_gain: int
    feedback_delay: int
    feedback_gain: int
    sos_filter_0_b0: int
    sos_filter_0_b1: int
    sos_filter_0_b2: int
    sos_filter_0_a1: int
    sos_filter_0_a2: int
    sos_filter_1_b0: int
    sos_filter_1_b1: int
    sos_filter_1_b2: int
    sos_filter_1_a1: int
    sos_filter_1_a2: int

    def as_dict(self, *, wire_unsigned: bool = False) -> dict[str, int]:
        result = dict(self.__dict__)
        if wire_unsigned:
            for key in result:
                result[key] = result[key] & UINT32_MAX
        return result


def _filter_coefficients(order: int, beta: float, decimated_ts: float) -> tuple[float, ...]:
    identity = (1.0, 0.0, 0.0, 0.0, 0.0)
    if order == 1:
        return identity + identity
    if order == 2:
        pole = 1.412136 / 2.0 / beta
        b0 = decimated_ts / (decimated_ts + pole)
        a1 = -pole / (decimated_ts + pole)
        return (b0, 0.0, 0.0, a1, 0.0) + identity
    if order == 3:
        gamma = math.tan(1.412136 * beta * decimated_ts / 2.0)
        a0 = gamma**2 + 1.412136 * gamma + 1.0
        b0 = gamma**2 / a0
        return (b0, 2*b0, b0, 2*(gamma**2-1)/a0,
                (gamma**2-1.412136*gamma+1)/a0) + identity
    if order == 4:
        pole = 0.6673 / beta
        first = (decimated_ts/(decimated_ts+pole), 0.0, 0.0,
                 -pole/(decimated_ts+pole), 0.0)
        gamma = math.tan(1.3206 / 2.0 * beta * decimated_ts)
        a0 = gamma**2 + gamma/1.1846 + 1.0
        b0 = gamma**2/a0
        second = (b0, 2*b0, b0, 2*(gamma**2-1)/a0,
                  (gamma**2-gamma/1.1846+1)/a0)
        return first + second
    gamma0 = math.tan(1.2223 * beta * decimated_ts / 2.0)
    a00 = gamma0**2 + gamma0/2.0467 + 1.0
    b00 = gamma0**2/a00
    first = (b00, 2*b00, b00, 2*(gamma0**2-1)/a00,
             (gamma0**2-gamma0/2.0467+1)/a00)
    gamma1 = math.tan(1.4717 * beta * decimated_ts / 2.0)
    a01 = gamma1**2 + gamma1/0.5577 + 1.0
    b01 = gamma1**2/a01
    second = (b01, 2*b01, b01, 2*(gamma1**2-1)/a01,
              (gamma1**2-gamma1/0.5577+1)/a01)
    return first + second


def user_parameters_to_registers(p: PLLUserParameters) -> PLLRegisterValues:
    p.validate()
    beta = 2.0 * math.pi * p.loop_bandwidth_hz
    tau = p.resonator_q / (p.frequency_hz * math.pi)
    factors = (0.0, 1.0, 1.412136/2.0, 0.5, 0.3827, 0.3090)
    c = factors[int(p.loop_order)]
    compensation = 2.0/0.89 * 8.0/(2.0*math.pi)
    gp = c * beta * PLL_SAMPLE_PERIOD_S * compensation * 2**32
    gi = (c * beta * PLL_SAMPLE_PERIOD_S**2 * p.cic_rate / tau
          * compensation * 2**8 * 2**32)
    # The working Python driver's order-1 constants approximate this formula
    # near its example frequency/CIC setting. The VI formula depends on the
    # actual resonance frequency and CIC rate, so retain those terms here.
    coefficients = _filter_coefficients(
        int(p.loop_order), beta, PLL_SAMPLE_PERIOD_S * int(p.cic_rate))
    coeff_words = [_i32(value * Q30, f"SOS coefficient {index}")
                   for index, value in enumerate(coefficients)]
    return PLLRegisterValues(
        mode=p.mode.encode(),
        phase_increment_upon_reset=frequency_to_word(p.frequency_hz),
        phase_increment_minimum=frequency_to_word(p.minimum_frequency_hz),
        phase_increment_maximum=frequency_to_word(p.maximum_frequency_hz),
        decimator_control=int(p.cic_rate) + (int(p.cic_bit_shift) << 16),
        gain_proportional=_u32(gp, "gain_proportional"),
        gain_integral=_u32(gi, "gain_integral"),
        delay=_u32(p.pll_delay_turns * 2**32, "delay"),
        nco_gain=_u32(p.drive_amplitude_fraction * UINT32_MAX, "nco_gain"),
        feedback_delay=_u32(p.feedback_delay * 2**16, "feedback_delay"),
        feedback_gain=_u32(p.feedback_gain * 2**24, "feedback_gain"),
        sos_filter_0_b0=coeff_words[0], sos_filter_0_b1=coeff_words[1],
        sos_filter_0_b2=coeff_words[2], sos_filter_0_a1=coeff_words[3],
        sos_filter_0_a2=coeff_words[4], sos_filter_1_b0=coeff_words[5],
        sos_filter_1_b1=coeff_words[6], sos_filter_1_b2=coeff_words[7],
        sos_filter_1_a1=coeff_words[8], sos_filter_1_a2=coeff_words[9],
    )


# Addresses are inferred from the write order saved inside the supplied VI.
# They are deliberately a profile, not constants embedded in transport code.
DEFAULT_REGISTER_ADDRESSES: Mapping[str, int] = {
    "mode": 0x00,
    # LockInSweep's saved transcript repeatedly writes address 3 while
    # stepping frequency; the initial transcript writes addresses 1/2 as
    # the lower/upper bounds respectively.
    "phase_increment_upon_reset": 0x03,
    "phase_increment_minimum": 0x01,
    "phase_increment_maximum": 0x02,
    "decimator_control": 0x0E,
    "gain_proportional": 0x11,
    "gain_integral": 0x12,
    "delay": 0x05,
    "feedback_delay": 0x06,
    "feedback_gain": 0x14,
    "nco_gain": 0x13,
    "sos_filter_0_b0": 0x16, "sos_filter_0_b1": 0x17,
    "sos_filter_0_b2": 0x18, "sos_filter_0_a1": 0x19,
    "sos_filter_0_a2": 0x1A, "sos_filter_1_b0": 0x1B,
    "sos_filter_1_b1": 0x1C, "sos_filter_1_b2": 0x1D,
    "sos_filter_1_a1": 0x1E, "sos_filter_1_a2": 0x1F,
}


def register_write_map(values: PLLRegisterValues,
                       addresses: Mapping[str, int] = DEFAULT_REGISTER_ADDRESSES) -> dict[int, int]:
    words = values.as_dict(wire_unsigned=True)
    missing = set(words) - set(addresses)
    if missing:
        raise RegisterValueError(f"register address missing for {sorted(missing)}")
    return {int(addresses[name]): word for name, word in words.items()}
