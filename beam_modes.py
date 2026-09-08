"""Euler--Bernoulli modes for a uniform doubly-clamped hollow tube.

All inputs and outputs use SI units.  This deliberately keeps the ideal beam
model separate from the GUI: fluid loading, axial stress, and end-mass effects
can later be introduced as explicit correction models instead of silently
changing the baseline calculation.
"""
from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class TubeBeam:
    length_m: float
    outer_diameter_m: float
    inner_diameter_m: float
    youngs_modulus_pa: float
    density_kg_m3: float
    fluid_density_kg_m3: float = 0.0

    @property
    def area_m2(self):
        return pi / 4.0 * (self.outer_diameter_m**2 - self.inner_diameter_m**2)

    @property
    def second_moment_m4(self):
        return pi / 64.0 * (self.outer_diameter_m**4 - self.inner_diameter_m**4)

    @property
    def inner_area_m2(self):
        return pi / 4.0 * self.inner_diameter_m**2

    @property
    def line_mass_kg_m(self):
        """Tube plus fully entrained internal-fluid mass per unit length."""
        return self.density_kg_m3 * self.area_m2 + self.fluid_density_kg_m3 * self.inner_area_m2

    def validate(self):
        if self.length_m <= 0 or self.outer_diameter_m <= 0:
            raise ValueError("길이와 외경은 0보다 커야 합니다.")
        if not 0 <= self.inner_diameter_m < self.outer_diameter_m:
            raise ValueError("내경은 0 이상이고 외경보다 작아야 합니다.")
        if self.youngs_modulus_pa <= 0 or self.density_kg_m3 <= 0 or self.fluid_density_kg_m3 < 0:
            raise ValueError("Young률·관 밀도는 0보다 크고 유체 밀도는 0 이상이어야 합니다.")


@dataclass(frozen=True)
class BeamMode:
    index: int
    lambda_n: float
    frequency_hz: float


def clamped_clamped_root(index: int) -> float:
    """Return the positive root lambda_n of cosh(lambda) cos(lambda) = 1."""
    if index < 1:
        raise ValueError("모드 번호는 1 이상이어야 합니다.")

    def characteristic(x):
        return np.cosh(x) * np.cos(x) - 1.0

    # Every non-zero clamped-clamped root lies in this bracket.
    lo = (index + 0.25) * pi
    hi = (index + 0.75) * pi
    return float(brentq(characteristic, lo, hi))


def calculate_modes(beam: TubeBeam, count: int):
    beam.validate()
    if count < 1:
        return []
    stiffness_factor = sqrt(
        beam.youngs_modulus_pa * beam.second_moment_m4 / beam.line_mass_kg_m
    ) / (2.0 * pi * beam.length_m**2)
    modes = []
    for n in range(1, count + 1):
        lam = clamped_clamped_root(n)
        modes.append(BeamMode(n, lam, lam**2 * stiffness_factor))
    return modes


def mode_shape(beam: TubeBeam, mode: BeamMode, x_m):
    """Normalized transverse mode shape at x, with x in [0, L]."""
    beam.validate()
    x = np.asarray(x_m, dtype=float)
    xi = x / beam.length_m
    lam = mode.lambda_n
    sigma = (np.cosh(lam) - np.cos(lam)) / (np.sinh(lam) - np.sin(lam))
    shape = np.cosh(lam * xi) - np.cos(lam * xi) - sigma * (
        np.sinh(lam * xi) - np.sin(lam * xi)
    )
    maximum = np.max(np.abs(shape))
    return shape / maximum if maximum else shape
