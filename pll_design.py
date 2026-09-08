"""Transparent model-based PI/PID design for the HF2 PLL frequency tracker."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize


@dataclass(frozen=True)
class PLLDesign:
    p_hz_per_deg: float
    i_hz2_per_deg: float
    d_per_deg: float
    achieved_bandwidth_hz: float
    phase_margin_deg: float
    gain_margin_db: float
    sensitivity_peak: float
    stable: bool
    message: str


def _metrics(p, i, d, phase_slope_deg_per_hz, f0_hz, q, demod_tc_s,
             demod_order, loop_rate_hz):
    frequencies = np.logspace(-3, np.log10(max(1.0, min(5e4, loop_rate_hz * 0.35))), 1800)
    w = 2.0 * np.pi * frequencies
    tau_res = q / (np.pi * f0_hz)  # amplitude/phase envelope ring-down time
    delay = 1.0 / max(loop_rate_hz, 1.0)
    plant = phase_slope_deg_per_hz / ((1.0 + 1j * w * tau_res) *
                                      (1.0 + 1j * w * demod_tc_s) ** demod_order) * np.exp(-1j * w * delay)
    controller = p + i / (1j * w) + d * (1j * w)
    loop = controller * plant
    closed = loop / (1.0 + loop)
    sensitivity = 1.0 / (1.0 + loop)
    magnitude = np.abs(loop)
    crossing = np.where(np.diff(np.sign(magnitude - 1.0)) != 0)[0]
    if len(crossing):
        k = crossing[0]
        frac = (1.0 - magnitude[k]) / max(magnitude[k + 1] - magnitude[k], 1e-15)
        wc = frequencies[k] + frac * (frequencies[k + 1] - frequencies[k])
        phase = np.unwrap(np.angle(loop))[k]
        pm = 180.0 + np.degrees(phase)
    else:
        wc, pm = 0.0, -180.0
    bw_idx = np.where(np.abs(closed) <= 1.0 / np.sqrt(2.0))[0]
    bw = frequencies[bw_idx[0]] if len(bw_idx) else frequencies[-1]
    phase_cross = np.where(np.diff(np.sign(np.unwrap(np.angle(loop)) + np.pi)) != 0)[0]
    gm_db = float("nan")
    if len(phase_cross):
        gm_db = float(-20.0 * np.log10(max(magnitude[phase_cross[0]], 1e-30)))
    return float(bw), float(pm), gm_db, float(np.max(np.abs(sensitivity))), bool(np.all(np.isfinite(loop)))


def design_pll(*, phase_slope_deg_per_hz, f0_hz, q, demod_tc_s,
               demod_order, loop_rate_hz, target_bandwidth_hz,
               minimum_phase_margin_deg=60.0, allow_integral=False,
               allow_derivative=False):
    if abs(phase_slope_deg_per_hz) < 1e-12 or min(f0_hz, q, demod_tc_s, loop_rate_hz, target_bandwidth_hz) <= 0:
        raise ValueError("유효한 공진기 및 PLL 모델 파라미터가 필요합니다.")
    sign = 1.0 if phase_slope_deg_per_hz > 0 else -1.0
    # Begin with P-only locking.  Integral action is optional because it
    # converts tiny detector offsets/noise into a continuously accumulating
    # frequency correction on a high-Q resonator.
    def unpack(v):
        p = sign * 10.0 ** v[0]
        next_index = 1
        if allow_integral:
            ti = 10.0 ** v[next_index]
            i = p / ti
            next_index += 1
        else:
            i = 0.0
        if allow_derivative:
            td = 10.0 ** v[next_index]
            d = p * td
        else:
            d = 0.0
        return p, i, d

    target_cap = min(5e4, loop_rate_hz * 0.15)
    target = min(target_bandwidth_hz, target_cap)
    def objective(v):
        p, i, d = unpack(v)
        bw, pm, gm, ms, finite = _metrics(p, i, d, phase_slope_deg_per_hz, f0_hz, q,
                                           demod_tc_s, demod_order, loop_rate_hz)
        if not finite or bw <= 0:
            return 1e9
        cost = np.log(max(bw, 1e-6) / target) ** 2
        cost += 0.05 * max(0.0, minimum_phase_margin_deg - pm) ** 2
        cost += 4.0 * max(0.0, ms - 2.0) ** 2
        if np.isfinite(gm):
            cost += 0.02 * max(0.0, 6.0 - gm) ** 2
        return float(cost)

    # If I is explicitly requested, bound Ti to be slow compared with the
    # requested loop bandwidth.  This avoids the formerly unconstrained,
    # very large I solutions.
    bounds = [(-8, 8)]
    if allow_integral:
        bounds.append((np.log10(5.0 / target), np.log10(200.0 / target)))
    if allow_derivative:
        bounds.append((-8, 1))
    global_result = differential_evolution(objective, bounds, seed=42, maxiter=40,
                                           popsize=8, polish=False, updating="immediate")
    local = minimize(objective, global_result.x, method="Nelder-Mead", options={"maxiter": 800})
    p, i, d = unpack(local.x)
    bw, pm, gm, ms, finite = _metrics(p, i, d, phase_slope_deg_per_hz, f0_hz, q,
                                       demod_tc_s, demod_order, loop_rate_hz)
    stable = finite and pm >= minimum_phase_margin_deg and ms <= 2.5
    message = ("설계 완료" if stable else
               "현재 TC/order/PID 선택으로 목표 BW를 만족하는 안정 설계를 찾지 못했습니다. "
               "계수를 적용하지 말고 TC, filter order 또는 목표 BW를 조정하세요.")
    return PLLDesign(p, i, d, bw, pm, gm, ms, stable, message)
