"""
residuals.py -- analytical-redundancy residual bank for the Thermodynamic Gatekeeper.

Input : measured channels only (time + SENSORS columns) and the NOMINAL plant parameters.
Output: 15 residuals per sample; each one has a declared SENSOR SUPPORT (the measured channels it reads).

Why 15 and not 4: the nodal energy balances are linearly dependent (coolant node + fuel node = global balance),
so four "named" residuals carry only two independent pieces of information. Isolation needs residuals built
from DIFFERENT SENSOR SUBSETS, so a fault in one channel trips a distinctive subset of them:

  E_ij    global energy balance, flux channel i and T_out channel j          (4)  fraction of rated power
  F_i     fuel-node balance, flux channel i                                  (2)  fraction of rated power
  C_j     coolant-node balance, T_out channel j                              (2)  fraction of rated power
  H_flow  measured flow vs pump speed                                        (1)  fraction of rated flow
  S_pump  pump speed vs the (known) speed command                            (1)  fraction
  L12     level_1 - level_2                                                  (1)  inventory fraction
  L1, L2  level_k vs the level expected from coolant temperature             (2)  inventory fraction
  D_flux  flux_1 - flux_2  (calibration-corrected)                           (1)  fraction of rated power
  D_tout  T_out_1 - T_out_2                                                  (1)  K

Both heat-storage terms (coolant AND fuel) are included. Storage derivatives use a centred least-squares slope
over +-half_window samples; to stay causal EVERY residual is delayed by half_window samples (2.5 s at 10 Hz).
Row k therefore only uses measurements from rows <= k, so `latest_residuals` on a trailing buffer gives exactly
the same numbers as `compute_residuals` on the whole run (live dashboard == offline evaluation).
The first 2*half_window rows are NaN.

The Gatekeeper must be built from NOMINAL parameters only.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

from src.simulator.params import SENSORS, PlantParams, equilibrium

HALF_WINDOW = 25                                      # samples (2.5 s at 10 Hz)

_E = {f"E{i}{j}": frozenset({f"flux_{i}", f"T_out_{j}", "T_in", "flow", "T_fuel"}) for i in (1, 2) for j in (1, 2)}
_F = {f"F{i}": frozenset({f"flux_{i}", "T_fuel", "T_in", "T_out_1", "T_out_2"}) for i in (1, 2)}
_C = {f"C{j}": frozenset({"T_fuel", "T_in", f"T_out_{j}", "flow"}) for j in (1, 2)}
_OTHER = {
    "H_flow": frozenset({"flow", "pump_speed"}),
    "S_pump": frozenset({"pump_speed"}),
    "L12": frozenset({"level_1", "level_2"}),
    "L1": frozenset({"level_1", "T_in", "T_out_1", "T_out_2"}),
    "L2": frozenset({"level_2", "T_in", "T_out_1", "T_out_2"}),
    "D_flux": frozenset({"flux_1", "flux_2"}),
    "D_tout": frozenset({"T_out_1", "T_out_2"}),
}
SUPPORT: Dict[str, FrozenSet[str]] = {**_E, **_F, **_C, **_OTHER}
RESIDUALS: List[str] = list(SUPPORT)
UNITS = {r: ("K" if r == "D_tout" else "frac. of P0" if r[0] in "EFC" else "fraction") for r in RESIDUALS}
MIN_SAMPLES = 2 * HALF_WINDOW + 1                      # buffer length needed for one valid residual row


def _slope(x: np.ndarray, h: int, dt: float) -> np.ndarray:
    i = np.arange(-h, h + 1)
    w = i / (i ** 2).sum() / dt
    s = np.convolve(x, w[::-1], mode="same")
    s[:h] = np.nan
    s[-h:] = np.nan
    return s


def _delay(x: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[h:] = x[:len(x) - h] if h > 0 else x
    return out


def compute_residuals(sensors: pd.DataFrame, model: PlantParams, pump_cmd: float = 1.0,
                      half_window: int = HALF_WINDOW) -> pd.DataFrame:
    """residual bank for a block of measured samples (columns: t + SENSORS); returns t + RESIDUALS"""
    h, p = half_window, model
    t = sensors["t"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    s = {k: sensors[k].to_numpy(float) for k in SENSORS}

    W = s["flow"] / p.flow_cal * p.W0                                    # kg/s
    pw = {1: s["flux_1"] / p.flux_cal_1, 2: s["flux_2"] / p.flux_cal_2}  # relative power per channel
    Tc = {j: 0.5 * (s["T_in"] + s[f"T_out_{j}"]) for j in (1, 2)}
    Tc_m = 0.5 * (Tc[1] + Tc[2])
    Q = {j: W * p.cp * (s[f"T_out_{j}"] - s["T_in"]) for j in (1, 2)}
    dTc = {j: _slope(Tc[j], h, dt) for j in (1, 2)}
    dTf = _slope(s["T_fuel"], h, dt)
    Tc_ref = equilibrium(p, 1.0)["Tc"]

    r: Dict[str, np.ndarray] = {}
    for i in (1, 2):
        for j in (1, 2):
            r[f"E{i}{j}"] = (p.P0 * pw[i] - Q[j] - p.Cc0 * dTc[j] - p.Cf * dTf) / p.P0
    for i in (1, 2):
        r[f"F{i}"] = (p.P0 * pw[i] - p.Cf * dTf - p.hA * (s["T_fuel"] - Tc_m)) / p.P0
    for j in (1, 2):
        r[f"C{j}"] = (p.hA * (s["T_fuel"] - Tc[j]) - Q[j] - p.Cc0 * dTc[j]) / p.P0
    r["H_flow"] = s["flow"] / p.flow_cal - s["pump_speed"]
    r["S_pump"] = s["pump_speed"] - pump_cmd
    r["L12"] = s["level_1"] - s["level_2"]
    expected = 1.0 + p.beta_exp * (Tc_m - Tc_ref)
    r["L1"] = s["level_1"] - p.level_off_1 - expected
    r["L2"] = s["level_2"] - p.level_off_2 - expected
    r["D_flux"] = pw[1] - pw[2]
    r["D_tout"] = s["T_out_1"] - s["T_out_2"]

    out = pd.DataFrame({"t": t})
    for name in RESIDUALS:
        out[name] = _delay(r[name], h)
    return out


def latest_residuals(buffer: pd.DataFrame, model: PlantParams, pump_cmd: float = 1.0,
                     half_window: int = HALF_WINDOW) -> Dict[str, float]:
    """residuals for the newest sample of a trailing buffer (needs >= MIN_SAMPLES rows); live-dashboard use"""
    need = 2 * half_window + 1
    if len(buffer) < need:
        return {name: float("nan") for name in RESIDUALS}
    row = compute_residuals(buffer.iloc[-need:], model, pump_cmd, half_window).iloc[-1]
    return {name: float(row[name]) for name in RESIDUALS}


# ---- structural analysis (needs no simulation) -------------------------------------------------------------
def structural_matrix() -> pd.DataFrame:
    """rows = sensors, columns = residuals; True where the sensor can influence the residual"""
    return pd.DataFrame({r: [s in SUPPORT[r] for s in SENSORS] for r in RESIDUALS}, index=SENSORS)


def structurally_non_isolable_pairs() -> List[Tuple[str, str]]:
    """sensor pairs with identical residual patterns: a fault in one cannot be told from the other"""
    m = structural_matrix()
    return [(a, b) for a, b in combinations(SENSORS, 2) if (m.loc[a] == m.loc[b]).all()]
