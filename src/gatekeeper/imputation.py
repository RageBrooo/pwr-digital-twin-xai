"""
imputation.py -- replacement values for a sensor the Gatekeeper has isolated as lying.

  twin channel (exact)        flux_1<->flux_2, T_out_1<->T_out_2, level_1<->level_2   (if the twin is not suspect)
  model-based estimate        built from the REMAINING healthy channels and NOMINAL parameters only

    T_out_k  = T_in + P0*power / (W*cp)                 power from healthy flux, W from the flow channel
    flux_k   = W*cp*(T_out - T_in) / P0
    T_in     from the steam-generator relation  T_in = ((a/2)*T_out + b) / (1 - a/2),
             a = 2(1-e)/(2-e), b = e*T_sec/(2-e)
    T_fuel   = T_c + P0*power/hA                        (coarse: inherits the hA and flux-calibration mismatch)
    flow     = pump speed (nominal speed-to-flow gain; assumes the pump is healthy)
    pump_speed = the known speed command
    level_k  = 1 + beta_exp*(T_c - T_c_ref)             (assumes no leak)

Units follow the sensors: flux in fraction of rated power, flow and pump speed as fractions, temperatures in degC.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from src.simulator.params import PlantParams, equilibrium

TWINS = {"flux_1": "flux_2", "flux_2": "flux_1", "T_out_1": "T_out_2", "T_out_2": "T_out_1",
         "level_1": "level_2", "level_2": "level_1"}


def _healthy_mean(s: Dict[str, float], names: Iterable[str], suspects) -> Optional[float]:
    vals = [s[n] for n in names if n not in suspects]
    return float(np.mean(vals)) if vals else None


def model_estimate(sensor: str, s: Dict[str, float], model: PlantParams, suspects=(), pump_cmd: float = 1.0) -> float:
    p = model
    sus = set(suspects) | {sensor}
    W = (s["flow"] / p.flow_cal) * p.W0 if "flow" not in sus else p.W0 * (s["pump_speed"] if "pump_speed" not in sus else pump_cmd)
    flux = _healthy_mean(s, ("flux_1", "flux_2"), sus)
    pw = flux / 1.0 if flux is not None else None                       # nominal detector calibration = 1
    tout = _healthy_mean(s, ("T_out_1", "T_out_2"), sus)
    tin = s["T_in"] if "T_in" not in sus else None

    if sensor.startswith("T_out"):
        return float(tin + p.P0 * pw / (W * p.cp))
    if sensor.startswith("flux"):
        return float(W * p.cp * (tout - tin) / p.P0)
    if sensor == "T_in":
        a, b = 2 * (1 - p.sg_eff) / (2 - p.sg_eff), p.sg_eff * p.T_sec / (2 - p.sg_eff)
        return float((a / 2 * tout + b) / (1 - a / 2))
    if sensor == "T_fuel":
        pw_ = pw if pw is not None else W * p.cp * (tout - tin) / p.P0
        tc = 0.5 * (tin + tout)
        return float(tc + p.P0 * pw_ / p.hA)
    if sensor == "flow":
        return float(s["pump_speed"] if "pump_speed" not in sus else pump_cmd)
    if sensor == "pump_speed":
        return float(pump_cmd)
    if sensor.startswith("level"):
        tc = 0.5 * (tin + tout)
        return float(1.0 + p.beta_exp * (tc - equilibrium(p, 1.0)["Tc"]))
    raise KeyError(sensor)


def impute(sensor: str, s: Dict[str, float], model: PlantParams, suspects=(), pump_cmd: float = 1.0) -> float:
    """best available replacement for `sensor`; twin first, model estimate otherwise"""
    twin = TWINS.get(sensor)
    if twin is not None and twin not in set(suspects) and twin != sensor:
        return float(s[twin])
    return model_estimate(sensor, s, model, suspects, pump_cmd)
