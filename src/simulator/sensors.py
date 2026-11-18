"""
sensors.py -- the measurement layer between the true plant and everything downstream.

    true plant state --> calibration (true plant's own errors) --> optional RTD lag
                     --> white Gaussian noise --> sensor faults (bias / drift / stuck) --> measured value

Noise is drawn as one (n_samples x n_sensors) block per call, row-major, so the random stream is identical
no matter how a run is chunked into advance() calls.

`sensor_view(df)` is the ONLY door through which the AI engine and the Gatekeeper should read simulator
output: it returns time + the SENSORS columns and nothing that starts with "true_".
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .fault_injector import FaultSpec, apply_sensor_faults
from .params import SENSORS, PlantParams, SensorConfig

TRUTH_PREFIX = "true_"


def sensor_view(df: pd.DataFrame) -> pd.DataFrame:
    """what a real plant would give you: time and measured channels only"""
    return df[["t"] + SENSORS].copy()


class SensorModel:
    def __init__(self, params: PlantParams, cfg: SensorConfig, dt: float, rng: np.random.Generator,
                 Tc_ref: float):
        self.p, self.cfg, self.dt, self.rng, self.Tc_ref = params, cfg, dt, rng, Tc_ref
        self.sigma = np.array([cfg.sigma[s] for s in SENSORS]) * cfg.noise_scale
        self.mem: dict = {"lag": {}}

    def reset(self):
        self.mem = {"lag": {}}

    def clear_fault_memory(self):
        self.mem = {"lag": self.mem["lag"]}                  # keep filter state, forget frozen values

    def physical(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """noise-free reading of every channel, including this plant's calibration errors"""
        p = self.p
        expansion = p.beta_exp * (truth["Tc"] - self.Tc_ref)
        return {
            "flux_1": p.flux_cal_1 * truth["n"], "flux_2": p.flux_cal_2 * truth["n"],
            "T_in": truth["Tin"], "T_out_1": truth["Tout"], "T_out_2": truth["Tout"],
            "T_fuel": truth["Tf"], "flow": p.flow_cal * truth["W"] / p.W0, "pump_speed": truth["w"],
            "level_1": truth["m"] + expansion + p.level_off_1,
            "level_2": truth["m"] + expansion + p.level_off_2,
        }

    def measure(self, t: np.ndarray, truth: Dict[str, np.ndarray],
                faults: Sequence[FaultSpec]) -> Dict[str, np.ndarray]:
        raw = self.physical(truth)
        noise = self.rng.normal(0.0, 1.0, size=(len(t), len(SENSORS))) * self.sigma
        out = {}
        for i, name in enumerate(SENSORS):
            x = np.array(raw[name], dtype=float)
            if self.cfg.tau_T > 0 and name in ("T_in", "T_out_1", "T_out_2"):
                a = np.exp(-self.dt / self.cfg.tau_T)
                prev = self.mem["lag"].get(name, x[0])
                for j in range(len(x)):
                    prev = a * prev + (1 - a) * x[j]
                    x[j] = prev
                self.mem["lag"][name] = prev
            x = x + noise[:, i]
            out[name] = apply_sensor_faults(name, x, t, faults, self.mem)
        return out
