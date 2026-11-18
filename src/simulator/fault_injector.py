"""
fault_injector.py -- fault specification and the logic that turns faults into plant / sensor effects.

Hardware faults act on the PLANT (inside the ODE right-hand side):
    coolant_leak      leak of `severity * LEAK_FULL_SCALE` [fraction of inventory / s], from onset
    pump_degradation  flow restriction ramping to `severity * PUMP_FULL_SCALE` over `pump_ramp` seconds
Sensor faults act on one MEASUREMENT channel only (plant untouched, except through the control loop):
    sensor_bias       constant offset from onset
    sensor_drift      linear ramp reaching `severity * full_scale` after `ramp_time`, and continuing
    sensor_stuck      channel frozen at its first post-onset sample (exactly zero variance afterwards)
`t_onset=None` means "at the moment of injection" (used by ReactorCore.inject for live faults).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .params import CLASSES, SENSORS, SENSOR_FAULTS

# severity 1.0 corresponds to this size, in sensor units
FAULT_FULL_SCALE: Dict[str, float] = {
    "flux_1": 0.10, "flux_2": 0.10, "T_in": 5.0, "T_out_1": 5.0, "T_out_2": 5.0, "T_fuel": 30.0,
    "flow": 0.10, "pump_speed": 0.10, "level_1": 0.10, "level_2": 0.10}
LEAK_FULL_SCALE = 4.0e-4      # fraction of inventory per second
PUMP_FULL_SCALE = 0.25        # flow restriction fraction


@dataclass(frozen=True)
class FaultSpec:
    kind: str = "normal"
    t_onset: Optional[float] = None
    severity: float = 0.5
    sensor: Optional[str] = None
    sign: int = 1
    ramp_time: float = 900.0
    pump_ramp: float = 60.0

    def __post_init__(self):
        if self.kind not in CLASSES:
            raise ValueError(f"unknown fault kind {self.kind!r}; choose from {CLASSES}")
        if self.kind in SENSOR_FAULTS and self.sensor not in SENSORS:
            raise ValueError(f"sensor fault needs sensor in {SENSORS}")
        if not 0.0 <= self.severity <= 1.5:
            raise ValueError("severity should be in [0, 1] (up to 1.5 allowed for stress tests)")

    def to_dict(self) -> dict:
        return dict(kind=self.kind, t_onset=self.t_onset, severity=self.severity, sensor=self.sensor,
                    sign=self.sign, ramp_time=self.ramp_time, pump_ramp=self.pump_ramp)

    @property
    def is_sensor_fault(self) -> bool:
        return self.kind in SENSOR_FAULTS


def hardware_state(faults: Sequence[FaultSpec], t: float) -> Tuple[float, float]:
    """(leak rate [1/s], pump flow restriction fraction) at time t"""
    leak = d_pump = 0.0
    for f in faults:
        if f.t_onset is None or t < f.t_onset:
            continue
        if f.kind == "coolant_leak":
            leak += f.severity * LEAK_FULL_SCALE
        elif f.kind == "pump_degradation":
            d_pump += f.severity * PUMP_FULL_SCALE * min(1.0, (t - f.t_onset) / f.pump_ramp)
    return leak, min(d_pump, 0.95)


def apply_sensor_faults(name: str, x: np.ndarray, t: np.ndarray, faults: Sequence[FaultSpec],
                        mem: dict) -> np.ndarray:
    """apply every active sensor fault that targets channel `name` to the (already noisy) samples x(t)"""
    for f in faults:
        if not f.is_sensor_fault or f.sensor != name or f.t_onset is None:
            continue
        active = t >= f.t_onset
        if not active.any():
            continue
        size = f.sign * f.severity * FAULT_FULL_SCALE[name]
        if f.kind == "sensor_bias":
            x = x + np.where(active, size, 0.0)
        elif f.kind == "sensor_drift":
            x = x + np.where(active, size * (t - f.t_onset) / f.ramp_time, 0.0)
        elif f.kind == "sensor_stuck":
            key = ("frozen", f.kind, f.sensor, f.t_onset)
            if key not in mem:
                mem[key] = float(x[np.argmax(active)])
            x = np.where(active, mem[key], x)
    return x


def active_label(faults: Sequence[FaultSpec], t: np.ndarray) -> np.ndarray:
    """label of the first fault that is active at each time ('normal' otherwise)"""
    lab = np.full(len(t), "normal", dtype=object)
    for f in reversed(list(faults)):
        if f.t_onset is not None:
            lab[t >= f.t_onset] = f.kind
    return lab.astype(str)
