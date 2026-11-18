"""
decision.py -- the Thermodynamic Gatekeeper's adjudicator.

It reads CALIBRATED residuals (calibration.py) and raw measurements, never ground truth, and decides:
   normal | coolant_leak | pump_degradation | sensor fault (which sensor(s), and bias / drift / stuck)

How it decides
 1. hysteresis  : a residual is ARMED after `persist` consecutive samples above its epsilon and is released only after
                  `release_k` consecutive samples below 0.5*epsilon, so marginal faults do not flicker.
 2. flatline    : a sensor whose last `flat_k` readings are IDENTICAL is stuck (any sensor, no physics needed).
 3. slow energy : "Es" = `slow_n`-sample mean of the four global energy residuals. Averaging removes noise, so it sees
                  small flow / T_in errors that no single E residual can. Pump degradation does NOT move Es
                  (the measured flow is the true flow); a flow-sensor error does. This is what separates the two.
                  Because Es needs `slow_n` samples, a lone H_flow trip is only called pump degradation after
                  `confirm_s` seconds ("confirming" in between). Disabled when eps has no "Es" entry.
 4. isolation   : structured-residual logic. Every candidate (10 sensors + leak + pump degradation) has
                    TRIPPABLE  the residuals that may move when it is the fault, and
                    REQUIRES   groups of residuals of which at least one must move for the fault to be detected.
                  Consistent = (tripped set) inside TRIPPABLE and every REQUIRES group hit.
                  One consistent candidate = isolated; several = ambiguous (a set); none = unknown.
 5. typing      : bias (step) vs drift (ramp) from the rise time of the dominant residual (last quiet sample <0.3 eps ->
                  first persistent trip). The type is REMEMBERED for `memory_s` seconds so a diagnosis that is dropped
                  and re-declared keeps its type.
 6. latching    : a diagnosis is held until nothing is armed for `hold` samples; candidates are refined, never widened.
 7. repair      : once a single sensor is isolated, imputation.py gives a replacement value.

The AI is NOT consulted here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulator.params import SENSORS, PlantParams

from .imputation import impute
from .residuals import RESIDUALS, SUPPORT

HARDWARE = ("coolant_leak", "pump_degradation")
E_ALL, C_ALL = ("E11", "E12", "E21", "E22"), ("C1", "C2")
SLOW = "Es"
DIRECT = ("D_flux", "D_tout", "L12", "S_pump", "H_flow")        # direct comparisons: best for step / ramp typing

TRIPPABLE: Dict[str, FrozenSet[str]] = {s: frozenset(r for r in RESIDUALS if s in SUPPORT[r]) for s in SENSORS}
TRIPPABLE["T_fuel"] = frozenset({"F1", "F2", "C1", "C2"})        # a static T_fuel error does not move the global balances
for _s in ("flux_1", "flux_2", "T_out_1", "T_out_2", "T_in", "flow"):   # sensors that enter the global balance
    TRIPPABLE[_s] = TRIPPABLE[_s] | {SLOW}
TRIPPABLE["coolant_leak"] = frozenset({"L1", "L2", "H_flow"})    # H_flow only through the low-inventory flow penalty
TRIPPABLE["pump_degradation"] = frozenset({"H_flow"})            # Es must stay quiet for a real pump fault

_E = frozenset(E_ALL) | {SLOW}   # a T_in / flow error shifts the GLOBAL balance (fast E or slow Es) as much as the coolant node
REQUIRES: Dict[str, List[FrozenSet[str]]] = {
    "flux_1": [frozenset({"D_flux", "E11", "E12", "F1"})],
    "flux_2": [frozenset({"D_flux", "E21", "E22", "F2"})],
    "T_out_1": [frozenset({"D_tout", "E11", "E21", "C1"})],
    "T_out_2": [frozenset({"D_tout", "E12", "E22", "C2"})],
    "T_in": [_E],
    "T_fuel": [frozenset({"F1", "F2", "C1", "C2"})],
    "flow": [_E],                                                 # H_flow alone is pump degradation, not a flow sensor
    "pump_speed": [frozenset({"S_pump"})],
    "level_1": [frozenset({"L12", "L1"})],
    "level_2": [frozenset({"L12", "L2"})],
    "coolant_leak": [frozenset({"L1"}), frozenset({"L2"})],
    "pump_degradation": [frozenset({"H_flow"})],
}


def consistent_candidates(tripped: Iterable[str]) -> List[str]:
    T = set(tripped)
    if not T:
        return []
    return [c for c in list(SENSORS) + list(HARDWARE)
            if T <= TRIPPABLE[c] and all(T & g for g in REQUIRES[c])]


@dataclass
class Verdict:
    t: float
    cls: str                          # normal | coolant_leak | pump_degradation | sensor_bias | sensor_drift | sensor_stuck | unknown
    candidates: Tuple[str, ...] = ()  # isolated sensor(s) or hardware fault
    kind: str = ""                    # bias | drift | stuck for sensor faults
    tripped: Tuple[str, ...] = ()
    detected: bool = False
    isolated: bool = False
    clean: Dict[str, float] = field(default_factory=dict)   # replacement values (only for an isolated sensor)
    reason: str = ""


class Gatekeeper:
    def __init__(self, model: PlantParams, eps: Dict[str, float], persist: int = 5, release_k: int = 5,
                 flat_k: int = 10, hold: int = 60, rise_max: float = 5.0, pump_cmd: float = 1.0,
                 slow_n: int = 60, confirm_s: float = 60.0, memory_s: float = 900.0):
        self.model, self.eps = model, eps
        self.persist, self.release_k, self.flat_k, self.hold = persist, release_k, flat_k, hold
        self.rise_max, self.pump_cmd, self.slow_n, self.confirm_s, self.memory_s = rise_max, pump_cmd, slow_n, confirm_s, memory_s
        self.use_slow = SLOW in eps
        self.names = list(RESIDUALS) + ([SLOW] if self.use_slow else [])
        self.reset()

    def reset(self):
        n = self.names
        self.hi = {r: 0 for r in n}
        self.lo = {r: 0 for r in n}
        self.cand_start: Dict[str, Optional[float]] = {r: None for r in n}
        self.armed = {r: False for r in n}
        self.start: Dict[str, Optional[float]] = {r: None for r in n}
        self.hist: Dict[str, Deque[Tuple[float, float]]] = {r: deque(maxlen=1500) for r in n}
        self.recent: Dict[str, Deque[float]] = {s: deque(maxlen=self.flat_k) for s in SENSORS}
        self.slow_buf: Deque[float] = deque(maxlen=self.slow_n)
        self.active: Optional[dict] = None
        self.memory: Optional[dict] = None
        self.quiet = 0
        self.last_ratio: Dict[str, float] = {r: 0.0 for r in n}

    # -------------------------------------------------------------------------------------------------------
    def _typing(self, T: List[str]) -> str:
        pool = [r for r in T if r in DIRECT] or list(T)
        dom = max(pool, key=lambda r: self.last_ratio[r])
        t_cross = self.start.get(dom)
        if t_cross is None:
            return "drift"
        for tt, rr in reversed(self.hist[dom]):
            if tt < t_cross and rr < 0.3:
                return "bias" if (t_cross - tt) <= self.rise_max else "drift"
        return "drift"                                              # no quiet reference before the trip

    def _slow_value(self, residuals: Dict[str, float]) -> float:
        e = [residuals[k] for k in E_ALL if residuals[k] is not None and np.isfinite(residuals[k])]
        if e:
            self.slow_buf.append(float(np.mean(e)))
        return float(np.mean(self.slow_buf)) if len(self.slow_buf) == self.slow_n else 0.0

    def _track(self, name: str, t: float, value) -> None:
        ratio = 0.0 if (value is None or not np.isfinite(value)) else abs(value) / self.eps[name]
        self.last_ratio[name] = ratio
        self.hist[name].append((t, ratio))
        if ratio > 1.0:
            if self.hi[name] == 0:
                self.cand_start[name] = t
            self.hi[name] += 1
            self.lo[name] = 0
            if not self.armed[name] and self.hi[name] >= self.persist:
                self.armed[name], self.start[name] = True, self.cand_start[name]
        elif ratio < 0.5:
            self.hi[name] = 0
            self.cand_start[name] = None
            self.lo[name] += 1
            if self.armed[name] and self.lo[name] >= self.release_k:
                self.armed[name], self.start[name] = False, None
        else:                                                       # hysteresis band: breaks a rising streak, keeps state
            self.hi[name] = 0
            self.cand_start[name] = None
            self.lo[name] = 0

    def update(self, t: float, sensors: Dict[str, float], residuals: Dict[str, float]) -> Verdict:
        for s in SENSORS:
            self.recent[s].append(sensors[s])
        for r in RESIDUALS:
            self._track(r, t, residuals[r])
        if self.use_slow:
            self._track(SLOW, t, self._slow_value(residuals))
        T = [r for r in self.names if self.armed[r]]
        flat = [s for s in SENSORS if len(self.recent[s]) == self.flat_k and max(self.recent[s]) == min(self.recent[s])]

        if flat:                                                    # flatline: certain, needs no physics
            clean = {}
            if len(flat) == 1:
                clean = {flat[0]: impute(flat[0], sensors, self.model, suspects=flat, pump_cmd=self.pump_cmd)}
            self.active = {"family": "stuck", "cands": frozenset(flat), "kind": "stuck"}
            return Verdict(t, "sensor_stuck", tuple(flat), "stuck", tuple(T), True, len(flat) == 1, clean,
                           "identical readings for %d samples" % self.flat_k)

        cands = frozenset(consistent_candidates(T))
        if cands == {"pump_degradation"} and self.use_slow:        # wait for the slow residual before blaming the pump
            t0 = self.start.get("H_flow")
            if t0 is not None and (t - t0) < self.confirm_s:
                self.quiet = 0
                return Verdict(t, "unknown", ("flow", "pump_degradation"), "", tuple(T), True, False, {},
                               "confirming: pump fault vs flow-sensor fault")
        if cands:                                                   # isolation
            self.quiet = 0
            if self.active and self.active["family"] == "phys" and (self.active["cands"] & cands):
                cands = self.active["cands"] & cands                # refine, never widen
                kind = self.active["kind"]
            else:
                if cands <= set(HARDWARE):
                    kind = ""
                elif self.memory and (self.memory["cands"] & cands) and (t - self.memory["t"]) < self.memory_s:
                    kind = self.memory["kind"]                      # re-declared diagnosis keeps its type
                else:
                    kind = self._typing(T)
            self.active = {"family": "phys", "cands": cands, "kind": kind}
            if kind:
                self.memory = {"cands": cands, "kind": kind, "t": t}
            return self._verdict(t, sensors, T, cands, kind)
        if T:                                                       # tripped residuals no single fault explains
            self.quiet = 0
            return Verdict(t, "unknown", (), "", tuple(T), True, False, {}, "tripped set matches no single fault")
        if self.active and self.active["family"] == "phys":         # latching
            self.quiet += 1
            if self.quiet <= self.hold:
                return self._verdict(t, sensors, T, self.active["cands"], self.active["kind"], held=True)
            self.active = None
        return Verdict(t, "normal")

    def _verdict(self, t, sensors, T, cands, kind, held=False) -> Verdict:
        cs = tuple(sorted(cands))
        if all(c in HARDWARE for c in cs) and len(cs) == 1:
            return Verdict(t, cs[0], cs, "", tuple(T), True, True, {}, "hardware signature" + (" (held)" if held else ""))
        if all(c in SENSORS for c in cs):
            clean = {}
            if len(cs) == 1:
                clean = {cs[0]: impute(cs[0], sensors, self.model, suspects=cs, pump_cmd=self.pump_cmd)}
            return Verdict(t, f"sensor_{kind}", cs, kind, tuple(T), True, len(cs) == 1, clean,
                           "sensor signature" + (" (held)" if held else ""))
        return Verdict(t, "unknown", cs, "", tuple(T), True, False, {}, "hardware / sensor ambiguity")


def process_run(df: pd.DataFrame, model: PlantParams, eps: Dict[str, float], **kw) -> pd.DataFrame:
    """df: one run, 1 Hz rows with t, SENSORS and CALIBRATED residual columns -> per-row verdicts"""
    g = Gatekeeper(model, eps, **kw)
    rows = []
    s_recs = df[SENSORS].to_dict("records")
    r_recs = df[RESIDUALS].to_dict("records")
    for i, t in enumerate(df.t.to_numpy()):
        v = g.update(float(t), s_recs[i], r_recs[i])
        rows.append((v.cls, ",".join(v.candidates), v.kind, v.detected, v.isolated, ",".join(v.tripped)))
    out = pd.DataFrame(rows, columns=["gate_cls", "gate_cands", "gate_kind", "gate_detected", "gate_isolated", "gate_tripped"])
    out.insert(0, "t", df.t.to_numpy())
    return out
