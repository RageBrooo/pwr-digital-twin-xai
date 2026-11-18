"""
reactor_core.py -- the TRUE PLANT as a steppable digital twin.

Physics (11 states): n, C1..C6, T_fuel, T_cool, m (coolant inventory fraction), w (pump speed fraction)
  * six-group point kinetics (stiff, Lambda = 2e-5 s) integrated with scipy Radau
  * reactivity = rod (feedforward + PI trim) + Doppler(T_fuel) + moderator(T_cool) + inventory(m) + rod-step
  * fuel node, coolant node (thermal mass scales with m), quasi-static steam generator setting T_in
  * inventory node with capacity-limited charging; pump speed lag; flow = W0*w*(1-d_pump)*penalty(m)

Interface
  core = ReactorCore(true_params, sensor_cfg, sim_cfg, seed=...)
  core.reset(profile)              -> tick at t = 0
  core.step()                      -> tick for the next 0.1 s sample
  core.advance(n)                  -> DataFrame of the next n samples (much faster than n calls of step())
  core.inject(FaultSpec(...))      -> fault becomes active at the current time (live) or at its own t_onset
  core.clear_faults()
  core.command_demand(target, rate), core.command_rod_step(d_rho)   -> live operator actions
  core.simulate(duration, profile, faults) -> full run as DataFrame (batch use)

A TICK (what step() yields) is a dict:
  {"t": float,
   "sensors": {flux_1, flux_2, T_in, T_out_1, T_out_2, T_fuel, flow, pump_speed, level_1, level_2},
   "inputs":  {"demand": float},                      # the operator's load command (known to everybody)
   "truth":   {n, Tf, Tc, Tin, Tout, W, inventory, d_pump, leak_rate, rho_ext},   # NEVER for AI/gatekeeper
   "label":   "normal" | fault kind,                  # ground-truth class
   "faults":  [FaultSpec dicts currently injected]}
DataFrames use the same names flattened: sensor columns as-is, truth columns prefixed "true_".
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .fault_injector import FaultSpec, active_label, hardware_state
from .params import (BETA, BETA_I, LAMBDA_I, SENSORS, PlantParams, SensorConfig, SimConfig, equilibrium,
                     flow_penalty)
from .scenario import LoadProfile
from .sensors import TRUTH_PREFIX, SensorModel


class ReactorCore:
    def __init__(self, true_params: PlantParams, sensor_cfg: Optional[SensorConfig] = None,
                 cfg: Optional[SimConfig] = None, ctrl_params: Optional[PlantParams] = None, seed: int = 0):
        self.p = true_params
        self.cfg = cfg or SimConfig()
        self.sc = sensor_cfg or SensorConfig()
        ref = equilibrium(self.p, 1.0)                       # reactivity reference: rated-power equilibrium
        self.Tf_ref, self.Tc_ref = ref["Tf"], ref["Tc"]
        self.ctrl = ctrl_params or PlantParams.nominal()     # rod controller knows NOMINAL physics only
        cref = equilibrium(self.ctrl, 1.0)
        self._cTf_ref, self._cTc_ref = cref["Tf"], cref["Tc"]
        self.rng = np.random.default_rng(seed)
        self.sensors = SensorModel(self.p, self.sc, self.cfg.dt, self.rng, self.Tc_ref)
        self.spc = int(round(self.cfg.ctrl_dt / self.cfg.dt))
        self.faults: List[FaultSpec] = []
        self.profile: Optional[LoadProfile] = None
        self.k = 0

    # ---------------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self.k * self.cfg.dt

    def _rho_ff(self, n_d: float) -> float:
        """reactivity cancelling the expected Doppler + moderator defect at demanded power (nominal model)"""
        q, e = self.ctrl, equilibrium(self.ctrl, n_d)
        return -(q.alpha_D * (e["Tf"] - self._cTf_ref) + q.alpha_M * (e["Tc"] - self._cTc_ref))

    # ---- lifecycle ---------------------------------------------------------------------------
    def reset(self, profile: LoadProfile, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sensors.rng = self.rng
        self.profile, self.faults, self.k = profile, [], 0
        self.sensors.reset()
        p, cfg = self.p, self.cfg
        n0 = profile.demand(0.0)
        eq = equilibrium(p, n0)
        rho_need = -(p.alpha_D * (eq["Tf"] - self.Tf_ref) + p.alpha_M * (eq["Tc"] - self.Tc_ref))
        self.n_dem_f = n0
        self.rho_int = rho_need - self._rho_ff(n0)
        self.a_ff = float(np.exp(-cfg.ctrl_dt / cfg.tau_ff))
        self.Y = np.concatenate(([n0], BETA_I / (LAMBDA_I * p.Lambda) * n0, [eq["Tf"], eq["Tc"], 1.0, 1.0]))
        self.buf = deque(maxlen=self.spc)
        self._ctrl_k = self.spc
        self._controller_update(first=True)
        t0 = np.array([0.0])
        truth = self._truth(self.Y[None, :], t0)
        meas = self.sensors.measure(t0, truth, self.faults)
        self._push_buffer(truth, meas)
        self._row0 = self._frame(t0, truth, meas)
        return self._tick(self._row0.iloc[0])

    def inject(self, fault: FaultSpec) -> FaultSpec:
        if fault.kind == "normal":
            return fault
        if fault.t_onset is None:
            fault = replace(fault, t_onset=self.t)
        self.faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        self.faults = []
        self.sensors.clear_fault_memory()

    def command_demand(self, target: float, rate: float = 0.002) -> None:
        self.profile.retarget(self.t, float(np.clip(target, 0.3, 1.05)), rate)

    def command_rod_step(self, d_rho: float) -> None:
        """reactivity step, picked up at the next controller boundary"""
        nxt = (self._ctrl_k) * self.cfg.dt
        self.profile.add_rod_step(nxt, d_rho)

    # ---- controller ------------------------------------------------------------------------
    def _controller_update(self, first: bool = False) -> None:
        cfg, t0 = self.cfg, self.t
        if first:
            e = 0.0
        else:
            e = self.profile.demand(t0) - float(np.mean(self.buf))
            self.rho_int += cfg.Ki * e * cfg.ctrl_dt
        self.n_dem_f = self.a_ff * self.n_dem_f + (1 - self.a_ff) * self.profile.demand(t0)
        self.rho_ext = self._rho_ff(self.n_dem_f) + self.rho_int + cfg.Kp * e + self.profile.rho_dist(t0)

    def _push_buffer(self, truth, meas) -> None:
        if self.cfg.ctrl_flux_mode == "true":
            vals = truth["n"]
        else:
            vals = 0.5 * (meas["flux_1"] + meas["flux_2"])
        self.buf.extend(np.atleast_1d(vals))

    # ---- physics ---------------------------------------------------------------------------
    def _rhs(self, t, y, rho_ext):
        p = self.p
        n, C, Tf, Tc, m, w = y[0], y[1:7], y[7], y[8], y[9], y[10]
        leak, d_pump = hardware_state(self.faults, t)
        rho = (rho_ext + p.alpha_D * (Tf - self.Tf_ref) + p.alpha_M * (Tc - self.Tc_ref) + p.k_void * (m - 1.0))
        dn = (rho - BETA) / p.Lambda * n + LAMBDA_I @ C
        dC = BETA_I / p.Lambda * n - LAMBDA_I * C
        W = p.W0 * w * (1 - d_pump) * flow_penalty(m)
        Tin = (2 * (1 - p.sg_eff) * Tc + p.sg_eff * p.T_sec) / (2 - p.sg_eff)
        Tout = 2 * Tc - Tin
        Q_ht = p.hA * (Tf - Tc)
        Q_cool = W * p.cp * (Tout - Tin)
        dTf = (p.P0 * n - Q_ht) / p.Cf
        dTc = (Q_ht - Q_cool) / (p.Cc0 * m)
        dm = -leak + min(max(p.k_charge * (1.0 - m), 0.0), p.charge_max)
        dw = (1.0 - w) / p.tau_pump
        return np.concatenate(([dn], dC, [dTf, dTc, dm, dw]))

    def _truth(self, Y: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.p
        n, Tf, Tc, m, w = Y[:, 0], Y[:, 7], Y[:, 8], Y[:, 9], Y[:, 10]
        hw = np.array([hardware_state(self.faults, ti) for ti in t]).reshape(-1, 2)
        leak, d_pump = hw[:, 0], hw[:, 1]
        W = p.W0 * w * (1 - d_pump) * flow_penalty(m)
        Tin = (2 * (1 - p.sg_eff) * Tc + p.sg_eff * p.T_sec) / (2 - p.sg_eff)
        return dict(n=n, Tf=Tf, Tc=Tc, m=m, w=w, W=W, Tin=Tin, Tout=2 * Tc - Tin, d_pump=d_pump, leak=leak,
                    rho_ext=np.full(len(t), self.rho_ext))

    def _frame(self, t, truth, meas) -> pd.DataFrame:
        d = {"t": t}
        for s in SENSORS:
            d[s] = meas[s]
        d["demand"] = np.interp(t, self.profile.t_pts, self.profile.n_pts)
        for key, name in [("n", "n"), ("Tf", "Tf"), ("Tc", "Tc"), ("Tin", "Tin"), ("Tout", "Tout"), ("W", "W"),
                          ("m", "inventory"), ("d_pump", "d_pump"), ("leak", "leak_rate"),
                          ("rho_ext", "rho_ext")]:
            d[TRUTH_PREFIX + name] = truth[key]
        d["label"] = active_label(self.faults, t)
        sensor_faults = [f for f in self.faults if f.is_sensor_fault]
        d["fault_sensor"] = ["" if not sensor_faults else sensor_faults[0].sensor] * len(t)
        d["n_faults"] = [len(self.faults)] * len(t)
        return pd.DataFrame(d)

    def _tick(self, row: pd.Series) -> dict:
        return {
            "t": float(row["t"]),
            "sensors": {s: float(row[s]) for s in SENSORS},
            "inputs": {"demand": float(row["demand"])},
            "truth": {c[len(TRUTH_PREFIX):]: float(row[c]) for c in row.index if c.startswith(TRUTH_PREFIX)},
            "label": str(row["label"]),
            "faults": [f.to_dict() for f in self.faults],
        }

    # ---- time stepping ---------------------------------------------------------------------
    def advance(self, n_samples: int) -> pd.DataFrame:
        """advance n_samples * dt seconds; integration is split at controller boundaries"""
        cfg, chunks, remaining = self.cfg, [], int(n_samples)
        while remaining > 0:
            if self.k == self._ctrl_k:
                self._controller_update()
                self._ctrl_k += self.spc
            m = min(remaining, self._ctrl_k - self.k)
            t0 = self.t
            te = t0 + cfg.dt * np.arange(1, m + 1)
            sol = solve_ivp(self._rhs, (t0, te[-1]), self.Y, method=cfg.method, t_eval=te,
                            args=(self.rho_ext,), rtol=cfg.rtol, atol=cfg.atol)
            if not sol.success:
                raise RuntimeError(f"integration failed at t={t0:.1f}s: {sol.message}")
            Yseg = sol.y.T
            truth = self._truth(Yseg, te)
            meas = self.sensors.measure(te, truth, self.faults)
            self._push_buffer(truth, meas)
            chunks.append(self._frame(te, truth, meas))
            self.Y, self.k, remaining = Yseg[-1].copy(), self.k + m, remaining - m
        return pd.concat(chunks, ignore_index=True)

    def step(self) -> dict:
        return self._tick(self.advance(1).iloc[0])

    def simulate(self, duration: float, profile: LoadProfile, faults: Iterable[FaultSpec] = ()) -> pd.DataFrame:
        self.reset(profile)
        for f in faults:
            self.inject(f)
        body = self.advance(int(round(duration / self.cfg.dt)))
        return pd.concat([self._row0, body], ignore_index=True)
