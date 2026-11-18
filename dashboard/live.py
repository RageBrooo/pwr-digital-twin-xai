"""
live.py -- the live digital twin behind the dashboard (no Streamlit in here, so it can be tested headless).

  steppable simulator (true plant)  ->  10 Hz sensors  ->  causal residual bank  ->  per-run baseline calibration
                                    ->  Gatekeeper (physics verdict + replacement value)
                                    ->  1 Hz sliding windows -> statistical AI-B (raw) and AI-C (calibrated) [+ SHAP]

The first CAL_SECONDS (270 s) are a healthy commissioning run used to calibrate the residual baseline, exactly as in the
offline evaluation; faults can only be injected after that. The Gatekeeper and the AI only ever receive MEASURED
sensors; the simulator's ground truth is stored separately for display.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_engine.features import CAL_WINDOW, extract_features                                        # noqa: E402
from src.gatekeeper.calibration import Baseline                                                         # noqa: E402
from src.gatekeeper.decision import Gatekeeper                                                          # noqa: E402
from src.gatekeeper.residuals import RESIDUALS, compute_residuals, latest_residuals, MIN_SAMPLES        # noqa: E402
from src.simulator import (SENSORS, FaultSpec, LoadProfile, PlantParams, ReactorCore, SensorConfig,      # noqa: E402
                           load_sim_config, make_plant_pair, sensor_view)

CAL_SECONDS = 270
CLASSES = ["normal", "coolant_leak", "pump_degradation", "sensor_bias", "sensor_drift", "sensor_stuck", "unknown"]
REQUIRED = ["models/rf_B_full.joblib", "models/rf_C_full_cal.joblib", "models/feature_names.joblib",
            "models/feature_names_C.joblib", "configs/epsilon.yaml"]


def missing_assets(root: Path = ROOT) -> List[str]:
    return [p for p in REQUIRED if not (Path(root) / p).exists()]


def load_assets(root: Path = ROOT) -> dict:
    root = Path(root)
    a = {"eps_cfg": yaml.safe_load((root / "configs" / "epsilon.yaml").read_text()),
         "FN": joblib.load(root / "models" / "feature_names.joblib"),
         "FNC": joblib.load(root / "models" / "feature_names_C.joblib"),
         "B": joblib.load(root / "models" / "rf_B_full.joblib"),
         "C": joblib.load(root / "models" / "rf_C_full_cal.joblib")}
    for k in ("B", "C"):
        a[k].n_jobs = 1                                  # single-row predictions: threading overhead only hurts
    return a


class LiveTwin:
    def __init__(self, assets: dict, mismatch: float = 0.03, noise_scale: float = 1.0, seed: int = 1,
                 demand: float = 0.95, history: int = 1800):
        self.assets = assets
        self.eps = assets["eps_cfg"]["epsilon"]
        self.model, true = make_plant_pair(mismatch, seed, nominal=PlantParams.nominal())
        sim_cfg, sens_cfg, _ = load_sim_config()
        sc = SensorConfig(sigma=sens_cfg.sigma, noise_scale=noise_scale, tau_T=sens_cfg.tau_T)
        self.core = ReactorCore(true, sc, sim_cfg, ctrl_params=self.model, seed=seed + 1)
        self.core.reset(LoadProfile(np.array([0.0, 1e7]), np.array([demand, demand])))
        self.fault_times: List[dict] = []
        self.hist: deque = deque(maxlen=history)
        self._explainer = None

        # ---- healthy commissioning run: calibrate the residual baseline and the calibrated-AI offsets
        block = self.core.advance(CAL_SECONDS * 10)
        res = compute_residuals(sensor_view(block), self.model)
        one = block.iloc[9::10].reset_index(drop=True)
        cal = one[["t", "flux_1", "flux_2"]].copy()
        for c in RESIDUALS:
            cal[c] = res[c].iloc[9::10].reset_index(drop=True)
        self.baseline = Baseline.fit(cal)
        w = (one.t >= CAL_WINDOW[0]) & (one.t <= CAL_WINDOW[1])
        self.mu = one.loc[w, SENSORS].mean()
        self.buf10 = block[["t"] + SENSORS].tail(MIN_SAMPLES + 20).reset_index(drop=True)
        self.win1: deque = deque((one[["t"] + SENSORS].tail(60)).to_dict("records"), maxlen=60)

        self.gate = Gatekeeper(self.model, self.eps, slow_n=assets["eps_cfg"].get("slow_n", 60))
        self.t = float(block.t.iloc[-1])

    # ---------------------------------------------------------------------------------------- operator actions
    def inject(self, kind: str, sensor: Optional[str] = None, severity: float = 0.6, sign: int = 1,
               ramp_time: float = 300.0) -> None:
        f = FaultSpec(kind=kind, sensor=sensor if kind.startswith("sensor") else None, severity=severity,
                      sign=sign, ramp_time=ramp_time)
        f = self.core.inject(f)
        self.fault_times.append(dict(t=f.t_onset, kind=kind, sensor=sensor or ""))

    def clear_faults(self) -> None:
        self.core.clear_faults()
        self.fault_times.append(dict(t=self.t, kind="cleared", sensor=""))

    def set_demand(self, target: float, rate: float = 0.003) -> None:
        self.core.command_demand(target, rate)

    def rod_step(self, d_rho: float) -> None:
        self.core.command_rod_step(d_rho)

    # ---------------------------------------------------------------------------------------- statistical AI
    def _predict(self, which: str):
        df = pd.DataFrame(list(self.win1))
        df["run_id"] = 0
        if which == "C":
            df[SENSORS] = df[SENSORS] - self.mu
        f = extract_features(df, window=60, add_pairwise=(which == "C"))
        cols = self.assets["FNC"] if which == "C" else self.assets["FN"]
        X = f[cols].iloc[[-1]]
        m = self.assets[which]
        p = m.predict_proba(X)[0]
        k = int(np.argmax(p))
        return str(m.classes_[k]), float(p[k]), X

    # ---------------------------------------------------------------------------------------- stepping
    def step(self, seconds: int = 1) -> None:
        for _ in range(int(seconds)):
            rows = self.core.advance(10)
            self.buf10 = pd.concat([self.buf10, rows[["t"] + SENSORS]], ignore_index=True).tail(MIN_SAMPLES + 20)
            last = rows.iloc[-1]
            sens = {s: float(last[s]) for s in SENSORS}                      # MEASURED values only
            self.win1.append({"t": float(last.t), **sens})
            res = latest_residuals(self.buf10, self.model)
            rc = self.baseline.apply_row(res, 0.5 * (sens["flux_1"] + sens["flux_2"]))
            v = self.gate.update(float(last.t), sens, rc)
            ai_b, p_b, _ = self._predict("B")
            ai_c, p_c, _ = self._predict("C")
            self.t = float(last.t)
            self.hist.append(dict(
                t=self.t, **sens, gate=v.cls, gate_cands=",".join(v.candidates), gate_kind=v.kind, gate_reason=v.reason,
                gate_isolated=v.isolated, clean=dict(v.clean), tripped=",".join(v.tripped),
                ai_B=ai_b, ai_B_p=p_b, ai_C=ai_c, ai_C_p=p_c,
                ratios={n: float(self.gate.last_ratio[n]) for n in self.gate.names},
                # ---- ground truth: for display only, never passed to the gatekeeper or the AI
                truth=str(last["label"]), true_inventory=float(last["true_inventory"]), true_power=float(last["true_n"]),
                true_flow=float(last["true_W"]) / self.core.p.W0, demand=float(last["demand"])))

    # ---------------------------------------------------------------------------------------- views
    def frame(self, last_seconds: Optional[int] = None) -> pd.DataFrame:
        d = pd.DataFrame(list(self.hist))
        if last_seconds and len(d):
            d = d[d.t >= d.t.iloc[-1] - last_seconds]
        return d.reset_index(drop=True)

    def ratio_matrix(self, last_seconds: int = 300) -> pd.DataFrame:
        d = self.frame(last_seconds)
        return pd.DataFrame([r for r in d.ratios], index=d.t).T if len(d) else pd.DataFrame()

    def latest(self) -> Optional[dict]:
        return self.hist[-1] if self.hist else None

    def explain(self, which: str = "B", top_k: int = 8):
        """SHAP attributions of the AI's current prediction (needs `shap`; computed on demand)"""
        from src.ai_engine.explainer import RFExplainer
        if self._explainer is None:
            self._explainer = {}
        if which not in self._explainer:
            names = self.assets["FNC"] if which == "C" else self.assets["FN"]
            self._explainer[which] = RFExplainer(self.assets[which], names)
        _, _, X = self._predict(which)
        return self._explainer[which].explain(X, top_k=top_k)[0]
