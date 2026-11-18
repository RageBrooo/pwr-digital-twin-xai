"""
runner.py -- one complete randomised run, and Parquet + JSON-sidecar persistence.

Every run is stored as  <name>.parquet  (10 Hz time series)  and  <name>.json  (metadata sidecar:
seed, mismatch level, the drawn parameter offsets, noise level, fault spec, load profile, configs).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .fault_injector import FaultSpec
from .params import (SENSOR_FAULTS, SENSORS, MismatchConfig, PlantParams, SensorConfig, SimConfig,
                     load_sim_config, make_plant_pair, mismatch_offsets)
from .reactor_core import ReactorCore
from .scenario import LoadProfile, random_profile


@dataclass
class RunResult:
    df: pd.DataFrame
    model_params: PlantParams        # the only plant knowledge the gatekeeper may use
    true_params: PlantParams         # analysis only
    fault: FaultSpec
    profile: LoadProfile
    meta: dict


def run_scenario(seed: int, kind: str = "normal", severity: float = 0.5, sensor: Optional[str] = None,
                 mismatch: float = 0.03, noise_scale: float = 1.0, duration: float = 1800.0,
                 warmup: float = 300.0, transients: bool = True, t_onset: Optional[float] = None,
                 ramp_time: float = 900.0, sensor_cfg: Optional[SensorConfig] = None,
                 cfg: Optional[SimConfig] = None, mcfg: Optional[MismatchConfig] = None) -> RunResult:
    sim_cfg, sc_cfg, mm_cfg = load_sim_config()
    cfg = cfg or sim_cfg
    sc = sensor_cfg or SensorConfig(sigma=sc_cfg.sigma, noise_scale=noise_scale, tau_T=sc_cfg.tau_T)
    mcfg = mcfg or mm_cfg
    rng = np.random.default_rng(seed)
    model, true = make_plant_pair(mismatch, seed, mcfg, nominal=PlantParams.nominal())
    profile = random_profile(rng, duration, warmup, cfg.ctrl_dt, transients)
    if t_onset is None:
        t_onset = float(round(rng.uniform(warmup + 60, 0.6 * duration)))
    if kind in SENSOR_FAULTS and sensor is None:
        sensor = str(rng.choice(SENSORS))
    fault = FaultSpec(kind=kind, t_onset=t_onset, severity=severity, sensor=sensor,
                      sign=int(rng.choice([-1, 1])), ramp_time=ramp_time)
    core = ReactorCore(true, sc, cfg, ctrl_params=model, seed=seed + 1)
    df = core.simulate(duration, profile, [fault])
    meta = dict(seed=seed, mismatch=mismatch, noise_scale=sc.noise_scale, duration=duration, warmup=warmup,
                transients=transients, fault=fault.to_dict(), mismatch_offsets=mismatch_offsets(model, true),
                model_params=model.to_dict(), true_params=true.to_dict(), sim_cfg=asdict(cfg),
                sensor_cfg=asdict(sc), profile=profile.to_dict())
    return RunResult(df, model, true, fault, profile, meta)


def save_run(res: RunResult, directory: Path, name: Optional[str] = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or f"run_{res.meta['seed']:06d}_{res.fault.kind}"
    res.df.to_parquet(directory / f"{name}.parquet", index=False)
    with open(directory / f"{name}.json", "w") as f:
        json.dump(res.meta, f, indent=1, default=float)
    return directory / f"{name}.parquet"


def load_run(path: Path) -> Tuple[pd.DataFrame, dict]:
    path = Path(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return pd.read_parquet(path), meta
