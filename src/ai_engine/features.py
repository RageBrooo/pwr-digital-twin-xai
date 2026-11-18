"""
features.py -- sliding-window features from MEASURED channels only (the statistical baseline's view).

Per sensor: current value, causal window mean, window std, and change over the window (x_t - x_{t-w}).
Optional `add_pairwise` adds the redundant-channel differences (flux_1-flux_2, T_out_1-T_out_2, level_1-level_2);
it is OFF by default because it hands the "statistical" model a model-free redundancy check.

Fix of the legacy bug: rows are only dropped when a FEATURE is NaN (the start-up window of each run).
Metadata columns are passed through untouched via `keep`, so blank metadata can never delete a run.
Never reads ground-truth columns: it only touches `t`, SENSORS and the columns named in `keep`.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from src.simulator.params import SENSORS

PAIRS = [("flux_1", "flux_2"), ("T_out_1", "T_out_2"), ("level_1", "level_2")]
CAL_WINDOW = (150.0, 270.0)       # same healthy window the gatekeeper calibrates on


def calibrate_sensors(df: pd.DataFrame, run_col: str = "run_id", window=CAL_WINDOW) -> pd.DataFrame:
    """give the statistical model the same per-run advantage as the gatekeeper: subtract each sensor's mean over the
    run's own healthy calibration window (removes constant calibration offsets; changes relative to it remain)"""
    out = df.copy()
    for _, idx in df.groupby(run_col).groups.items():
        g = df.loc[idx]
        m = (g.t >= window[0]) & (g.t <= window[1])
        out.loc[idx, SENSORS] = g[SENSORS] - g.loc[m, SENSORS].mean()
    return out


def feature_names(add_pairwise: bool = False) -> List[str]:
    base = list(SENSORS)
    names = base + [f"{s}_{k}" for s in base for k in ("mean", "std", "delta")]
    if add_pairwise:
        names += [f"{a}-{b}" for a, b in PAIRS] + [f"{a}-{b}_mean" for a, b in PAIRS]
    return names


def extract_features(df: pd.DataFrame, run_col: str = "run_id", window: int = 60,
                     keep: Sequence[str] = (), add_pairwise: bool = False) -> pd.DataFrame:
    """df: 1 Hz samples, one or more runs (rows ordered in time within a run). `window` is in samples."""
    keep = [c for c in dict.fromkeys([run_col, "t", *keep]) if c in df.columns]
    out = []
    for _, g in df.groupby(run_col, sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        cols = {s: g[s] for s in SENSORS}
        if add_pairwise:
            for a, b in PAIRS:
                cols[f"{a}-{b}"] = g[a] - g[b]
        feats = {}
        for name, x in cols.items():
            feats[name] = x
            if name in SENSORS or name.count("-") == 1:
                feats[f"{name}_mean"] = x.rolling(window, min_periods=window).mean()
                if name in SENSORS:
                    feats[f"{name}_std"] = x.rolling(window, min_periods=window).std()
                    feats[f"{name}_delta"] = x - x.shift(window - 1)
        f = pd.concat([g[keep], pd.DataFrame(feats)], axis=1)
        out.append(f.dropna(subset=list(feats)))
    return pd.concat(out, ignore_index=True)
