"""
calibration.py -- per-run residual baseline ("commissioning calibration").

The mismatch between plant and model shows up as a CONSTANT offset in each residual (it scales with power for the
energy residuals). A single threshold must cover the worst plant in the whole mismatch envelope, which makes the
energy residuals blind to anything smaller than ~15-20 % of rated power. Learning each plant's own offset during a
known-healthy window and watching DEVIATIONS from it restores sensitivity to the noise level.

    offset_r = mean(r) / mean(power)   for residuals whose offset scales with power (E, F, C, D_flux)
    offset_r = mean(r)                 for all others
    r_cal    = r - offset_r * power    (or r - offset_r)

ASSUMPTION (state it in the report): the plant is healthy during the calibration window, which must end before
any fault. Uses measured quantities only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .residuals import RESIDUALS

CAL_WINDOW: Tuple[float, float] = (150.0, 270.0)       # seconds; after start-up settling, before any fault
PROPORTIONAL = ("E11", "E12", "E21", "E22", "F1", "F2", "C1", "C2", "D_flux")


def power_proxy(flux_1, flux_2):
    return 0.5 * (np.asarray(flux_1, float) + np.asarray(flux_2, float))


@dataclass
class Baseline:
    offsets: Dict[str, float]
    window: Tuple[float, float] = CAL_WINDOW

    @classmethod
    def fit(cls, df: pd.DataFrame, window: Tuple[float, float] = CAL_WINDOW, min_rows: int = 20) -> "Baseline":
        """df: t, flux_1, flux_2 and the RESIDUAL columns of ONE run"""
        m = (df.t >= window[0]) & (df.t <= window[1])
        d = df[m].dropna(subset=RESIDUALS)
        if len(d) < min_rows:
            raise ValueError(f"only {len(d)} usable rows in the calibration window {window}")
        pw = power_proxy(d.flux_1, d.flux_2).mean()
        off = {r: float(d[r].mean() / pw) if r in PROPORTIONAL else float(d[r].mean()) for r in RESIDUALS}
        return cls(off, window)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        pw = power_proxy(df.flux_1, df.flux_2)
        for r in RESIDUALS:
            out[r] = df[r] - self.offsets[r] * (pw if r in PROPORTIONAL else 1.0)
        return out

    def apply_row(self, res: Dict[str, float], power: float) -> Dict[str, float]:
        return {r: res[r] - self.offsets[r] * (power if r in PROPORTIONAL else 1.0) for r in RESIDUALS}

    def to_dict(self) -> dict:
        return {"offsets": self.offsets, "window": list(self.window)}
