"""
scenario.py -- load-following demand, rod-step disturbances, and the random scenario generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class LoadProfile:
    t_pts: np.ndarray
    n_pts: np.ndarray
    rod_steps: List[Tuple[float, float]] = field(default_factory=list)      # (time, d_rho)

    def demand(self, t: float) -> float:
        return float(np.interp(t, self.t_pts, self.n_pts))

    def rho_dist(self, t: float) -> float:
        return float(sum(d for ts, d in self.rod_steps if ts <= t + 1e-9))

    # ---- live editing (dashboard) ----------------------------------------------------------
    def retarget(self, t_now: float, target: float, rate: float = 0.002) -> None:
        """from t_now, ramp the demand to `target` at `rate` [fraction of rated power / s]"""
        cur = self.demand(t_now)
        keep = self.t_pts < t_now
        t_end = t_now + max(abs(target - cur) / max(rate, 1e-6), 1.0)
        self.t_pts = np.concatenate([self.t_pts[keep], [t_now, t_end, t_end + 1e6]])
        self.n_pts = np.concatenate([self.n_pts[keep], [cur, target, target]])

    def add_rod_step(self, t: float, d_rho: float) -> None:
        self.rod_steps.append((float(t), float(d_rho)))

    def to_dict(self) -> dict:
        return dict(t_pts=self.t_pts.tolist(), n_pts=self.n_pts.tolist(), rod_steps=self.rod_steps)


def random_profile(rng: np.random.Generator, duration: float, warmup: float = 300.0,
                   ctrl_dt: float = 1.0, transients: bool = True) -> LoadProfile:
    """steady warm-up, then random load-follow ramps and rod-step disturbances (normal operation)"""
    n0 = rng.uniform(0.85, 1.0)
    t_pts, n_pts, rod = [0.0, warmup], [n0, n0], []
    if transients:
        t, level = warmup, n0
        while True:
            t += rng.uniform(100.0, 300.0)
            if t > duration - 60:
                break
            if rng.random() < 0.65:                                   # load-follow ramp
                tgt = float(np.clip(level + rng.uniform(-0.2, 0.15), 0.7, 1.0))
                rate = rng.uniform(0.0005, 0.003)                     # 3-18 %/min
                t_end = t + abs(tgt - level) / rate
                t_pts += [t, t_end]
                n_pts += [level, tgt]
                level, t = tgt, t_end
            else:                                                      # rod-step disturbance, 5-25 pcm
                d = rng.choice([-1, 1]) * rng.uniform(5e-5, 25e-5)
                rod.append((round(t / ctrl_dt) * ctrl_dt, float(d)))
    t_pts.append(duration + 1)
    n_pts.append(n_pts[-1])
    return LoadProfile(np.array(t_pts), np.array(n_pts), rod)
