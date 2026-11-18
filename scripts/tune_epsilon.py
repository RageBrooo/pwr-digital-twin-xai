"""
tune_epsilon.py -- residual thresholds from NORMAL operation of the TRAINING runs only.

Mismatch is a per-RUN constant offset, so a pooled row quantile under-covers new plants. The default method takes the
q-quantile of |r| INSIDE each normal train run (rows labelled normal: includes pre-onset parts of fault runs and all
load-following transients) and sets  epsilon_r = margin * max over train runs of that per-run quantile.
Each run is first calibrated on its own healthy window (calibration.py) unless --no-calibrate.
The slow energy residual "Es" (mean of the four E residuals over --slow-n samples, computed from the gate's start) is
tuned the same way. The held-out VAL runs give an honest false-trip rate; test runs and fault rows are never used.
Writes configs/epsilon.yaml.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai_engine.dataset import WARMUP_S, assert_disjoint, load_manifest, load_runs, load_split, save_split, split_runs  # noqa: E402
from src.gatekeeper.calibration import CAL_WINDOW, Baseline                                                              # noqa: E402
from src.gatekeeper.decision import E_ALL, SLOW                                                                          # noqa: E402
from src.gatekeeper.residuals import RESIDUALS                                                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--q", type=float, default=0.999)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--method", default="worst_run", choices=["worst_run", "pooled"])
    ap.add_argument("--slow-n", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "configs" / "epsilon.yaml"))
    ap.add_argument("--no-calibrate", action="store_true", help="skip the per-run baseline (eps must then cover all mismatch)")
    args = ap.parse_args()
    raw = Path(args.raw)
    man = load_manifest(raw)
    sp_path = ROOT / "data" / "processed" / "split.json"
    if sp_path.exists():
        split = load_split(sp_path)
    else:
        split = split_runs(man)
        save_split(split, sp_path)
    assert_disjoint(split)
    parts = []
    for _, g in load_runs(raw).groupby("run_id", sort=False):
        if not args.no_calibrate:
            g = Baseline.fit(g).apply(g)
            g = g[g.t > CAL_WINDOW[1]]
        g = g.copy()
        g[SLOW] = g[list(E_ALL)].mean(axis=1).rolling(args.slow_n, min_periods=args.slow_n).mean()
        parts.append(g)
    names = list(RESIDUALS) + [SLOW]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.t >= WARMUP_S) & (df.label == "normal")].dropna(subset=names)
    tr, va = df[df.run_id.isin(split["train"])], df[df.run_id.isin(split["val"])]
    if args.method == "pooled":
        eps = {r: float(np.quantile(tr[r].abs(), args.q)) for r in names}
    else:
        per_run = tr.groupby("run_id")[names].agg(lambda x: np.quantile(x.abs(), args.q))
        eps = {r: float(args.margin * per_run[r].max()) for r in names}
    fa = {r: float((va[r].abs() > eps[r]).mean()) for r in names}
    any_fa = float(np.column_stack([va[r].abs() > eps[r] for r in names]).any(axis=1).mean())
    Path(args.out).write_text(yaml.safe_dump({"calibrated": not args.no_calibrate, "cal_window": list(CAL_WINDOW),
                                              "method": args.method, "q": args.q, "margin": args.margin, "slow_n": args.slow_n,
                                              "rate_hz": 1.0, "n_train_rows": int(len(tr)), "epsilon": eps}, sort_keys=False))
    print(f"train normal rows: {len(tr)} ({tr.run_id.nunique()} runs) | val normal rows: {len(va)} ({va.run_id.nunique()} runs)")
    print(f"{'residual':8s} {'epsilon':>10s} {'val false-trip rate':>20s}")
    for r in names:
        print(f"{r:8s} {eps[r]:10.5f} {fa[r]:20.5f}")
    print(f"any residual tripping on validation normal rows: {any_fa:.5f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
