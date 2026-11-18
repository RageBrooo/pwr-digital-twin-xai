"""
signature_matrix.py -- empirical residual-trip matrix on HELD-OUT test runs.

For each fault condition (kind:sensor) the share of settled-window rows (>= --settle s after onset) in which each
residual exceeds its epsilon. A residual "trips" for a condition when that share >= --trip. Reports:
  * the matrix (rates and binary),
  * conditions the physics cannot see at all (empty signature),
  * condition groups with IDENTICAL binary signatures = non-isolable by residuals alone,
  * the false-trip rate on the test runs' normal rows.
Needs configs/epsilon.yaml (scripts/tune_epsilon.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai_engine.dataset import WARMUP_S, load_manifest, load_runs, load_split   # noqa: E402
from src.gatekeeper.calibration import CAL_WINDOW, Baseline                         # noqa: E402
from src.gatekeeper.residuals import RESIDUALS                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--eps", default=str(ROOT / "configs" / "epsilon.yaml"))
    ap.add_argument("--settle", type=float, default=300.0)
    ap.add_argument("--trip", type=float, default=0.5)
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    eps = yaml.safe_load(Path(args.eps).read_text())["epsilon"]
    man = load_manifest(Path(args.raw))
    split = load_split(ROOT / "data" / "processed" / "split.json")
    df = load_runs(Path(args.raw), split[args.part]).merge(man[["run_id", "condition", "onset"]], on="run_id")
    cfg = yaml.safe_load(Path(args.eps).read_text())
    if cfg.get("calibrated", False):
        df = pd.concat([Baseline.fit(g).apply(g) for _, g in df.groupby("run_id", sort=False)], ignore_index=True)
        df = df[df.t > CAL_WINDOW[1]]
    df = df[df.t >= WARMUP_S].dropna(subset=RESIDUALS)
    trips = pd.DataFrame({r: df[r].abs() > eps[r] for r in RESIDUALS})
    settled = df.onset.notna() & (df.t >= df.onset + args.settle)
    rates = trips[settled].groupby(df.condition[settled]).mean()
    binary = (rates >= args.trip).astype(int)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    rates.round(3).to_csv(out / f"signature_rates_{args.part}.csv")
    binary.to_csv(out / f"signature_binary_{args.part}.csv")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(f"trip rate per residual, settled window, {args.part} runs (n runs per condition: {df[settled].groupby('condition').run_id.nunique().min()}-{df[settled].groupby('condition').run_id.nunique().max()})\n")
    print(rates.round(2).to_string())
    print("\nbinary signature (trip rate >= %.2f):\n" % args.trip, binary.to_string())
    empty = binary.index[(binary.sum(axis=1) == 0)].tolist()
    print("\nconditions with an EMPTY signature (physics residuals cannot see them):", empty)
    groups = binary.groupby(list(binary.columns)).apply(lambda g: list(g.index))
    dup = [g for g in groups if len(g) > 1 and not all(c in empty for c in g)]
    print("groups with identical non-empty signatures (not isolable by residuals alone):")
    for g in dup:
        print("   ", g)
    nrm = df[(df.label == "normal")]
    print("\nfalse-trip rate on normal rows of %s runs: %s" % (args.part, {r: round(float((nrm[r].abs() > eps[r]).mean()), 4) for r in RESIDUALS}))


if __name__ == "__main__":
    main()
