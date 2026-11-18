"""
generate_data.py -- build the dataset: normal runs + both hardware faults + every sensor-fault type on EVERY sensor.

Each run: its own seed, its own random model/true mismatch (make_plant_pair), random severity / onset / drift ramp,
random load-following transients. Stored per run at 1 Hz (sensors, demand, label, residuals computed at 10 Hz).
Ground-truth plant columns (true_*) are NOT stored.

  python scripts/generate_data.py --runs-per-condition 4 --normal-runs 24 --duration 1800 --workers 4
  python scripts/generate_data.py --out data/raw_holdout --id-offset 1000 ...   (fresh holdout, never used for any decision)
Resumable: finished runs are skipped.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gatekeeper.residuals import RESIDUALS, compute_residuals          # noqa: E402
from src.simulator import SENSOR_FAULTS, SENSORS, run_scenario, sensor_view  # noqa: E402


def conditions():
    return [("normal", ""), ("coolant_leak", ""), ("pump_degradation", "")] + \
           [(k, s) for k in SENSOR_FAULTS for s in SENSORS]


def one_run(job):
    run_id, kind, sensor, a = job
    out = Path(a["out"])
    pq, js = out / f"run_{run_id:05d}.parquet", out / f"run_{run_id:05d}.json"
    if pq.exists() and js.exists():
        return run_id
    rng = np.random.default_rng(10_000_000 + run_id)
    dur = a["duration"]
    mismatch = float(rng.uniform(a.get("min_mismatch", 0.0), a["max_mismatch"]))
    severity = float(rng.uniform(0.2, 1.0))
    onset = float(round(rng.uniform(0.25, 0.5) * dur))
    ramp = float(rng.uniform(300.0, 900.0) * dur / 1800.0)
    res = run_scenario(run_id, kind, severity=severity, sensor=sensor or None, mismatch=mismatch,
                       noise_scale=a["noise_scale"], duration=dur, t_onset=onset, ramp_time=ramp)
    R = compute_residuals(sensor_view(res.df), res.model_params)
    df = res.df[["t"] + SENSORS + ["demand", "label"]].copy()
    df["fault_sensor"] = sensor
    for c in RESIDUALS:
        df[c] = R[c].values
    df = df.iloc[::10].reset_index(drop=True)                      # 10 Hz -> 1 Hz
    df.insert(0, "run_id", run_id)
    df.to_parquet(pq, index=False)
    meta = dict(run_id=run_id, kind=kind, sensor=sensor, condition=f"{kind}:{sensor}" if sensor else kind,
                severity=None if kind == "normal" else severity, onset=None if kind == "normal" else onset,
                ramp_time=ramp, sign=res.fault.sign, mismatch=mismatch, noise_scale=a["noise_scale"],
                duration=dur, mismatch_offsets=res.meta["mismatch_offsets"])
    js.write_text(json.dumps(meta, default=float))
    return run_id


def build_manifest(out: Path):
    rows = []
    for p in sorted(out.glob("run_*.json")):
        m = json.loads(p.read_text())
        m.pop("mismatch_offsets", None)
        rows.append(m)
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--runs-per-condition", type=int, default=4)
    ap.add_argument("--normal-runs", type=int, default=24)
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--min-mismatch", type=float, default=0.0, help="set equal to --max-mismatch for a fixed mismatch level")
    ap.add_argument("--max-mismatch", type=float, default=0.05)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after N runs (debug)")
    ap.add_argument("--id-offset", type=int, default=0, help="first run id; ids are the seeds, so a new offset = a FRESH independent holdout")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = dict(out=str(out), duration=args.duration, max_mismatch=args.max_mismatch, min_mismatch=args.min_mismatch, noise_scale=args.noise_scale)
    jobs, rid = [], args.id_offset
    for kind, sensor in conditions():
        for _ in range(args.normal_runs if kind == "normal" else args.runs_per_condition):
            jobs.append((rid, kind, sensor, a))
            rid += 1
    if args.limit:
        jobs = jobs[:args.limit]
    t0 = time.time()
    print(f"{len(jobs)} runs, {args.duration:.0f} s each, workers={args.workers}", flush=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(one_run, jobs), 1):
                if i % 10 == 0:
                    print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            one_run(j)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    man = build_manifest(out)
    print(f"done: {len(man)} runs in manifest, {man.condition.nunique()} conditions, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
