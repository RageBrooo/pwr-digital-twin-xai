"""
sweeps.py -- how robust is the Gatekeeper (and the statistical AI) away from the conditions it was tuned on?

Thresholds (epsilon) and models stay FIXED at their nominal training; only the world changes:
  mismatch sweep   fixed plant/model mismatch level (new datasets)         --mismatch 0 0.025 0.05 0.075
  noise sweep      sensor noise multiplier (new datasets)                  --noise 0.5 1 2 3
  severity bins    post-hoc on the fresh holdout (no new data)
  calibration      post-hoc on the fresh holdout: shorter healthy window (60 s, 30 s), a late window, and a
                   CONTAMINATED window (the fault starts 60 s before the window ends)

Generated datasets live in --out (resumable; finished runs are skipped). Each sweep point uses its own seeds.
Writes data/results/sweeps_summary.csv and sweeps.png, and prints compact tables.

  python scripts/sweeps.py --workers 4                 (~30-40 min on 4 cores; add --quick for ~half)
  python scripts/sweeps.py --skip-generate             (re-evaluate datasets already generated)
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_data as gen                                                  # noqa: E402
from scripts.evaluate_gatekeeper import HW, SYSTEMS, bootstrap, build_frame               # noqa: E402
from src.ai_engine.dataset import CLASSES                                                  # noqa: E402

SHOW = ["GATE", "AI-C", "AI-B"]


def make_dataset(out: Path, id_offset: int, workers: int, rpc: int, normal: int, duration: float,
                 min_mm: float, max_mm: float, noise: float):
    if (out / "manifest.csv").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    a = dict(out=str(out), duration=duration, max_mismatch=max_mm, min_mismatch=min_mm, noise_scale=noise)
    jobs, rid = [], id_offset
    for kind, sensor in gen.conditions():
        for _ in range(normal if kind == "normal" else rpc):
            jobs.append((rid, kind, sensor, a))
            rid += 1
    if workers > 1:
        with Pool(workers) as pool:
            list(pool.imap_unordered(gen.one_run, jobs))
    else:
        for j in jobs:
            gen.one_run(j)
    gen.build_manifest(out)


def summarise(d: pd.DataFrame, sweep: str, level, settle: float = 300.0, n_boot: int = 300):
    settled = d.onset.notna() & (d.t >= d.onset + settle)
    scored = d[(d.label == "normal") | settled]
    ci = bootstrap(scored, SYSTEMS, n_boot=n_boot)
    nrm, sf = d[d.label == "normal"], d[settled & d.label.str.startswith("sensor")]
    rows = []
    for s in SYSTEMS:
        rec = scored.assign(ok=scored[s] == scored.label).groupby("label").ok.mean()
        r = dict(sweep=sweep, level=level, system=s, n_runs=int(d.run_id.nunique()),
                 accuracy=float((scored[s] == scored.label).mean()), ci_lo=ci[s]["accuracy"][0], ci_hi=ci[s]["accuracy"][1],
                 false_alarm=float((nrm[s] != "normal").mean()), sensor_called_hw=float(sf[s].isin(HW).mean()))
        r.update({f"recall_{k}": float(rec.get(k, np.nan)) for k in CLASSES})
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(ROOT / "data" / "raw_holdout"))
    ap.add_argument("--out", default=str(ROOT / "data" / "sweeps"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--eps", default=str(ROOT / "configs" / "epsilon.yaml"))
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--mismatch", type=float, nargs="*", default=[0.0, 0.025, 0.05, 0.075])
    ap.add_argument("--noise", type=float, nargs="*", default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--runs-per-condition", type=int, default=2)
    ap.add_argument("--normal-runs", type=int, default=8)
    ap.add_argument("--duration", type=float, default=1200.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="1 run per condition, 4 normal runs")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-posthoc", action="store_true")
    ap.add_argument("--skip-new-data", action="store_true", help="only the post-hoc sweeps on the holdout")
    args = ap.parse_args()
    rpc, nrm = (1, 4) if args.quick else (args.runs_per_condition, args.normal_runs)
    cfg = yaml.safe_load(Path(args.eps).read_text())
    gate_kw = dict(persist=5, release_k=5, flat_k=10, hold=60, rise_max=5.0, confirm_s=60.0)
    models = Path(args.models)
    rows = []

    if not args.skip_new_data:
        points = [("mismatch", m, dict(min_mm=m, max_mm=m, noise=1.0)) for m in args.mismatch] + \
                 [("noise", n, dict(min_mm=0.0, max_mm=0.05, noise=n)) for n in args.noise]
        for k, (sweep, level, kw) in enumerate(points):
            out = Path(args.out) / f"{sweep}_{level}"
            if not args.skip_generate:
                print(f"[generate] {sweep}={level}", flush=True)
                make_dataset(out, 5000 + 1000 * k, args.workers, rpc, nrm, args.duration, **kw)
            d = build_frame(out, None, cfg, models, gate_kw)
            rows += summarise(d, sweep, level)
            print(f"[done] {sweep}={level}", flush=True)

    if not args.skip_posthoc:
        hold = Path(args.holdout)
        d0 = build_frame(hold, None, cfg, models, gate_kw)
        fault = d0[d0.onset.notna() & (d0.t >= d0.onset + 300) & d0.severity.notna()]
        bins = pd.cut(fault.severity, [0.2, 0.4, 0.6, 0.8, 1.0001], right=False, include_lowest=True)
        for b, g in fault.groupby(bins, observed=True):
            for s in SYSTEMS:
                sg = g[g.label.str.startswith("sensor")]
                rows.append(dict(sweep="severity", level=str(b), system=s, n_runs=int(g.run_id.nunique()),
                                 accuracy=float((g[s] == g.label).mean()), recall_sensor=float((sg[s] == sg.label).mean())))
        variants = {"120 s (default)": None, "60 s": (150.0, 210.0), "30 s": (150.0, 180.0), "30 s, late": (240.0, 270.0),
                    "contaminated (+60 s of fault)": lambda onset: (150.0, 270.0) if np.isnan(onset) else (150.0, onset + 60.0)}
        for name, fw in variants.items():
            d = d0 if fw is None else build_frame(hold, None, cfg, models, gate_kw, fit_window=fw)
            rows += summarise(d, "calibration_window", name)
            print(f"[done] calibration window: {name}", flush=True)

    res = pd.DataFrame(rows)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sweeps_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for sweep in res.sweep.unique():
        t = res[res.sweep == sweep]
        if sweep == "severity":
            print("\nseverity bins (settled fault rows): accuracy")
            print(t.pivot(index="level", columns="system", values="accuracy").round(3)[SHOW].to_string())
            continue
        acc = t.pivot(index="level", columns="system", values="accuracy")[SHOW].round(3)
        fa = t.pivot(index="level", columns="system", values="false_alarm")[["GATE", "AI-C"]].add_suffix("_falsealarm").round(3)
        hw = t.pivot(index="level", columns="system", values="sensor_called_hw")[["GATE"]].add_suffix("_sensor->hardware").round(3)
        print(f"\n{sweep}: accuracy per system, false alarms, GATE sensor-fault-called-hardware")
        print(pd.concat([acc, fa, hw], axis=1).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        for a, sweep in zip(ax.ravel()[:2], ["mismatch", "noise"]):
            t = res[res.sweep == sweep]
            for s, c in zip(SHOW, ["tab:green", "tab:blue", "tab:orange"]):
                u = t[t.system == s].sort_values("level")
                if len(u):
                    a.errorbar(u.level, u.accuracy, yerr=[u.accuracy - u.ci_lo, u.ci_hi - u.accuracy], marker="o", label=s, color=c, capsize=3)
            a.set_title(f"accuracy vs {sweep}" + (" (fraction)" if sweep == "mismatch" else " (x nominal)"))
            a.set_ylim(0, 1.02)
            a.legend()
        t = res[res.sweep == "severity"]
        if len(t):
            t.pivot(index="level", columns="system", values="accuracy")[SHOW].plot.bar(
                ax=ax[1, 0], color=["tab:green", "tab:blue", "tab:orange"], rot=0)
            ax[1, 0].set_title("accuracy by fault severity (settled fault rows)")
            ax[1, 0].set_ylim(0, 1.02)
        t = res[res.sweep == "calibration_window"]
        if len(t):
            t[t.system == "GATE"].set_index("level")["accuracy"].plot.barh(ax=ax[1, 1], color="tab:green")
            ax[1, 1].set_title("GATE accuracy vs calibration window")
            ax[1, 1].set_xlim(0, 1.02)
        plt.tight_layout()
        plt.savefig(out / "sweeps.png", dpi=130)
        print("\nwrote", out / "sweeps_summary.csv", "and", out / "sweeps.png")
    except Exception as e:                                           # plotting must never lose the numbers
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
