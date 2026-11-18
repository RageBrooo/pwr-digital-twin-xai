"""
evaluate_gatekeeper.py -- head-to-head on HELD-OUT runs: statistical AI vs the physics Gatekeeper.

  AI-A      hardware-only Random Forest
  AI-B      full-class Random Forest on raw sensor windows
  AI-C      full-class Random Forest on per-run CALIBRATED sensors + redundant-channel differences (the fair baseline)
  GATE      physics only: calibrated residuals -> isolation -> verdict (the AI is not used)

Rows scored: normal rows after the calibration window + fault rows >= --settle s after onset.
Reports per-class recall with 95 % bootstrap intervals (resampling RUNS, not rows), accuracy, macro-F1, false alarms,
"fooled" rate (sensor fault called a HARDWARE fault), hardware fault called a sensor fault, sensor identification,
detection latency.

  --part val | test      the split made by train_models.py  (use val while developing; test once)
  --part all --raw DIR   every run in DIR, ignoring the split: use with a FRESH holdout from
                         `generate_data.py --out DIR --id-offset 1000`, never used for any decision
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_models import prepare_cal                                                  # noqa: E402
from src.ai_engine.dataset import CLASSES, WARMUP_S, load_manifest, load_runs, load_split     # noqa: E402
from src.ai_engine.features import extract_features, feature_names                            # noqa: E402
from src.gatekeeper.calibration import CAL_WINDOW, Baseline                                   # noqa: E402
from src.gatekeeper.decision import process_run                                               # noqa: E402
from src.simulator.params import PlantParams                                                  # noqa: E402

HW = ("coolant_leak", "pump_degradation")
SYSTEMS = ["AI-A", "AI-B", "AI-C", "GATE"]


def bootstrap(scored, systems, n_boot=1000, seed=0):
    """95 % intervals for accuracy and per-class recall, resampling whole runs"""
    rng = np.random.default_rng(seed)
    runs = scored.run_id.unique()
    idx = {r: np.where(scored.run_id.to_numpy() == r)[0] for r in runs}
    lab = scored.label.to_numpy()
    out = {}
    for s in systems:
        ok = (scored[s].to_numpy() == lab)
        per = {r: dict(n=len(i), c=int(ok[i].sum()), **{k: (int(((lab[i] == k)).sum()), int((ok[i] & (lab[i] == k)).sum())) for k in CLASSES})
               for r, i in idx.items()}
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(runs, len(runs))
            row = {"accuracy": sum(per[r]["c"] for r in pick) / sum(per[r]["n"] for r in pick)}
            for k in CLASSES:
                tot = sum(per[r][k][0] for r in pick)
                row[k] = sum(per[r][k][1] for r in pick) / tot if tot else np.nan
            draws.append(row)
        d = pd.DataFrame(draws)
        out[s] = {c: (float(np.nanpercentile(d[c], 2.5)), float(np.nanpercentile(d[c], 97.5))) for c in d.columns}
    return out


def build_frame(raw, ids, cfg, models_dir, gate_kw, fit_window=None):
    """one row per scored sample: AI predictions (AI-A/B/C) and the Gatekeeper verdict (GATE).
    fit_window: None (default window) | (start, end) | callable(onset) -> (start, end)  -- for calibration-window sweeps.
    The gate always starts after CAL_WINDOW[1]; only the window used to FIT the baseline changes."""
    raw = Path(raw)
    man_df = load_manifest(raw)
    man = man_df.set_index("run_id")
    eps, calibrated = cfg["epsilon"], cfg.get("calibrated", False)
    FN = joblib.load(Path(models_dir) / "feature_names.joblib")
    FNC = joblib.load(Path(models_dir) / "feature_names_C.joblib")
    ai = {k: joblib.load(Path(models_dir) / f"rf_{n}.joblib") for k, n in
          {"A": "A_hardware_only", "B": "B_full", "C": "C_full_cal"}.items()}
    model = PlantParams.nominal()
    df = load_runs(raw, ids)
    frames = []
    for rid, g in df.groupby("run_id", sort=False):
        onset = man.loc[rid, "onset"]
        g = g.sort_values("t").reset_index(drop=True)
        if calibrated:
            if fit_window is None:
                assert np.isnan(onset) or onset > CAL_WINDOW[1], f"run {rid}: fault starts inside the calibration window"
                win = CAL_WINDOW
            else:
                win = fit_window(onset) if callable(fit_window) else fit_window
            gc = Baseline.fit(g, win).apply(g)
        else:
            gc = g
        gc = gc[gc.t > CAL_WINDOW[1]].reset_index(drop=True)
        gate = process_run(gc, model, eps, slow_n=cfg.get("slow_n", 60), **gate_kw)
        feat = extract_features(g[g.t >= WARMUP_S], keep=[])
        out = gc[["run_id", "t", "label", "fault_sensor"]].merge(feat[["run_id", "t"] + FN], on=["run_id", "t"], how="inner")
        out = out.merge(gate, on="t", how="left")
        out["onset"], out["kind"], out["condition"] = onset, man.loc[rid, "kind"], man.loc[rid, "condition"]
        out["severity"] = man.loc[rid, "severity"]
        frames.append(out)
    d = pd.concat(frames, ignore_index=True)
    featC = prepare_cal(raw, man_df, ids)[["run_id", "t"] + FNC]
    d = d.merge(featC, on=["run_id", "t"], how="inner", suffixes=("", "_C"))
    d["AI-A"], d["AI-B"] = ai["A"].predict(d[FN]), ai["B"].predict(d[FN])
    XC = d[[(c + "_C") if (c + "_C") in d.columns else c for c in FNC]].copy()
    XC.columns = FNC
    d["AI-C"] = ai["C"].predict(XC)
    d["GATE"] = d.gate_cls
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--eps", default=str(ROOT / "configs" / "epsilon.yaml"))
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--part", default="val", choices=["val", "test", "all"])
    ap.add_argument("--settle", type=float, default=300.0)
    ap.add_argument("--persist", type=int, default=5)
    ap.add_argument("--release-k", type=int, default=5)
    ap.add_argument("--flat-k", type=int, default=10)
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--rise-max", type=float, default=5.0)
    ap.add_argument("--confirm-s", type=float, default=60.0)
    args = ap.parse_args()

    raw = Path(args.raw)
    man_df = load_manifest(raw)
    man = man_df.set_index("run_id")
    ids = None if args.part == "all" else load_split(ROOT / "data" / "processed" / "split.json")[args.part]
    cfg = yaml.safe_load(Path(args.eps).read_text())
    eps, calibrated = cfg["epsilon"], cfg.get("calibrated", False)
    gate_kw = dict(persist=args.persist, release_k=args.release_k, flat_k=args.flat_k, hold=args.hold,
                   rise_max=args.rise_max, confirm_s=args.confirm_s)
    d = build_frame(raw, ids, cfg, Path(args.models), gate_kw)

    post = d.onset.notna() & (d.t >= d.onset)
    settled = d.onset.notna() & (d.t >= d.onset + args.settle)
    normal = d.label == "normal"
    scored = d[normal | settled]
    res = {"part": args.part, "n_runs": int(d.run_id.nunique()), "n_rows_scored": int(len(scored)), "calibrated": bool(calibrated),
           "params": vars(args)}

    rec = pd.DataFrame({s: scored.assign(ok=scored[s] == scored.label).groupby("label").ok.mean() for s in SYSTEMS}).reindex(CLASSES)
    summary = pd.DataFrame({"accuracy": {s: accuracy_score(scored.label, scored[s]) for s in SYSTEMS},
                            "macro_F1": {s: f1_score(scored.label, scored[s], average="macro", labels=CLASSES) for s in SYSTEMS}})
    nrm, sf, hwf = d[normal], d[settled & d.label.str.startswith("sensor")], d[settled & d.label.isin(HW)]
    summary["false_alarm_rate"] = {s: float((nrm[s] != "normal").mean()) for s in SYSTEMS}
    summary["sensor_fault_called_HARDWARE"] = {s: float(sf[s].isin(HW).mean()) for s in SYSTEMS}
    summary["sensor_fault_missed(normal)"] = {s: float((sf[s] == "normal").mean()) for s in SYSTEMS}
    summary["hardware_fault_called_SENSOR"] = {s: float(hwf[s].str.startswith("sensor_").mean()) for s in SYSTEMS}
    ci = bootstrap(scored, SYSTEMS)

    pd.set_option("display.width", 230)
    print(f"\n=== {args.part.upper()} runs: {res['n_runs']} | rows scored {len(scored)} | calibrated residuals: {calibrated} ===")
    print("\nper-class recall [95% CI over runs] (settled fault window + normal rows):")
    tab = pd.DataFrame({s: {k: f"{rec.loc[k, s]:.3f} [{ci[s][k][0]:.2f},{ci[s][k][1]:.2f}]" for k in CLASSES} for s in SYSTEMS})
    print(tab.to_string())
    print("\n", summary.round(3).to_string())
    print("\naccuracy 95% CI:", {s: f"[{ci[s]['accuracy'][0]:.3f},{ci[s]['accuracy'][1]:.3f}]" for s in SYSTEMS})

    sid = sf.assign(contains=[s in c.split(",") for s, c in zip(sf.fault_sensor, sf.gate_cands.fillna(""))],
                    n_c=[len([x for x in c.split(",") if x]) for c in sf.gate_cands.fillna("")])
    sid["exact"] = sid.contains & (sid.n_c == 1)
    ident = sid.groupby("condition")[["contains", "exact"]].mean()
    ident["cls_correct"] = sf.assign(ok=sf.GATE == sf.label).groupby("condition").ok.mean()
    ident["AI-C_correct"] = sf.assign(ok=sf["AI-C"] == sf.label).groupby("condition").ok.mean()
    print("\nGATE sensor identification: truth in candidate set %.3f | exactly isolated %.3f | median set size %.1f"
          % (sid.contains.mean(), sid.exact.mean(), sid.n_c.median()))
    print("\nper condition:\n", ident.round(2).to_string())

    lat = []
    for rid, g in d[post].groupby("run_id"):
        row = {"run_id": rid, "kind": g.kind.iloc[0], "onset": g.onset.iloc[0]}
        for s in ("AI-C", "GATE"):
            hit = g[g[s] == g.label]
            row[s] = float(hit.t.min() - row["onset"]) if len(hit) else np.nan
        lat.append(row)
    lat = pd.DataFrame(lat)
    latsum = pd.concat([lat.groupby("kind")[["AI-C", "GATE"]].median().round(0),
                        lat.groupby("kind")[["AI-C", "GATE"]].apply(lambda x: x.isna().mean()).round(2).add_suffix("_never")], axis=1)
    print("\ndetection latency [s] (median over runs ever called correctly) and share never detected:\n", latsum.to_string())

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.part if args.part != "all" else "holdout"
    tab.to_csv(out / f"gate_eval_recall_ci_{tag}.csv")
    summary.round(4).to_csv(out / f"gate_eval_summary_{tag}.csv")
    ident.round(4).to_csv(out / f"gate_eval_ident_{tag}.csv")
    latsum.to_csv(out / f"gate_eval_latency_{tag}.csv")
    (out / f"gate_eval_{tag}.json").write_text(json.dumps({**res, "ci": ci}, default=str, indent=1))


if __name__ == "__main__":
    main()
