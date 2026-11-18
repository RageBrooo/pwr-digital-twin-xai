"""
train_models.py -- statistical baselines, trained and scored with a RUN-LEVEL split.

  A_hardware_only  normal + coolant_leak + pump_degradation only (sensor faults are out-of-distribution)
  B_full           all six classes, raw sensor window features
  C_full_cal       all six classes, features from sensors CALIBRATED per run (healthy-window offset removed) plus the
                   redundant-channel differences: the fair "statistical model with the gatekeeper's advantage"

Scored on held-out runs: per-class recall over (normal rows + post-onset fault rows) and over the settled window.
Writes models/ and data/results/ai_metrics.json + csv tables.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai_engine.dataset import (CLASSES, WARMUP_S, assert_disjoint, load_manifest, load_runs, load_split,  # noqa: E402
                                   save_split, split_runs)
from src.ai_engine.features import CAL_WINDOW, calibrate_sensors, extract_features, feature_names           # noqa: E402


def _attach(feat, man):
    feat = feat.merge(man[["run_id", "kind", "onset"]], on="run_id", how="left")
    feat["post"] = feat.t >= feat.onset.fillna(np.inf)
    return feat


def prepare(raw: Path, split_path: Path, seed: int):
    man = load_manifest(raw)
    if split_path.exists():
        split = load_split(split_path)
    else:
        split = split_runs(man, seed)
        save_split(split, split_path)
    df = load_runs(raw)
    df = df[df.t >= WARMUP_S]
    feat = extract_features(df, keep=["label", "fault_sensor"])
    return man, split, _attach(feat, man)


def prepare_cal(raw: Path, man: pd.DataFrame, run_ids=None):
    """features for the calibrated statistical model (rows after the calibration window only)"""
    df = load_runs(raw, run_ids)
    df = calibrate_sensors(df)
    df = df[df.t >= WARMUP_S]
    feat = extract_features(df, keep=["label", "fault_sensor"], add_pairwise=True)
    feat = feat[feat.t > CAL_WINDOW[1]]
    return _attach(feat, man)


def fit_rf(X, y, seed, n_estimators):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=20, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(X, y)


def score(df, pred, settle):
    df = df.assign(pred=pred)
    out = {}
    for name, mask in {"post_onset": df.post | (df.label == "normal"),
                       "settled": (df.t >= df.onset.fillna(np.inf) + settle) | (df.label == "normal")}.items():
        d = df[mask]
        rec = d.assign(ok=d.pred == d.label).groupby("label").ok.mean()
        out[name] = dict(accuracy=float(accuracy_score(d.label, d.pred)),
                         macro_f1=float(f1_score(d.label, d.pred, average="macro")),
                         recall={k: float(v) for k, v in rec.items()}, n_rows=int(len(d)), n_runs=int(d.run_id.nunique()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--stride", type=int, default=2, help="use every k-th training row (adjacent rows are near-duplicates)")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--settle", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res_dir, mdl_dir = Path(args.results), Path(args.models)
    res_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    man, split, feat = prepare(Path(args.raw), ROOT / "data" / "processed" / "split.json", args.seed)
    featC = prepare_cal(Path(args.raw), man)
    assert_disjoint(split)
    FN, FNC = feature_names(), feature_names(add_pairwise=True)
    part = {k: feat[feat.run_id.isin(v)] for k, v in split.items()}
    partC = {k: featC[featC.run_id.isin(v)] for k, v in split.items()}
    print({k: f"{v.run_id.nunique()} runs / {len(v)} rows" for k, v in part.items()})
    tr, trC = part["train"].iloc[::args.stride], partC["train"].iloc[::args.stride]
    A_tr = tr[~(tr.post & tr.label.str.startswith("sensor"))]
    models = {"B_full": (fit_rf(tr[FN], tr.label, args.seed, args.trees), FN, part),
              "A_hardware_only": (fit_rf(A_tr[FN], A_tr.label, args.seed, args.trees), FN, part),
              "C_full_cal": (fit_rf(trC[FNC], trC.label, args.seed, args.trees), FNC, partC)}
    metrics = {"split_runs": {k: len(v) for k, v in split.items()}}
    for name, (m, cols, parts) in models.items():
        joblib.dump(m, mdl_dir / f"rf_{name}.joblib")
        metrics[f"classes_{name}"] = list(m.classes_)
        for sp in ("val", "test"):
            metrics[f"{name}/{sp}"] = score(parts[sp], m.predict(parts[sp][cols]), args.settle)
            if sp == "test":
                d = parts[sp][parts[sp].post | (parts[sp].label == "normal")]
                pd.DataFrame(confusion_matrix(d.label, m.predict(d[cols]), labels=CLASSES), index=CLASSES,
                             columns=CLASSES).to_csv(res_dir / f"confusion_{name}_test.csv")
        r = metrics[f"{name}/test"]["settled"]
        print(f"\n{name}: test, settled ->", {k: round(v, 3) for k, v in r["recall"].items()}, "| acc", round(r["accuracy"], 3))
    mA = models["A_hardware_only"][0]
    te = part["test"]
    sf = te[(te.t >= te.onset.fillna(np.inf) + args.settle) & te.label.str.startswith("sensor")]
    tab = pd.crosstab(sf.label, mA.predict(sf[FN]), normalize="index")
    tab.round(3).to_csv(res_dir / "baselineA_on_sensor_faults_test.csv")
    print("\nHardware-only model on sensor faults (test, settled): share of rows predicted as ...\n", tab.round(2).to_string())
    joblib.dump(FN, mdl_dir / "feature_names.joblib")
    joblib.dump(FNC, mdl_dir / "feature_names_C.joblib")
    (res_dir / "ai_metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
