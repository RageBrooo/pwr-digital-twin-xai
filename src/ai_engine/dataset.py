"""
dataset.py -- loading runs and splitting them BY RUN (never by row or window).

A run is one simulator execution (one seed, one mismatch draw, one fault condition). All rows of a run go to
exactly one of train / val / test. The split is stratified by fault condition so every condition appears in the
test set, and is saved to disk for reproducibility. `assert_disjoint` is called wherever a split is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

WARMUP_S = 150.0          # initial controller settling; rows before this are never used for training/tuning/scoring
CLASSES = ["normal", "coolant_leak", "pump_degradation", "sensor_bias", "sensor_drift", "sensor_stuck"]


def load_manifest(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / "manifest.csv")


def load_runs(raw_dir: Path, run_ids=None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    ids = sorted(run_ids) if run_ids is not None else sorted(int(p.stem.split("_")[1]) for p in raw_dir.glob("run_*.parquet"))
    frames = [pd.read_parquet(raw_dir / f"run_{i:05d}.parquet") for i in ids]
    return pd.concat(frames, ignore_index=True)


def split_runs(manifest: pd.DataFrame, seed: int = 0) -> Dict[str, List[int]]:
    """per-condition split (~60/20/20, at least one test run per condition)"""
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for _, g in manifest.groupby("condition"):
        ids = list(rng.permutation(g.run_id.values))
        n = len(ids)
        n_test = max(1, round(0.2 * n))
        n_val = max(1, round(0.2 * n)) if n >= 4 else 0
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]
    split = {k: sorted(int(i) for i in v) for k, v in split.items()}
    assert_disjoint(split)
    return split


def assert_disjoint(split: Dict[str, List[int]]) -> None:
    sets = {k: set(v) for k, v in split.items()}
    for a in sets:
        for b in sets:
            if a < b:
                inter = sets[a] & sets[b]
                assert not inter, f"runs {sorted(inter)[:5]} appear in both {a} and {b}"


def save_split(split: Dict[str, List[int]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split))


def load_split(path: Path) -> Dict[str, List[int]]:
    split = json.loads(Path(path).read_text())
    assert_disjoint(split)
    return split
