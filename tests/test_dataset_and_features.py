"""Regression tests for the bugs found in the validation: dropped runs, row-level splits, non-causal features."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ai_engine.dataset import assert_disjoint, split_runs
from src.ai_engine.features import calibrate_sensors, extract_features, feature_names
from src.simulator import SENSORS

ROOT = Path(__file__).resolve().parents[1]


def _frame(n_runs=3, n=200, blank_meta=True, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(n_runs):
        d = pd.DataFrame({s: rng.normal(size=n) for s in SENSORS})
        d["t"] = np.arange(n, dtype=float)
        d["run_id"] = r
        d["fault_sensor"] = "" if (blank_meta and r % 2 == 0) else "T_in"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_runs_with_blank_metadata_are_not_dropped():
    """legacy bug: dropna() on a CSV round-trip deleted every run whose fault_sensor was blank"""
    df = _frame()
    df.loc[df.fault_sensor == "", "fault_sensor"] = np.nan          # exactly what read_csv produces
    f = extract_features(df, keep=["fault_sensor"])
    assert set(f.run_id) == {0, 1, 2}
    assert (f.groupby("run_id").size() == 200 - 59).all()


def test_feature_columns_match_declared_names_and_have_no_nans():
    f = extract_features(_frame(), keep=["fault_sensor"])
    assert all(c in f.columns for c in feature_names())
    assert not f[feature_names()].isna().any().any()


def test_features_are_causal():
    df = _frame(n_runs=1, n=300)
    a = extract_features(df)
    df2 = df.copy()
    df2.loc[df2.t > 150, SENSORS] = 1e6
    b = extract_features(df2)
    assert np.allclose(a[a.t <= 150][feature_names()].to_numpy(), b[b.t <= 150][feature_names()].to_numpy())


def _manifest():
    rows, rid = [], 0
    for cond, n in [("normal", 12), ("coolant_leak", 4), ("sensor_bias:flux_1", 4), ("sensor_drift:T_in", 3), ("pump_degradation", 2)]:
        for _ in range(n):
            rows.append(dict(run_id=rid, condition=cond))
            rid += 1
    return pd.DataFrame(rows)


def test_split_is_by_run_covers_everything_and_tests_every_condition():
    man = _manifest()
    sp = split_runs(man, seed=1)
    assert sorted(sum(sp.values(), [])) == sorted(man.run_id)            # every run used exactly once
    assert_disjoint(sp)
    test_conditions = set(man[man.run_id.isin(sp["test"])].condition)
    assert test_conditions == set(man.condition)


def test_overlapping_split_is_rejected():
    with pytest.raises(AssertionError):
        assert_disjoint({"train": [1, 2, 3], "val": [3], "test": [4]})


def test_generate_data_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_data.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    a = dict(out=str(tmp_path), duration=300.0, max_mismatch=0.05, noise_scale=1.0)
    gen.one_run((0, "sensor_bias", "T_in", a))
    gen.one_run((1, "normal", "", a))
    man = gen.build_manifest(tmp_path)
    df = pd.read_parquet(tmp_path / "run_00000.parquet")
    assert len(man) == 2 and set(man.condition) == {"sensor_bias:T_in", "normal"}
    assert not any(c.startswith("true_") for c in df.columns)           # no ground truth stored
    assert df.t.diff().dropna().eq(1.0).all()                            # 1 Hz
    assert {"E11", "L12", "H_flow", "label", "fault_sensor"} <= set(df.columns)
    assert df.label.isin(["normal", "sensor_bias"]).all()


def test_calibrate_sensors_removes_each_runs_offset_but_keeps_changes():
    df = _frame(n_runs=2, n=400, blank_meta=False)
    df.loc[df.run_id == 1, "flux_1"] += 0.05                    # run 1 has a calibration offset
    df.loc[df.t >= 300, "T_in"] += 2.0                          # a later change must survive
    out = calibrate_sensors(df)
    w = lambda d, c: d[(d.t >= 150) & (d.t <= 270)][c].mean()
    assert abs(w(out[out.run_id == 1], "flux_1")) < 1e-9 and abs(w(out[out.run_id == 0], "flux_1")) < 1e-9
    assert out[(out.run_id == 0) & (out.t >= 300)].T_in.mean() > 1.5


def test_pairwise_features_exist_when_requested():
    f = extract_features(_frame(1, 200), add_pairwise=True)
    assert "flux_1-flux_2" in f.columns and all(c in f.columns for c in feature_names(add_pairwise=True))
