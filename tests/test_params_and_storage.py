"""Config files, mismatch generation (inverse-crime control) and Parquet + sidecar storage."""
import json

import numpy as np

from src.simulator import (PlantParams, load_run, load_sim_config, make_plant_pair, mismatch_offsets,
                           run_scenario, save_run)


def test_yaml_nominal_matches_dataclass_defaults():
    assert PlantParams.nominal() == PlantParams()


def test_simulator_yaml_loads():
    sim, sens, mm = load_sim_config()
    assert sim.dt == 0.1 and len(sens.sigma) == 10 and "Cf" in mm.fields


def test_zero_mismatch_means_identical_plants():
    model, true = make_plant_pair(0.0, 5, nominal=PlantParams.nominal())
    assert model == true


def test_mismatch_is_deterministic_bounded_and_seed_dependent():
    a = make_plant_pair(0.05, 1, nominal=PlantParams.nominal())
    b = make_plant_pair(0.05, 1, nominal=PlantParams.nominal())
    c = make_plant_pair(0.05, 2, nominal=PlantParams.nominal())
    assert a == b and a[1] != c[1]
    off = mismatch_offsets(*a)
    for k in ("Cf", "Cc0", "hA", "cp", "W0", "beta_exp", "flow_cal"):
        assert abs(off[k]) <= 0.05 + 1e-12
    assert abs(off["flux_cal_1"]) <= 0.05 * 1.25 + 1e-12 and abs(off["flux_cal_2"]) <= 0.05 * 1.25 + 1e-12
    assert abs(off["level_off_1"]) <= 0.1 * 0.05 + 1e-12
    assert "alpha_D" not in off and "P0" not in off                      # rated power is not mismatched


def test_mismatch_sweep_is_monotone_in_offset_size():
    sizes = [np.mean([abs(v) for k, v in mismatch_offsets(*make_plant_pair(m, 3, nominal=PlantParams.nominal())).items()
                      if not k.startswith("level")]) for m in (0.01, 0.03, 0.05)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_parquet_roundtrip_with_metadata_sidecar(tmp_path):
    res = run_scenario(21, "sensor_drift", sensor="flux_2", mismatch=0.04, duration=200, warmup=50, t_onset=100)
    path = save_run(res, tmp_path)
    assert path.exists() and path.with_suffix(".json").exists()
    df, meta = load_run(path)
    assert df.shape == res.df.shape and np.allclose(df.flux_1, res.df.flux_1)
    assert meta["seed"] == 21 and meta["mismatch"] == 0.04 and meta["fault"]["sensor"] == "flux_2"
    assert set(meta["mismatch_offsets"]) >= {"Cf", "flux_cal_1", "level_off_1"}
    json.dumps(meta)                                                      # fully serialisable
