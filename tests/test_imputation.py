"""Imputation accuracy against GROUND TRUTH (tests may read true_*; the gatekeeper may not)."""
import numpy as np
import pytest

from src.gatekeeper.imputation import TWINS, impute
from src.simulator import SENSORS, run_scenario


@pytest.fixture(scope="module")
def run():
    r = run_scenario(4, "normal", mismatch=0.0, noise_scale=1.0, duration=600.0, warmup=200.0, t_onset=1e9)
    return r, r.df.iloc[1000::40]                                      # settled rows, every 4 s


def _ideal(df, sensor):
    return {"T_out_1": df.true_Tout, "T_out_2": df.true_Tout, "T_in": df.true_Tin, "T_fuel": df.true_Tf,
            "flux_1": df.true_n, "flux_2": df.true_n, "flow": df.true_W / 17500.0, "pump_speed": df.true_inventory * 0 + 1.0,
            "level_1": df.true_inventory, "level_2": df.true_inventory}[sensor]


TOL = {"T_out_1": 0.5, "T_out_2": 0.5, "T_in": 0.5, "T_fuel": 6.0, "flux_1": 0.02, "flux_2": 0.02, "flow": 0.02,
       "pump_speed": 0.01, "level_1": 0.01, "level_2": 0.01}


@pytest.mark.parametrize("sensor", SENSORS)
def test_replacement_is_close_to_the_true_value(run, sensor):
    r, d = run
    est = np.array([impute(sensor, row, r.model_params, suspects=(sensor,)) for row in d[SENSORS].to_dict("records")])
    assert np.abs(est - _ideal(d, sensor).values).mean() < TOL[sensor]


@pytest.mark.parametrize("sensor", [s for s in SENSORS if s not in TWINS])
def test_single_channel_sensors_use_the_model_estimate(run, sensor):
    r, d = run
    row = d[SENSORS].iloc[0].to_dict()
    row[sensor] = row[sensor] + 1e3                                  # wildly wrong reading must not leak into the estimate
    assert abs(impute(sensor, row, r.model_params, suspects=(sensor,)) - _ideal(d, sensor).iloc[0]) < 3 * TOL[sensor]


def test_twin_failover_is_skipped_when_the_twin_is_suspect(run):
    r, d = run
    row = d[SENSORS].iloc[0].to_dict()
    row["T_out_2"] += 50
    v = impute("T_out_1", row, r.model_params, suspects=("T_out_1", "T_out_2"))
    assert abs(v - _ideal(d, "T_out_1").iloc[0]) < 0.5                # fell back to the model estimate
