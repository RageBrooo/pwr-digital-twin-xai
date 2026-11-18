"""
Residual bank: zero on a perfect world, structured by sensor support, causal, and streaming-consistent.
All runs here are noise-free with ZERO mismatch unless stated (the gatekeeper's model == the plant).
"""
import numpy as np
import pytest

from src.gatekeeper.residuals import (HALF_WINDOW, RESIDUALS, SUPPORT, compute_residuals, latest_residuals,
                                      structural_matrix, structurally_non_isolable_pairs)
from src.simulator import SENSORS, run_scenario, sensor_view

KW = dict(mismatch=0.0, noise_scale=0.0, duration=700.0, warmup=150.0, t_onset=350.0, transients=True)


def _res(kind="normal", seed=3, **kw):
    r = run_scenario(seed, kind, **{**KW, **kw})
    return compute_residuals(sensor_view(r.df), r.model_params), r


def _tolerance(base):
    return {c: max(5 * float(base[c].abs().max()), 1e-6) for c in RESIDUALS}


# ---------------------------------------------------------------------------------------------- structure
def test_supports_only_reference_real_sensors():
    assert all(s in SENSORS for sup in SUPPORT.values() for s in sup)
    assert len(RESIDUALS) == 15 and structural_matrix().shape == (10, 15)


def test_no_two_sensors_share_a_residual_pattern():
    assert structurally_non_isolable_pairs() == []


# ---------------------------------------------------------------------------------------------- perfect world
def test_all_residuals_small_in_a_perfect_world_with_transients():
    R, _ = _res()
    R = R.iloc[2 * HALF_WINDOW:]
    assert R[RESIDUALS].notna().all().all()
    for c in ("H_flow", "S_pump", "L12", "L1", "L2", "D_flux", "D_tout"):
        assert R[c].abs().max() < 1e-9, c                          # static residuals are exactly zero
    for c in ("E11", "E12", "E21", "E22", "F1", "F2"):
        assert R[c].abs().max() < 0.01, c                          # <1 % of rated power (rod-step spikes only)
    for c in ("C1", "C2"):
        assert R[c].abs().max() < 0.002, c


def test_fuel_storage_term_matters():
    """the term the legacy residuals dropped: leaving it out must be clearly worse during load ramps"""
    from src.simulator.params import PlantParams
    from src.gatekeeper import residuals as rm
    r = run_scenario(1, "normal", **KW)
    p = r.model_params
    R = compute_residuals(sensor_view(r.df), p).iloc[60:]
    no_cf = compute_residuals(sensor_view(r.df), PlantParams(**{**p.to_dict(), "Cf": 1e-9})).iloc[60:]
    assert R["E11"].abs().quantile(0.99) < 0.5 * no_cf["E11"].abs().quantile(0.99)


# ---------------------------------------------------------------------------------------------- isolation
@pytest.mark.parametrize("sensor", SENSORS)
def test_bias_trips_only_residuals_that_read_that_sensor(sensor):
    base, _ = _res()
    tol = _tolerance(base.iloc[2 * HALF_WINDOW:])
    R, run = _res("sensor_bias", severity=0.5, sensor=sensor)
    post = R[R.t > run.fault.t_onset + 20][RESIDUALS].abs().max()
    tripped = {c for c in RESIDUALS if post[c] > tol[c]}
    assert tripped, f"bias on {sensor} tripped nothing"
    assert all(sensor in SUPPORT[c] for c in tripped), f"{sensor}: {tripped - {c for c in tripped if sensor in SUPPORT[c]}}"


def test_pump_degradation_trips_only_the_hydraulic_residual():
    base, _ = _res()
    tol = _tolerance(base.iloc[2 * HALF_WINDOW:])
    R, run = _res("pump_degradation", severity=0.6)
    post = R[R.t > run.fault.t_onset + 120][RESIDUALS].abs().max()
    assert {c for c in RESIDUALS if post[c] > tol[c]} == {"H_flow"}


def test_leak_moves_both_levels_but_not_their_difference():
    base, _ = _res()
    tol = _tolerance(base.iloc[2 * HALF_WINDOW:])
    R, run = _res("coolant_leak", severity=0.3)
    post = R[R.t > run.fault.t_onset + 100][RESIDUALS].abs().max()
    tripped = {c for c in RESIDUALS if post[c] > tol[c]}
    assert {"L1", "L2"} <= tripped and "L12" not in tripped and "S_pump" not in tripped


# ---------------------------------------------------------------------------------------------- causality
def test_residuals_are_causal():
    r = run_scenario(5, "normal", **{**KW, "noise_scale": 1.0, "mismatch": 0.03})
    v = sensor_view(r.df)
    k = 3000
    full = compute_residuals(v, r.model_params)
    tampered = v.copy()
    rng = np.random.default_rng(0)
    for s in SENSORS:
        tampered.loc[k + 1:, s] = rng.normal(size=len(v) - k - 1) * 1e3     # destroy the future
    cut = compute_residuals(tampered, r.model_params)
    a, b = full.iloc[:k + 1][RESIDUALS].to_numpy(), cut.iloc[:k + 1][RESIDUALS].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_streaming_helper_matches_offline_computation():
    r = run_scenario(6, "sensor_drift", sensor="T_out_1", **{**KW, "noise_scale": 1.0, "mismatch": 0.04})
    v = sensor_view(r.df)
    full = compute_residuals(v, r.model_params)
    for k in (200, 2500, 5000, len(v) - 1):
        live = latest_residuals(v.iloc[:k + 1], r.model_params)
        assert np.allclose([live[c] for c in RESIDUALS], full.iloc[k][RESIDUALS].to_numpy(), equal_nan=True)
    assert all(np.isnan(x) for x in latest_residuals(v.iloc[:10], r.model_params).values())


def test_gatekeeper_only_receives_nominal_parameters():
    r = run_scenario(7, "normal", **{**KW, "mismatch": 0.05})
    assert r.model_params != r.true_params
    assert r.model_params.flux_cal_1 == 1.0 and r.model_params.Cf == r.model_params.nominal().Cf
