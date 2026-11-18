"""Sensor layer and fault injection: sizes, shapes, independence of channels."""
import numpy as np
import pytest

from src.simulator import FAULT_FULL_SCALE, SENSORS, run_scenario, sensor_view

KW = dict(mismatch=0.03, duration=500.0, warmup=100.0, t_onset=200.0, transients=False)


def _pair(kind, sensor=None, severity=0.5, seed=11, **kw):
    base = run_scenario(seed, "normal", **{**KW, **kw}).df
    fault = run_scenario(seed, kind, severity=severity, sensor=sensor, **{**KW, **kw})
    return base, fault.df, fault.fault


def test_bias_is_a_constant_offset_on_one_channel():
    base, flt, f = _pair("sensor_bias", "T_out_1", 0.6)
    d = (flt.T_out_1 - base.T_out_1)[flt.t >= 200.0]
    assert np.allclose(d, f.sign * 0.6 * FAULT_FULL_SCALE["T_out_1"], atol=1e-9)
    assert np.allclose((flt.T_out_2 - base.T_out_2), 0.0, atol=1e-9)         # sibling channel untouched
    assert np.allclose(flt.true_Tout, base.true_Tout)                          # plant untouched


def test_drift_is_a_linear_ramp():
    base, flt, f = _pair("sensor_drift", "T_in", 1.0, ramp_time=200.0)
    d = (flt.T_in - base.T_in)[flt.t >= 200.0].values
    t = flt.t[flt.t >= 200.0].values - 200.0
    assert np.allclose(d, f.sign * FAULT_FULL_SCALE["T_in"] * t / 200.0, atol=1e-9)


def test_stuck_has_zero_variance_after_onset_and_only_there():
    _, flt, _ = _pair("sensor_stuck", "T_fuel")
    assert flt.T_fuel[flt.t >= 200.0].std() < 1e-9
    assert flt.T_fuel[flt.t < 200.0].std() > 0.1
    assert flt.T_out_1[flt.t >= 200.0].std() > 0.01


def test_level_channels_are_independent_so_a_level_fault_is_isolable():
    base, flt, f = _pair("sensor_bias", "level_1", 0.5)
    assert np.allclose(flt.level_2, base.level_2)                 # healthy twin channel untouched
    assert np.allclose(flt.true_inventory, base.true_inventory)   # real inventory untouched
    pre = base[base.t < 200.0]
    assert abs(np.corrcoef(pre.level_1, pre.level_2)[0, 1]) < 0.15          # independent noise


def test_leak_moves_both_level_channels_and_pump_fault_moves_neither():
    leak = run_scenario(12, "coolant_leak", severity=0.8, **KW).df
    pump = run_scenario(12, "pump_degradation", severity=0.8, **KW).df
    assert leak.true_inventory.iloc[-1] < 0.99 and leak.level_1.iloc[-1] < 0.99 and leak.level_2.iloc[-1] < 0.99
    assert abs(pump.true_inventory.iloc[-1] - 1.0) < 1e-3
    assert pump.flow.iloc[-1] < 0.9 * pump.flow.iloc[0]


def test_noise_level_scales_with_noise_scale():
    quiet = run_scenario(13, "normal", noise_scale=0.0, mismatch=0.0, duration=200, transients=False, t_onset=1e9).df
    loud = run_scenario(13, "normal", noise_scale=2.0, mismatch=0.0, duration=200, transients=False, t_onset=1e9).df
    assert quiet.T_in.std() < 1e-6
    assert 0.15 < loud.T_in.std() < 0.25                          # sigma 0.10 K * 2


def test_sensor_view_exposes_only_measurements():
    df = run_scenario(14, "normal", duration=100, warmup=50, t_onset=1e9).df
    v = sensor_view(df)
    assert list(v.columns) == ["t"] + SENSORS
    assert not any(c.startswith("true_") or c in ("label", "demand") for c in v.columns)
