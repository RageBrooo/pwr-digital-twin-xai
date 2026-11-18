"""The stateful stepping interface that the dashboard depends on."""
import numpy as np

from src.simulator import (SENSORS, FaultSpec, LoadProfile, PlantParams, ReactorCore, SensorConfig, SimConfig,
                           make_plant_pair, random_profile)


def _core(seed=3, mismatch=0.03):
    model, true = make_plant_pair(mismatch, seed, nominal=PlantParams.nominal())
    return ReactorCore(true, SensorConfig(), SimConfig(), ctrl_params=model, seed=seed)


def _profile(seed=3):
    return random_profile(np.random.default_rng(seed), 600, warmup=100)


def test_tick_structure():
    c = _core()
    tick0 = c.reset(_profile())
    tick = c.step()
    for tk in (tick0, tick):
        assert set(tk) == {"t", "sensors", "inputs", "truth", "label", "faults"}
        assert set(tk["sensors"]) == set(SENSORS)
        assert all(k not in tk["sensors"] for k in tk["truth"])
    assert abs(tick["t"] - 0.1) < 1e-9 and tick["label"] == "normal"


def test_chunking_does_not_change_the_run():
    a, b = _core(), _core()
    a.reset(_profile()); b.reset(_profile())
    one = a.advance(300)
    many = [b.step() for _ in range(300)]
    flux_b = np.array([t["sensors"]["flux_1"] for t in many])
    tout_b = np.array([t["sensors"]["T_out_1"] for t in many])
    assert np.abs(one.flux_1.values - flux_b).max() < 1e-3          # identical noise stream, tiny solver diffs
    assert np.abs(one.T_out_1.values - tout_b).max() < 1e-2
    c = _core(); c.reset(_profile())
    odd = [c.advance(7), c.advance(23), c.advance(270)]               # chunks that straddle controller ticks
    assert np.abs(np.concatenate([d.flux_1.values for d in odd]) - one.flux_1.values).max() < 1e-3


def test_live_leak_injection_starts_now():
    c = _core()
    c.reset(_profile())
    c.advance(500)                                                    # 50 s
    f = c.inject(FaultSpec("coolant_leak", severity=0.8))
    assert abs(f.t_onset - 50.0) < 1e-9
    df = c.advance(3000)
    assert (df.label[df.t <= 50.0] == "normal").all() and (df.label[df.t > 50.1] == "coolant_leak").all()
    assert df.true_inventory.iloc[-1] < 0.95
    assert df.level_1.iloc[-1] < 0.96 and df.level_2.iloc[-1] < 0.96


def test_clear_faults_stops_a_sensor_fault():
    c = _core()
    c.reset(_profile())
    c.inject(FaultSpec("sensor_bias", sensor="T_out_1", severity=1.0, sign=1))
    biased = c.advance(100)
    c.clear_faults()
    clean = c.advance(100)
    assert biased.T_out_1.mean() - biased.T_out_2.mean() > 4.0
    assert abs(clean.T_out_1.mean() - clean.T_out_2.mean()) < 0.2


def test_live_demand_command_moves_power():
    c = _core(mismatch=0.0)
    c.reset(LoadProfile(np.array([0.0, 1e6]), np.array([0.95, 0.95])))
    c.advance(300)
    c.command_demand(0.75, rate=0.005)
    df = c.advance(2500)                                              # 250 s
    assert abs(df.true_n.iloc[-1] - 0.75) < 0.03


def test_live_rod_step_is_rejected_by_the_controller():
    c = _core(mismatch=0.0)
    c.reset(LoadProfile(np.array([0.0, 1e6]), np.array([0.9, 0.9])))
    c.advance(200)
    c.command_rod_step(2.0e-4)
    df = c.advance(3000)
    assert df.true_n.max() > 0.905                                    # disturbance is visible ...
    assert abs(df.true_n.iloc[-1] - 0.9) < 0.01                       # ... and then rejected
