"""Headless tests of the dashboard engine (skipped until models/ and configs/epsilon.yaml exist)."""
import pytest

from dashboard.live import ROOT, LiveTwin, load_assets, missing_assets
from src.simulator import SENSORS

pytestmark = pytest.mark.skipif(bool(missing_assets(ROOT)),
                                reason="run generate_data.py, train_models.py and tune_epsilon.py first")


@pytest.fixture(scope="module")
def assets():
    return load_assets(ROOT)


def test_commissioning_run_calibrates_and_first_steps_work(assets):
    tw = LiveTwin(assets, seed=1)
    assert abs(tw.t - 270.0) < 0.2 and tw.baseline.offsets
    tw.step(5)
    cur = tw.latest()
    assert cur["gate"] == "normal" and cur["truth"] == "normal" and set(SENSORS) <= set(cur)
    assert {"ai_B", "ai_C", "ratios", "truth", "true_inventory"} <= set(cur)


def test_normal_operation_raises_no_false_alarm(assets):
    tw = LiveTwin(assets, seed=2, mismatch=0.05)
    tw.step(150)
    assert (tw.frame().gate == "normal").mean() > 0.97


def test_live_level_bias_is_isolated_as_a_sensor_fault_not_a_leak(assets):
    tw = LiveTwin(assets, seed=3)
    tw.step(30)
    tw.inject("sensor_bias", "level_1", severity=0.8)
    tw.step(120)
    cur = tw.latest()
    assert cur["gate"] == "sensor_bias" and cur["gate_cands"] == "level_1" and "level_1" in cur["clean"]


def test_live_pump_degradation_is_confirmed(assets):
    tw = LiveTwin(assets, seed=4)
    tw.step(30)
    tw.inject("pump_degradation", severity=0.8)
    tw.step(150)
    assert tw.latest()["gate"] == "pump_degradation"


def test_live_stuck_sensor_is_caught_by_flatline(assets):
    tw = LiveTwin(assets, seed=5)
    tw.step(30)
    tw.inject("sensor_stuck", "T_in")
    tw.step(40)
    assert tw.latest()["gate"] == "sensor_stuck" and tw.latest()["gate_cands"] == "T_in"


def test_clearing_a_fault_returns_the_gatekeeper_to_normal(assets):
    tw = LiveTwin(assets, seed=6)
    tw.step(30)
    tw.inject("sensor_bias", "T_out_1", severity=1.0)
    tw.step(60)
    assert tw.latest()["gate"] != "normal"
    tw.clear_faults()
    tw.step(150)
    assert tw.latest()["gate"] == "normal"


def test_operator_demand_change_moves_the_plant(assets):
    tw = LiveTwin(assets, seed=7, demand=0.95)
    tw.step(10)
    tw.set_demand(0.75, rate=0.005)
    tw.step(200)
    assert tw.latest()["true_power"] < 0.85


def test_gatekeeper_receives_only_measured_sensors(assets):
    tw = LiveTwin(assets, seed=8)
    seen = []
    orig = tw.gate.update
    tw.gate.update = lambda t, s, r: (seen.append(set(s)), orig(t, s, r))[1]
    tw.step(3)
    assert seen and all(k == set(SENSORS) for k in seen)
