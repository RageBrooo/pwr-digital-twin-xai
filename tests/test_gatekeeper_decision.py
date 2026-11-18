"""Gatekeeper adjudicator: isolation logic, flatline, step/ramp typing, latching, and end-to-end on simulated runs."""
import numpy as np
import pytest

from src.gatekeeper.calibration import Baseline
from src.gatekeeper.decision import HARDWARE, TRIPPABLE, Gatekeeper, consistent_candidates, process_run
from src.gatekeeper.residuals import RESIDUALS, compute_residuals
from src.simulator import SENSORS, PlantParams, run_scenario, sensor_view

EPS = dict(E11=.03, E12=.03, E21=.03, E22=.03, F1=.03, F2=.03, C1=.03, C2=.03, H_flow=.04, S_pump=.01,
           L12=.012, L1=.012, L2=.012, D_flux=.02, D_tout=.6, Es=.012)
HEALTHY = dict(flux_1=.9, flux_2=.9, T_in=291., T_out_1=321., T_out_2=321., T_fuel=640., flow=1., pump_speed=1.,
               level_1=1., level_2=1.)


# ------------------------------------------------------------------------------------------- isolation table
@pytest.mark.parametrize("sensor", SENSORS)
def test_every_candidate_explains_its_own_full_trip_pattern(sensor):
    assert sensor in consistent_candidates(TRIPPABLE[sensor])


def test_hardware_signatures():
    assert consistent_candidates({"H_flow"}) == ["pump_degradation"]
    assert consistent_candidates({"L1", "L2"}) == ["coolant_leak"]
    assert consistent_candidates({"L1", "L2", "H_flow"}) == ["coolant_leak"]          # leak with flow penalty


def test_level_channels_are_isolated_not_called_a_leak():
    """legacy bug: a level_1 bias was reported as a coolant leak"""
    assert consistent_candidates({"L12", "L1"}) == ["level_1"]
    assert consistent_candidates({"L12", "L2"}) == ["level_2"]


def test_flow_and_pump_speed_sensor_faults_are_not_pump_degradation():
    """legacy bug: any flow / pump_speed sensor fault was reported as pump degradation"""
    assert consistent_candidates({"S_pump", "H_flow"}) == ["pump_speed"]
    assert consistent_candidates({"H_flow", "E11", "E12", "E21", "E22", "C1", "C2"}) == ["flow"]


def test_twin_channels_are_ambiguous_until_an_energy_residual_breaks_the_tie():
    assert set(consistent_candidates({"D_tout"})) == {"T_out_1", "T_out_2"}
    assert consistent_candidates({"D_tout", "E11", "E21", "C1"}) == ["T_out_1"]
    assert set(consistent_candidates({"D_flux"})) == {"flux_1", "flux_2"}
    assert consistent_candidates({"D_flux", "E21", "E22", "F2"}) == ["flux_2"]


def test_single_channel_thermal_sensors():
    assert consistent_candidates({"F1", "F2", "C1", "C2"}) == ["T_fuel"]              # E quiet -> fuel temperature
    assert set(consistent_candidates({"E11", "E12", "E21", "E22", "C1", "C2"})) == {"T_in", "flow"}


def test_inconsistent_pattern_is_unknown_not_forced():
    assert consistent_candidates({"D_tout", "D_flux"}) == []
    assert consistent_candidates({"L12", "L1", "L2"}) == []
    assert consistent_candidates(set()) == []


# ------------------------------------------------------------------------------------------- dynamics
def _feed(g, n, resid=None, sensors=None, t0=0.0, frozen=(), seed=0):
    """feed n samples; sensor values carry realistic noise (a perfectly constant channel IS a stuck sensor)"""
    rng = np.random.default_rng(seed + int(t0))
    out = None
    for i in range(n):
        r = {k: 0.0 for k in RESIDUALS}
        r.update(resid(i) if callable(resid) else (resid or {}))
        s = dict(HEALTHY if sensors is None else sensors(i))
        for k in s:
            if k not in frozen:
                s[k] += rng.normal(0, 1e-3)
        out = g.update(t0 + i, s, r)
    return out


def _noisy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda i, **kw: {"D_tout": float(rng.normal(0, 0.1))}


def test_step_is_bias_and_ramp_is_drift():
    nz = _noisy(1)
    g = Gatekeeper(PlantParams(), EPS)
    _feed(g, 100, nz)
    v = _feed(g, 20, lambda i: {"D_tout": 1.5 + nz(i)["D_tout"]}, t0=100)
    assert v.cls == "sensor_bias" and set(v.candidates) == {"T_out_1", "T_out_2"} and not v.isolated

    g = Gatekeeper(PlantParams(), EPS)
    _feed(g, 100, nz)
    v = _feed(g, 200, lambda i: {"D_tout": 0.01 * i + nz(i)["D_tout"]}, t0=100)       # 0.01 K/s ramp
    assert v.cls == "sensor_drift"


def test_flatline_detects_stuck_on_any_sensor_without_any_residual():
    g = Gatekeeper(PlantParams(), EPS, flat_k=10)
    assert _feed(g, 20).cls == "normal"
    v = _feed(g, 20, sensors=lambda i: dict(HEALTHY, T_in=291.123), frozen=("T_in",), t0=20)
    assert v.cls == "sensor_stuck" and v.candidates == ("T_in",) and v.isolated and "T_in" in v.clean


def test_ordinary_noise_never_looks_stuck():
    g = Gatekeeper(PlantParams(), EPS)
    rng = np.random.default_rng(0)
    for i in range(300):
        s = {k: v + rng.normal(0, 1e-3) for k, v in HEALTHY.items()}
        assert g.update(float(i), s, {k: 0.0 for k in RESIDUALS}).cls == "normal"


def test_diagnosis_is_latched_then_released():
    g = Gatekeeper(PlantParams(), EPS, hold=30)
    _feed(g, 20, {})
    _feed(g, 90, {"H_flow": 0.2}, t0=20)                                           # long enough to be confirmed
    assert _feed(g, 10, {}, t0=110).cls == "pump_degradation"                       # held
    assert _feed(g, 60, {}, t0=120).cls == "normal"                                 # released after `hold` quiet samples


def test_persistence_rejects_a_short_spike():
    g = Gatekeeper(PlantParams(), EPS, persist=5)
    _feed(g, 20, {})
    assert _feed(g, 3, {"H_flow": 0.5}, t0=20).cls == "normal"


def test_single_isolated_sensor_comes_with_a_replacement_value():
    g = Gatekeeper(PlantParams(), EPS)
    _feed(g, 20, {})
    v = _feed(g, 10, {"L12": 0.05, "L1": 0.05}, sensors=lambda i: dict(HEALTHY, level_1=1.05), t0=20)
    assert v.cls == "sensor_bias" and v.candidates == ("level_1",) and abs(v.clean["level_1"] - 1.0) < 5e-3


# ------------------------------------------------------------------------------------------- end to end
def _run(kind, sensor=None, sev=0.8, seed=11, ramp=300.0):
    r = run_scenario(seed, kind, severity=sev, sensor=sensor, mismatch=0.03, duration=1000.0, t_onset=450.0,
                     ramp_time=ramp)
    R = compute_residuals(sensor_view(r.df), r.model_params)
    df = r.df[["t"] + SENSORS].copy()
    for c in RESIDUALS:
        df[c] = R[c].values
    df = df.iloc[::10].reset_index(drop=True)
    df = Baseline.fit(df).apply(df)
    df = df[df.t > 270].reset_index(drop=True)
    out = process_run(df, r.model_params, EPS)
    return out[out.t < 450], out[out.t >= 700]


@pytest.mark.parametrize("kind,sensor,sev,cls,cand", [
    ("coolant_leak", None, 0.8, "coolant_leak", "coolant_leak"),
    ("pump_degradation", None, 0.8, "pump_degradation", "pump_degradation"),
    ("sensor_bias", "level_1", 0.8, "sensor_bias", "level_1"),
    ("sensor_bias", "pump_speed", 0.8, "sensor_bias", "pump_speed"),
    ("sensor_bias", "T_fuel", 1.0, "sensor_bias", "T_fuel"),
    ("sensor_drift", "flux_1", 1.0, "sensor_drift", "flux_1"),
    ("sensor_stuck", "T_in", 0.5, "sensor_stuck", "T_in"),
])
def test_end_to_end_on_simulated_faults(kind, sensor, sev, cls, cand):
    pre, late = _run(kind, sensor, sev)
    assert (pre.gate_cls == "normal").mean() > 0.99                                  # no false alarm before onset
    assert (late.gate_cls == cls).mean() > 0.95
    assert (late.gate_cands == cand).mean() > 0.95


def test_end_to_end_normal_run_stays_normal():
    pre, late = _run("normal")
    assert (pre.gate_cls == "normal").all() and (late.gate_cls == "normal").mean() > 0.99


# ------------------------------------------------------------------------------------------- new in this version
def test_slow_residual_separates_flow_sensor_fault_from_pump_degradation():
    assert consistent_candidates({"H_flow"}) == ["pump_degradation"]
    assert consistent_candidates({"H_flow", "Es"}) == ["flow"]                      # real pump fault never moves Es
    assert set(consistent_candidates({"Es"})) == {"T_in", "flow"}


def test_pump_degradation_waits_for_the_slow_residual_before_being_declared():
    g = Gatekeeper(PlantParams(), EPS, confirm_s=60.0)
    _feed(g, 20, {})
    early = _feed(g, 30, {"H_flow": 0.2}, t0=20)
    assert early.cls == "unknown" and "pump_degradation" in early.candidates        # confirming
    late = _feed(g, 60, {"H_flow": 0.2}, t0=50)
    assert late.cls == "pump_degradation"


def test_flow_sensor_fault_is_not_called_pump_degradation():
    g = Gatekeeper(PlantParams(), EPS, confirm_s=60.0)
    _feed(g, 20, {})
    r = {"H_flow": 0.2, "E11": -0.05, "E12": -0.05, "E21": -0.05, "E22": -0.05}     # the slow mean (-0.05) trips Es once full
    v = _feed(g, 120, r, t0=20)
    assert v.cls.startswith("sensor_") and v.candidates == ("flow",)


def test_without_a_slow_threshold_the_old_behaviour_is_kept():
    eps = {k: v for k, v in EPS.items() if k != "Es"}
    g = Gatekeeper(PlantParams(), eps)
    _feed(g, 20, {})
    assert _feed(g, 12, {"H_flow": 0.2}, t0=20).cls == "pump_degradation"


def test_hysteresis_keeps_a_marginal_fault_armed():
    g = Gatekeeper(PlantParams(), EPS)
    _feed(g, 40, {})
    _feed(g, 8, {"D_tout": 0.9}, t0=40)                                             # armed: 1.5 x epsilon (0.6)
    seq = [1.3 * 0.6, 0.7 * 0.6, 0.8 * 0.6, 1.2 * 0.6, 0.6 * 0.6, 0.9 * 0.6]       # then hovering around epsilon
    cls = [_feed(g, 1, {"D_tout": seq[i % 6]}, t0=48 + i).cls for i in range(60)]
    assert "normal" not in cls                                                      # hysteresis: no flicker once armed


def test_bias_type_is_remembered_when_the_diagnosis_is_dropped_and_redeclared():
    nz = _noisy(2)
    g = Gatekeeper(PlantParams(), EPS, hold=20)
    _feed(g, 100, nz)
    assert _feed(g, 15, lambda i: {"D_tout": 1.5 + nz(i)["D_tout"]}, t0=100).cls == "sensor_bias"
    assert _feed(g, 60, nz, t0=115).cls == "normal"                                 # fault vanished, diagnosis cleared
    v = _feed(g, 40, lambda i: {"D_tout": 0.03 * i + nz(i)["D_tout"]}, t0=175)    # re-declared on a slow rise
    assert v.cls == "sensor_bias"                                                   # type kept from the earlier declaration


@pytest.mark.parametrize("sensor,sev", [("flow", 1.0)])
def test_end_to_end_flow_sensor_fault(sensor, sev):
    pre, late = _run("sensor_bias", sensor, sev)
    assert (late.gate_cls == "sensor_bias").mean() > 0.9 and (late.gate_cands == "flow").mean() > 0.9
