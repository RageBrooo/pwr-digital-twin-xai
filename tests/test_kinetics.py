"""Sanity checks for the physics core: steady state, prompt jump, solver agreement, energy balance."""
import numpy as np
from scipy.integrate import solve_ivp

from src.simulator import (BETA, BETA_I, LAMBDA_I, PlantParams, ReactorCore, SimConfig, make_plant_pair,
                           random_profile)


def _core(mismatch=0.0, noise=0.0, seed=1, **cfg):
    model, true = make_plant_pair(mismatch, seed, nominal=PlantParams.nominal())
    from src.simulator import SensorConfig
    return ReactorCore(true, SensorConfig(noise_scale=noise), SimConfig(**cfg), ctrl_params=model, seed=seed)


def _flat_profile(duration=300.0, level=0.9):
    from src.simulator import LoadProfile
    return LoadProfile(np.array([0.0, duration + 1]), np.array([level, level]))


def test_delayed_neutron_data():
    assert abs(BETA - 0.006502) < 1e-6
    assert len(BETA_I) == len(LAMBDA_I) == 6


def test_feedback_coefficients_are_negative():
    p = PlantParams.nominal()
    assert p.alpha_D < 0 and p.alpha_M < 0


def test_steady_state_is_held_without_mismatch_or_noise():
    core = _core()
    core.reset(_flat_profile())
    df = core.advance(1200)                                   # 120 s
    assert df.true_n.max() - df.true_n.min() < 1e-4
    assert df.true_Tf.max() - df.true_Tf.min() < 0.05


def test_energy_balance_at_steady_state():
    core = _core()
    core.reset(_flat_profile())
    df = core.advance(600)
    last = df.iloc[-1]
    p = core.p
    q_cool = last.true_W * p.cp * (last.true_Tout - last.true_Tin)
    assert abs(q_cool / (p.P0 * last.true_n) - 1) < 1e-3


def test_prompt_jump_matches_theory():
    core = _core()
    core.reset(_flat_profile())
    rho = 5e-4                                                # 50 pcm step
    sol = solve_ivp(core._rhs, (0, 0.05), core.Y, method="Radau", args=(core.rho_ext + rho,),
                    rtol=1e-8, atol=1e-12)
    ratio = sol.y[0, -1] / core.Y[0]
    assert abs(ratio / (BETA / (BETA - rho)) - 1) < 0.01


def test_radau_and_bdf_agree():
    a, b = _core(method="Radau"), _core(method="BDF")
    prof = _flat_profile(level=0.85)
    a.reset(prof); b.reset(prof)
    for c in (a, b):
        c.command_rod_step(1.5e-4)
    da, db = a.advance(600), b.advance(600)
    assert np.abs(da.true_n - db.true_n).max() < 5e-4


def test_loop_is_stable_with_transients():
    model, true = make_plant_pair(0.05, 7, nominal=PlantParams.nominal())
    from src.simulator import SensorConfig
    core = ReactorCore(true, SensorConfig(), SimConfig(), ctrl_params=model, seed=7)
    df = core.simulate(900, random_profile(np.random.default_rng(7), 900))
    assert df.true_n.between(0.4, 1.2).all()
    assert np.isfinite(df.drop(columns=["label", "fault_sensor"]).values).all()
