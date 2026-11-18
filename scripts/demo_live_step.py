"""
demo_live_step.py -- the loop the Streamlit dashboard will run: step the plant, inject faults mid-run,
read ONLY sensors (the gatekeeper's view), keep truth for the evaluation panel.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.simulator import (FaultSpec, LoadProfile, PlantParams, ReactorCore, SensorConfig, SimConfig,
                           make_plant_pair)

model, true = make_plant_pair(mismatch=0.03, seed=42, nominal=PlantParams.nominal())
core = ReactorCore(true, SensorConfig(), SimConfig(), ctrl_params=model, seed=42)
core.reset(LoadProfile(np.array([0.0, 1e6]), np.array([0.95, 0.95])))

script = {  # sim-time [s] -> operator action
    120: ("load ramp to 80 %", lambda: core.command_demand(0.80, rate=0.003)),
    300: ("INJECT level_1 bias",   lambda: core.inject(FaultSpec("sensor_bias", sensor="level_1", severity=0.5))),
    420: ("clear faults",          lambda: core.clear_faults()),
    480: ("INJECT coolant leak",   lambda: core.inject(FaultSpec("coolant_leak", severity=0.7))),
}
t_wall = time.time(); n_ticks = 0
print(f"{'t[s]':>6} {'label':>14} {'level_1':>8} {'level_2':>8} {'flow':>7} {'T_out_1':>8} | {'true inv':>8}  event")
for k in range(1, 7201):                       # 720 s of plant time, one 0.1 s tick at a time
    tick = core.step(); n_ticks += 1
    event = ""
    t_int = round(tick["t"], 1)
    if t_int in script and abs(t_int - round(t_int)) < 1e-9:
        event = script[int(t_int)][0]; script[int(t_int)][1]()
    if k % 600 == 0 or event:
        s, tr = tick["sensors"], tick["truth"]
        print(f"{tick['t']:6.1f} {tick['label']:>14} {s['level_1']:8.4f} {s['level_2']:8.4f} {s['flow']:7.4f} "
              f"{s['T_out_1']:8.2f} | {tr['inventory']:8.4f}  {event}")
dt_wall = time.time() - t_wall
print(f"\n{n_ticks} single steps in {dt_wall:.1f} s -> {dt_wall / n_ticks * 1000:.2f} ms/step "
      f"({n_ticks * 0.1 / dt_wall:.0f}x real time with step(); advance(n) is faster)")
