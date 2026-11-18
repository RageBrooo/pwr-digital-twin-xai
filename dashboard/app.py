"""
Streamlit dashboard -- live digital twin: inject faults, watch the statistical AI and the physics Gatekeeper react.

Run from the PROJECT ROOT (the folder containing src/, scripts/, models/):
    streamlit run dashboard/app.py

Needs the outputs of: generate_data.py, train_models.py, tune_epsilon.py (models/ and configs/epsilon.yaml).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import plotly.graph_objects as go                     # noqa: E402
import streamlit as st                                # noqa: E402
from plotly.subplots import make_subplots             # noqa: E402

from dashboard.live import CAL_SECONDS, CLASSES, LiveTwin, load_assets, missing_assets   # noqa: E402
from src.simulator import SENSORS                                                          # noqa: E402

st.set_page_config(page_title="Reactor digital twin: AI vs physics Gatekeeper", layout="wide")
COLORS = {"normal": "#2ca02c", "coolant_leak": "#d62728", "pump_degradation": "#ff7f0e", "sensor_bias": "#1f77b4",
          "sensor_drift": "#9467bd", "sensor_stuck": "#8c564b", "unknown": "#7f7f7f"}
FAULT_KINDS = ["coolant_leak", "pump_degradation", "sensor_bias", "sensor_drift", "sensor_stuck"]


PLOT_CFG = {"toImageButtonOptions": {"format": "png", "scale": 3, "filename": "reactor_twin"}}   # camera icon = 3x PNG


def style(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", size=13))
    return fig


def show(fig):
    style(fig)
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    except TypeError:                                  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


@st.cache_resource(show_spinner="Loading models ...")
def assets():
    return load_assets(ROOT)


def new_twin(mismatch, noise, seed):
    with st.spinner("Running the 270 s healthy commissioning run and calibrating the baseline ..."):
        st.session_state.twin = LiveTwin(assets(), mismatch=mismatch, noise_scale=noise, seed=seed)
    st.session_state.explain = None


# ------------------------------------------------------------------------------------------------ guard
miss = missing_assets(ROOT)
if miss:
    st.error("Missing files: " + ", ".join(miss))
    st.code("python scripts/generate_data.py --runs-per-condition 4 --normal-runs 24 --duration 1800 --workers 4\n"
            "python scripts/train_models.py\npython scripts/tune_epsilon.py")
    st.stop()

# ------------------------------------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("1. Plant")
mismatch = sb.slider("model/plant mismatch", 0.0, 0.075, 0.03, 0.005, help="relative offset of the true plant from the model the Gatekeeper assumes")
noise = sb.slider("sensor noise (x nominal)", 0.5, 3.0, 1.0, 0.25)
seed = sb.number_input("random seed", 0, 10_000, 1)
if sb.button("Reset twin") or "twin" not in st.session_state:
    new_twin(mismatch, noise, int(seed))
tw = st.session_state.twin

sb.header("2. Operator")
demand = sb.slider("load demand", 0.70, 1.00, 0.95, 0.01)
c1, c2, c3 = sb.columns(3)
if c1.button("Ramp"):
    tw.set_demand(demand)
if c2.button("Rod +"):
    tw.rod_step(1.5e-4)
if c3.button("Rod -"):
    tw.rod_step(-1.5e-4)

sb.header("3. Inject a fault")
kind = sb.selectbox("fault type", FAULT_KINDS)
sensor = sb.selectbox("sensor", SENSORS, disabled=not kind.startswith("sensor"))
severity = sb.slider("severity", 0.2, 1.0, 0.6, 0.05)
sign = 1 if sb.radio("direction", ["+", "-"], horizontal=True) == "+" else -1
ramp = sb.slider("drift ramp time [s]", 120, 900, 300, 30, disabled=(kind != "sensor_drift"))
d1, d2 = sb.columns(2)
if d1.button("Inject", type="primary"):
    tw.inject(kind, sensor, severity, sign, ramp)
if d2.button("Clear faults"):
    tw.clear_faults()

sb.header("4. Run settings")
step = sb.select_slider("seconds per manual step", [5, 10, 30, 60, 120], value=30)
per_tick = sb.slider("plant seconds per tick while running", 1, 20, 5)
window = sb.slider("plot window [s]", 60, 900, 300, 30)
explain_on = sb.checkbox("SHAP explanation of the AI prediction (slower)")
st.session_state.setdefault("playing", False)
playing = st.session_state["playing"]
if not tw.hist:
    tw.step(60)                                        # first screen: a minute of normal operation

st.title("Reactor digital twin: statistical AI vs physics Gatekeeper")
st.caption(f"First {CAL_SECONDS} s were a healthy commissioning run (residual baseline calibrated). "
           "The AI and the Gatekeeper see MEASURED sensors only; ground truth is shown for you, not for them.")


c = st.columns([1.3, 1.0, 1.5, 1.4, 1.5, 1.1])
if c[0].button("⏸ Pause" if playing else "▶ Start simulation", type="primary", key="playbtn", help="runs the plant continuously"):
    st.session_state["playing"] = not playing
    st.rerun()
if c[1].button(f"Step +{step} s", key="stepbtn"):
    tw.step(step)
if c[2].button("Inject level_1 bias", key="q_level"):
    tw.inject("sensor_bias", "level_1", 0.8)
if c[3].button("Inject coolant leak", key="q_leak"):
    tw.inject("coolant_leak", None, 0.8)
if c[4].button("Inject pump degradation", key="q_pump"):
    tw.inject("pump_degradation", None, 0.8)
if c[5].button("Clear faults", key="q_clear"):
    tw.clear_faults()
if not tw.fault_times:
    st.info("**How to use:** press **Start simulation** (or Step), then press one of the **Inject** buttons, or build your own "
            "fault in the sidebar. Watch the GATEKEEPER card, the residual heatmap and the class timeline react.")


def render():
    if st.session_state.get("play_state"):
        tw.step(st.session_state.get("per_tick_state", 1))
    d = tw.frame(window)
    cur = tw.latest()
    if cur is None:
        return
    # ---------------------------------------------------------------- headline cards
    k = st.columns(5)
    k[0].metric("plant time", f"{cur['t']:.0f} s")
    k[1].metric("ground truth", cur["truth"])
    k[2].metric("AI-B (raw sensors)", cur["ai_B"])
    k[2].caption(f"confidence p = {cur['ai_B_p']:.2f}")
    k[3].metric("AI-C (calibrated sensors)", cur["ai_C"])
    k[3].caption(f"confidence p = {cur['ai_C_p']:.2f}")
    who = f" [{cur['gate_cands'].replace(',', ' / ')}]" if cur["gate_cands"] else ""
    k[4].metric("GATEKEEPER", cur["gate"] + (f" ({cur['gate_kind']})" if cur["gate_kind"] and cur["gate"] != "sensor_" + cur["gate_kind"] else ""))
    k[4].caption(f"isolated: {who.strip(' []')}" if who else "no fault isolated")
    if cur["gate"] != "normal" or cur["truth"] != "normal":
        ai = cur["ai_C"]
        if cur["gate"] == "normal":
            st.info("Gatekeeper sees no fault (physics residuals quiet).")
        elif ai == "normal":
            st.success(f"The Gatekeeper caught a fault the AI missed: **{cur['gate']}**{who}. Reason: {cur['gate_reason']}.")
        elif ai != cur["gate"]:
            st.warning(f"AI says **{ai}**, physics says **{cur['gate']}**{who}. The physics verdict is the validated one.")
        else:
            st.success(f"AI and Gatekeeper agree: **{cur['gate']}**{who}.")
    if cur["clean"]:
        for s, val in cur["clean"].items():
            st.write(f"**Repair:** sensor `{s}` isolated as faulty. Measured {cur[s]:.4g} -> replacement {val:.4g} "
                     "(twin channel or model estimate), so downstream models keep valid inputs.")
    # ---------------------------------------------------------------- telemetry
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("neutron flux (relative power)", "coolant temperatures [degC]",
                                        "flow & pump speed", "pressurizer level (inventory)"))
    for s, r, c in [("flux_1", 1, 1), ("flux_2", 1, 1), ("T_in", 1, 2), ("T_out_1", 1, 2), ("T_out_2", 1, 2),
                    ("flow", 2, 1), ("pump_speed", 2, 1), ("level_1", 2, 2), ("level_2", 2, 2)]:
        fig.add_trace(go.Scatter(x=d.t, y=d[s], name=s, mode="lines", line=dict(width=1.5)), row=r, col=c)
    fig.add_trace(go.Scatter(x=d.t, y=d.demand, name="demand", mode="lines", line=dict(dash="dot", color="gray")), row=1, col=1)
    for f in tw.fault_times:
        if f["t"] >= d.t.iloc[0]:
            fig.add_vline(x=f["t"], line_dash="dash", line_color="red" if f["kind"] != "cleared" else "green")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.12))
    show(fig)
    # ---------------------------------------------------------------- residuals + verdict timeline
    left, right = st.columns([3, 2])
    with left:
        rm = tw.ratio_matrix(window)
        if len(rm):
            hm = go.Figure(go.Heatmap(z=np.clip(rm.values, 0, 3), x=rm.columns, y=rm.index, zmin=0, zmax=3,
                                      colorscale=[[0, "#f7f7f7"], [0.33, "#fde0a8"], [1, "#b2182b"]],
                                      colorbar=dict(title="|r|/eps")))
            hm.update_layout(title="residuals / threshold (1.0 = tripped; slow Es included)", height=380,
                             margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
            show(hm)
    with right:
        tl = go.Figure()
        for i, (name, col) in enumerate([("truth", "truth"), ("GATEKEEPER", "gate"), ("AI-C", "ai_C"), ("AI-B", "ai_B")]):
            tl.add_trace(go.Scatter(x=d.t, y=[name] * len(d), mode="markers", name=name, showlegend=False,
                                    marker=dict(symbol="square", size=9, color=[COLORS.get(c, "#7f7f7f") for c in d[col]]),
                                    text=d[col], hovertemplate="%{x:.0f}s %{text}"))
        tl.update_layout(title="class timeline (green = normal)", height=380, margin=dict(l=10, r=10, t=40, b=10))
        show(tl)
        st.caption("Colours: " + "  ".join(f":{'green' if c == 'normal' else 'red' if c in ('coolant_leak', 'pump_degradation') else 'orange' if c.startswith('sensor') else 'gray'}[{c}]" for c in CLASSES))
    # ---------------------------------------------------------------- SHAP
    if explain_on:
        try:
            ex = tw.explain("C" if cur["ai_C"] != "normal" else "B")
            names = [t[0] for t in ex["top"]][::-1]
            vals = [t[1] for t in ex["top"]][::-1]
            sh = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=["#b2182b" if v > 0 else "#2166ac" for v in vals]))
            sh.update_layout(title=f"SHAP: why the AI says '{ex['predicted']}' (p={ex['probability']:.2f})", height=320,
                             margin=dict(l=10, r=10, t=40, b=10))
            show(sh)
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
    with st.expander("Ground truth of the plant (hidden from the AI and the Gatekeeper)"):
        st.write(pd.DataFrame({"true inventory": d.true_inventory, "true power": d.true_power, "true flow": d.true_flow}).tail(5))


st.session_state["play_state"] = playing
st.session_state["per_tick_state"] = per_tick
frag = st.fragment(run_every=1.0)(render) if (playing and hasattr(st, "fragment")) else render
frag()
