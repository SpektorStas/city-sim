from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from citysim.config import Scenario
from citysim.engine import Simulation
from citysim.metrics import compactness


# ---------- helpers ----------
def list_scenarios() -> list[str]:
    base = Path("scenarios")
    if not base.exists():
        return []
    return sorted([str(p) for p in base.glob("*.yaml")])


def _imshow(ax, arr, title: str):
    im = ax.imshow(arr)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_firms(ax, sim: Simulation):
    """
    Scatter firms by industry with different markers.
    (Streamlit + matplotlib: keep it simple; no custom colors required.)
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]
    inds = sorted({f.industry for f in sim.firms})
    for j, iid in enumerate(inds):
        ys = [f.y for f in sim.firms if f.industry == iid]
        xs = [f.x for f in sim.firms if f.industry == iid]
        ax.scatter(xs, ys, s=18, marker=markers[j % len(markers)], label=iid)
    ax.legend(loc="upper right", fontsize=9, frameon=True)


def last_value(x):
    if x is None or len(x) == 0:
        return np.nan
    return x[-1]


# ---------- streamlit UI ----------
st.set_page_config(page_title="City-Sim", layout="wide")
st.title("City-Sim — spatial economy simulator (Streamlit)")

scenarios = list_scenarios()
if not scenarios:
    st.error("Не нашёл папку scenarios/ или *.yaml внутри неё. Создай scenarios/demo.yaml и т.п.")
    st.stop()

with st.sidebar:
    st.header("Run")
    scenario_path = st.selectbox("Scenario (YAML)", scenarios, index=min(0, len(scenarios) - 1))
    steps = st.slider("Steps", min_value=1, max_value=300, value=60, step=1)

    with st.expander("Advanced", expanded=False):
        iters_outer = st.slider("Outer iterations (markets)", 1, 15, 6)
        iters_b2b = st.slider("B2B iterations (logistics)", 1, 60, 20)

    run_btn = st.button("Run simulation", type="primary")

st.caption("Подсказка: сценарии лежат в папке scenarios/. Меняй параметры в YAML — и запускай заново.")

if not run_btn:
    st.info("Выбери сценарий и нажми **Run simulation**.")
    st.stop()

# ---------- load & run ----------
sc = Scenario.from_yaml(scenario_path).raw
sim = Simulation(sc)
hist = sim.run(steps)

# ---------- summary metrics ----------
colA, colB, colC, colD = st.columns(4)

colA.metric("Firms (end)", f"{len(sim.firms)}")
colB.metric("Avg wage (end)", f"{last_value(hist.get('avg_wage', [])):.3f}")
colC.metric("Avg income per capita (end)", f"{last_value(hist.get('avg_income_pc', [])):.3f}")
colD.metric("Compactness", f"{compactness(sim.mask):.3f}")

# ---------- maps ----------
st.subheader("Maps (end state)")

m1, m2, m3 = st.columns(3)

with m1:
    fig, ax = plt.subplots()
    pop = np.where(sim.mask, sim.N, np.nan)
    _imshow(ax, pop, "Population N(v)")
    st.pyplot(fig, clear_figure=True)

with m2:
    fig, ax = plt.subplots()
    wmap = np.where(sim.mask, sim.wage, np.nan)
    _imshow(ax, wmap, "Wage w(v)")
    st.pyplot(fig, clear_figure=True)

with m3:
    fig, ax = plt.subplots()
    rmap = np.where(sim.mask, sim.rent, np.nan)
    _imshow(ax, rmap, "Rent r(v)")
    st.pyplot(fig, clear_figure=True)

st.subheader("Firm locations (end)")
fig, ax = plt.subplots()
ax.imshow(np.where(sim.mask, 0.15, np.nan))
plot_firms(ax, sim)
ax.set_title("Firms by industry")
ax.set_xticks([])
ax.set_yticks([])
st.pyplot(fig, clear_figure=True)

# ---------- time series ----------
st.subheader("Time series")

# Safe-get series (older versions might not have all keys)
ts = {}
for k in [
    "avg_wage",
    "avg_income_pc",
    "avg_profit_margin",
    "n_firms",
    "opens",
    "closes",
    "relocations",
    "total_output",
    "total_output_B2C",
    "total_output_B2B",
    "avg_input_price",
]:
    if k in hist:
        ts[k] = hist[k]

# Show in two blocks to keep readable
c1, c2 = st.columns(2)
with c1:
    st.write("**Welfare & firms**")
    chart_keys = {k: ts[k] for k in ["avg_wage", "avg_income_pc", "avg_profit_margin", "n_firms"] if k in ts}
    if chart_keys:
        st.line_chart(chart_keys)
    else:
        st.warning("Нет ключей для графика welfare/firms в history.")

with c2:
    st.write("**Dynamics (entry/exit/relocation)**")
    chart_keys = {k: ts[k] for k in ["opens", "closes", "relocations"] if k in ts}
    if chart_keys:
        st.line_chart(chart_keys)
    else:
        st.warning("Нет ключей opens/closes/relocations в history.")

st.write("**Output & input prices**")
chart_keys = {k: ts[k] for k in ["total_output", "total_output_B2C", "total_output_B2B", "avg_input_price"] if k in ts}
if chart_keys:
    st.line_chart(chart_keys)

# ---------- profitability by industry ----------
pm = hist.get("profit_margin_by_industry", {})
if isinstance(pm, dict) and pm:
    st.subheader("Profitability by industry (profit margin)")

    # table: last + mean
    rows = []
    for iid, series in pm.items():
        series = np.array(series, dtype=float) if series is not None else np.array([], dtype=float)
        last = float(series[-1]) if len(series) else np.nan
        mean = float(np.nanmean(series)) if len(series) else np.nan
        rows.append((iid, last, mean))
    rows.sort(key=lambda x: (np.isnan(x[1]), x[0]))

    st.table(
        {
            "industry": [r[0] for r in rows],
            "last_margin": [r[1] for r in rows],
            "mean_margin": [r[2] for r in rows],
        }
    )

    st.write("**Margins (time series)**")
    st.line_chart({iid: pm[iid] for iid in pm.keys()})
else:
    st.info("В history нет profit_margin_by_industry (проверь engine.py v0.3+).")

# ---------- raw history download ----------
st.subheader("Export")
st.download_button(
    "Download history as JSON",
    data=str(hist).encode("utf-8"),
    file_name="history.json",
    mime="application/json",
)
