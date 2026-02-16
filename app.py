import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from citysim.config import Scenario
from citysim.engine import Simulation
from citysim.metrics import compactness

st.set_page_config(page_title="City Simulator (MVP)", layout="wide")

st.title("City Simulator (MVP) — форма города, дороги, отрасли, substitutes/complements")

with st.sidebar:
    st.header("Сценарий")
    scenario_path = st.selectbox("Выбери сценарий", ["scenarios/demo.yaml"])
    steps = st.slider("Шаги симуляции", 1, 100, 30)
    run_btn = st.button("Run")

sc = Scenario.from_yaml(scenario_path).raw

if run_btn:
    sim = Simulation(sc)
    hist = sim.run(steps)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Карта населения")
        fig, ax = plt.subplots()
        pop = np.where(sim.mask, sim.N, np.nan)
        im = ax.imshow(pop)
        ax.set_title("Population N(v)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        st.metric("Compactness", f"{compactness(sim.mask):.3f}")

    with col2:
        st.subheader("Динамика")
        st.line_chart({"avg_wage": hist["avg_wage"], "n_firms": hist["n_firms"], "total_output": hist["total_output"]})

        st.subheader("Фирмы (точки)")
        fig2, ax2 = plt.subplots()
        ax2.imshow(np.where(sim.mask, 0.2, np.nan))
        ys = [f.y for f in sim.firms]
        xs = [f.x for f in sim.firms]
        ax2.scatter(xs, ys, s=12)
        ax2.set_title("Firm locations")
        st.pyplot(fig2)

else:
    st.info("Нажми Run, чтобы посчитать симуляцию.")
