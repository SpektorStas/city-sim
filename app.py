import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from citysim.config import Scenario
from citysim.engine import Simulation
from citysim.metrics import compactness

st.set_page_config(page_title="City Simulator (v0.3)", layout="wide")
st.title("City Simulator v0.3 — B2C + B2B + логистика + метрики эффективности")

with st.sidebar:
    st.header("Сценарий")
    scenario_path = st.selectbox("Выбери сценарий", ["scenarios/demo.yaml", "scenarios/demo_v02.yaml"])
    steps = st.slider("Шаги симуляции", 1, 150, 40)
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
        st.subheader("Ключевые метрики (временные ряды)")
        st.line_chart(
            {
                "avg_wage": hist["avg_wage"],
                "avg_income_pc": hist["avg_income_pc"],
                "avg_profit_margin": hist["avg_profit_margin"],
                "n_firms": hist["n_firms"],
                "opens": hist["opens"],
                "closes": hist["closes"],
                "relocations": hist["relocations"],
            }
        )

        st.subheader("Производство и цены инпутов")
        st.line_chart(
            {
                "total_output": hist["total_output"],
                "output_B2C": hist["total_output_B2C"],
                "output_B2B": hist["total_output_B2B"],
                "avg_input_price": hist["avg_input_price"],
            }
        )

    st.subheader("Фирмы (точки)")
    fig2, ax2 = plt.subplots()
    ax2.imshow(np.where(sim.mask, 0.2, np.nan))
    ys = [f.y for f in sim.firms]
    xs = [f.x for f in sim.firms]
    ax2.scatter(xs, ys, s=12)
    ax2.set_title("Firm locations (end of run)")
    st.pyplot(fig2)

    st.subheader("Рентабельность по отраслям (profit margin = profit/revenue)")
    pm = hist["profit_margin_by_industry"]
    # таблица: последняя точка
    last = {iid: pm[iid][-1] if len(pm[iid]) else np.nan for iid in pm.keys()}
    st.table(last)

    st.subheader("Динамика рентабельности по отраслям")
    # line chart expects same length arrays
    st.line_chart({iid: pm[iid] for iid in pm.keys()})

else:
    st.info("Нажми Run, чтобы посчитать симуляцию.")
