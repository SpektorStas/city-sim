from __future__ import annotations

from pathlib import Path
import time

import streamlit as st
import pandas as pd
import yaml

from citysim.config import Scenario
from citysim.engine import Simulation
from citysim.metrics import compactness
from citysim.scenario_builder import (
    default_scenario_like,
    scenario_to_tables,
    tables_to_scenario,
    validate_scenario,
    dump_yaml,
    load_yaml_text,
    INDUSTRY_COLS,
    LINK_COLS,
    GROUP_COLS,
)
from citysim.viz import plot_city


# ---------------------------
# Utilities
# ---------------------------

def list_scenarios():
    base = Path("scenarios")
    if not base.exists():
        return []
    return sorted([str(p) for p in base.glob("*.yaml")])


def ensure_state_from_yaml(path: str):
    raw = Scenario.from_yaml(path).raw
    st.session_state["base_scenario"] = raw
    st.session_state["scenario"] = default_scenario_like(raw)

    df_i, df_l, df_g = scenario_to_tables(st.session_state["scenario"])
    st.session_state["df_industries"] = df_i
    st.session_state["df_links"] = df_l
    st.session_state["df_groups"] = df_g


def reset_sim():
    st.session_state["sim"] = None
    st.session_state["history"] = None


# ---------------------------
# Page
# ---------------------------

st.set_page_config(page_title="City-Sim", layout="wide")
st.title("City-Sim — scenario builder + runner (Streamlit)")

scenarios = list_scenarios()
if not scenarios:
    st.error("Не нашёл scenarios/*.yaml. Положи туда demo_v02.yaml и т.п.")
    st.stop()

with st.sidebar:
    st.header("Scenario")
    scenario_path = st.selectbox("Load YAML", scenarios, index=0)
    if st.button("Load into Builder"):
        ensure_state_from_yaml(scenario_path)
        reset_sim()

# init on first load
if "scenario" not in st.session_state:
    ensure_state_from_yaml(scenarios[0])
    reset_sim()

tabs = st.tabs(["Build", "Run", "Export"])

# ---------------------------
# BUILD
# ---------------------------
with tabs[0]:
    st.subheader("Build scenario in UI (no manual YAML edits)")

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Industries")
        st.caption("Добавляй/редактируй отрасли. id должно быть уникальным.")
        df_i = st.session_state["df_industries"]
        edited_i = st.data_editor(
            df_i,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_industries",
        )

    with colB:
        st.markdown("### B2B Links (endogenous inputs)")
        st.caption("Связь: buyer (покупатель-индустрия) ← supplier (поставщик-индустрия).")
        df_l = st.session_state["df_links"]
        edited_l = st.data_editor(
            df_l,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_links",
        )

    st.markdown("### Households groups")
    st.caption("industries_csv и weights_csv вводи через запятую, например: retail, cafe | 1, 0.8")
    df_g = st.session_state["df_groups"]
    edited_g = st.data_editor(
        df_g,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_groups",
    )

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Apply changes", type="primary"):
            st.session_state["df_industries"] = edited_i[INDUSTRY_COLS].copy()
            st.session_state["df_links"] = edited_l[LINK_COLS].copy()
            st.session_state["df_groups"] = edited_g[GROUP_COLS].copy()

            s_new = tables_to_scenario(
                st.session_state["base_scenario"],
                st.session_state["df_industries"],
                st.session_state["df_links"],
                st.session_state["df_groups"],
            )
            errs = validate_scenario(s_new)
            if errs:
                st.error("Есть ошибки в сценарии:")
                for e in errs:
                    st.write("- " + e)
            else:
                st.success("Сценарий обновлён.")
                st.session_state["scenario"] = s_new
                reset_sim()

    with col2:
        if st.button("Reset to YAML base"):
            ensure_state_from_yaml(scenario_path)
            reset_sim()
            st.success("Сброшено к исходному YAML.")

    with col3:
        st.metric("Industries", len(st.session_state["df_industries"]))
        st.metric("Links", len(st.session_state["df_links"]))
        st.metric("Groups", len(st.session_state["df_groups"]))

    st.markdown("### Quick settings (grid/roads/population)")
    s = st.session_state["scenario"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s["grid"]["w"] = st.number_input("Grid w", 10, 300, int(s["grid"]["w"]))
        s["grid"]["h"] = st.number_input("Grid h", 10, 300, int(s["grid"]["h"]))
    with c2:
        s["grid"]["walk_radius"] = st.number_input("Walk radius", 0.0, 10.0, float(s["grid"].get("walk_radius", 2.0)))
    with c3:
        s["roads"]["connectivity"] = st.slider("Road connectivity", 0.05, 1.0, float(s["roads"].get("connectivity", 1.0)))
    with c4:
        s["population"]["total"] = st.number_input("Population total", 100.0, 1e9, float(s["population"]["total"]))

    st.session_state["scenario"] = s

# ---------------------------
# RUN
# ---------------------------
with tabs[1]:
    st.subheader("Run / Step / Run-N")
    s = st.session_state["scenario"]

    run_colA, run_colB, run_colC = st.columns([1, 1, 2])

    with run_colA:
        if st.button("Create / Reset sim"):
            errs = validate_scenario(s)
            if errs:
                st.error("Исправь ошибки в Build:")
                for e in errs:
                    st.write("- " + e)
            else:
                st.session_state["sim"] = Simulation(s)
                st.session_state["history"] = st.session_state["sim"].history
                st.success("Simulation created.")

    with run_colB:
        if st.button("Step"):
            if st.session_state.get("sim") is None:
                st.warning("Сначала Create / Reset sim")
            else:
                st.session_state["sim"].step()
                st.session_state["history"] = st.session_state["sim"].history

    with run_colC:
        batch = st.slider("Run N steps", 1, 200, 10)
        if st.button("Run"):
            if st.session_state.get("sim") is None:
                st.warning("Сначала Create / Reset sim")
            else:
                prog = st.progress(0)
                for i in range(batch):
                    st.session_state["sim"].step()
                    prog.progress((i + 1) / batch)
                st.session_state["history"] = st.session_state["sim"].history

    sim = st.session_state.get("sim")
    if sim is None:
        st.info("Создай симуляцию и делай Step/Run.")
        st.stop()

    hist = st.session_state["history"]

    # top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Firms (now)", f"{len(sim.firms)}")
    m2.metric("Avg wage", f"{hist['avg_wage'][-1] if hist['avg_wage'] else 0:.3f}")
    m3.metric("Avg income pc", f"{hist['avg_income_pc'][-1] if hist['avg_income_pc'] else 0:.3f}")
    m4.metric("Compactness", f"{compactness(sim.mask):.3f}")

    # city map
    st.markdown("### City view")
    show_pop = st.checkbox("Show homes (population heatmap)", value=True)
    show_roads = st.checkbox("Show roads", value=True)
    show_firms = st.checkbox("Show firms", value=True)
    roads_stride = st.slider("Road draw stride (reduce clutter)", 1, 10, 2)

    fig = plot_city(sim, show_population=show_pop, show_roads=show_roads, show_firms=show_firms, roads_stride=roads_stride)
    st.pyplot(fig, clear_figure=True)

    # time series
    st.markdown("### Time series")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Welfare & firms**")
        st.line_chart({
            "avg_wage": hist.get("avg_wage", []),
            "avg_income_pc": hist.get("avg_income_pc", []),
            "avg_profit_margin": hist.get("avg_profit_margin", []),
            "n_firms": hist.get("n_firms", []),
        })
    with c2:
        st.write("**Dynamics**")
        payload = {
            "opens": hist.get("opens", []),
            "closes": hist.get("closes", []),
        }
        if "relocations" in hist:
            payload["relocations"] = hist.get("relocations", [])
        st.line_chart(payload)

    st.write("**Output & input prices**")
    st.line_chart({
        "total_output": hist.get("total_output", []),
        "output_B2C": hist.get("total_output_B2C", []),
        "output_B2B": hist.get("total_output_B2B", []),
        "avg_input_price": hist.get("avg_input_price", []),
    })

    # profitability by industry
    pm = hist.get("profit_margin_by_industry", {})
    if pm:
        st.markdown("### Profitability by industry")
        rows = []
        for iid, series in pm.items():
            last = series[-1] if series else float("nan")
            mean = sum(series) / len(series) if series else float("nan")
            rows.append((iid, last, mean))
        st.table({
            "industry": [r[0] for r in rows],
            "last_margin": [r[1] for r in rows],
            "mean_margin": [r[2] for r in rows],
        })
        st.line_chart({iid: pm[iid] for iid in pm.keys()})

# ---------------------------
# EXPORT
# ---------------------------
with tabs[2]:
    st.subheader("Export / Import YAML")

    s = st.session_state["scenario"]
    yaml_text = dump_yaml(s)

    st.download_button(
        "Download scenario.yaml",
        data=yaml_text.encode("utf-8"),
        file_name="scenario.yaml",
        mime="text/yaml",
    )

    st.markdown("### Import YAML")
    up = st.file_uploader("Upload .yaml", type=["yaml", "yml"])
    if up is not None:
        txt = up.read().decode("utf-8")
        try:
            s2 = load_yaml_text(txt)
            # initialize builder from imported yaml
            st.session_state["base_scenario"] = s2
            st.session_state["scenario"] = default_scenario_like(s2)
            df_i, df_l, df_g = scenario_to_tables(st.session_state["scenario"])
            st.session_state["df_industries"] = df_i
            st.session_state["df_links"] = df_l
            st.session_state["df_groups"] = df_g
            reset_sim()
            st.success("YAML imported into builder.")
        except Exception as e:
            st.error(f"Failed to parse YAML: {e}")

    st.markdown("### Preview YAML")
    st.code(yaml_text, language="yaml")
