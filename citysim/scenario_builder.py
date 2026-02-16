from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


# ---------------------------
# Helpers
# ---------------------------

def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str):
            x = x.strip()
            # allow "1e-3" strings
            return float(x)
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _csv_list(s: str) -> List[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    s = str(s)
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _csv_floats(s: str) -> List[float]:
    if s is None:
        return []
    if isinstance(s, list):
        return [_to_float(x) for x in s]
    parts = _csv_list(s)
    return [_to_float(p) for p in parts]


def dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_yaml_text(text: str) -> dict:
    return yaml.safe_load(text)


# ---------------------------
# Default scenario (minimal)
# ---------------------------

def default_scenario_like(demo: dict) -> dict:
    """
    Start from existing demo_v02.yaml structure:
    keep grid/city_shape/roads/population/factor_markets/dynamics as-is,
    but user can edit industries/links/households in UI.
    """
    return copy.deepcopy(demo)


# ---------------------------
# Tables conversion
# ---------------------------

INDUSTRY_COLS = [
    "id", "mode", "sigma_firms",
    "cbar", "labor_share", "rent_share",
    "entry_cost", "fixed_cost", "firms_init",
    "labor_demand_coeff", "space_demand_coeff",
    "tau_H",  # B2C only (optional)
    # b2b defaults (optional, used as fallback for B2B firms)
    "b2b_nu", "b2b_tau_L", "b2b_eta", "b2b_eps",
]

LINK_COLS = [
    "buyer", "supplier", "name",
    "a", "nu", "tau_L", "eta", "eps"
]

GROUP_COLS = [
    "name", "budget_share", "sigma_top", "industries_csv", "weights_csv"
]


def scenario_to_tables(s: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # -------- industries table --------
    inds_rows = []
    for ind in s.get("industries", []):
        b2b = ind.get("b2b", {}) or {}
        inds_rows.append({
            "id": ind.get("id", ""),
            "mode": ind.get("mode", "B2C"),
            "sigma_firms": _to_float(ind.get("sigma_firms", 6.0)),
            "cbar": _to_float(ind.get("cbar", 1.0)),
            "labor_share": _to_float(ind.get("labor_share", 0.5)),
            "rent_share": _to_float(ind.get("rent_share", 0.1)),
            "entry_cost": _to_float(ind.get("entry_cost", 10.0)),
            "fixed_cost": _to_float(ind.get("fixed_cost", 0.0)),
            "firms_init": _to_int(ind.get("firms_init", 0)),
            "labor_demand_coeff": _to_float(ind.get("labor_demand_coeff", 0.02)),
            "space_demand_coeff": _to_float(ind.get("space_demand_coeff", 0.01)),
            "tau_H": _to_float(ind.get("tau_H", 0.0)),  # safe even if absent
            "b2b_nu": _to_float(b2b.get("nu", 4.0)),
            "b2b_tau_L": _to_float(b2b.get("tau_L", 0.08)),
            "b2b_eta": _to_float(b2b.get("eta", 0.75)),
            "b2b_eps": str(b2b.get("eps", "1e-3")),
        })

    df_inds = pd.DataFrame(inds_rows, columns=INDUSTRY_COLS)

    # -------- links table (endogenous inputs) --------
    link_rows = []
    for buyer in s.get("industries", []):
        buyer_id = buyer.get("id")
        for inp in buyer.get("inputs", []) or []:
            if inp.get("type") != "endogenous":
                continue
            link_rows.append({
                "buyer": buyer_id,
                "supplier": inp.get("from_industry", ""),
                "name": inp.get("name", "input"),
                "a": _to_float(inp.get("a", 0.2)),
                "nu": _to_float(inp.get("nu", buyer.get("b2b", {}).get("nu", 4.0))),
                "tau_L": _to_float(inp.get("tau_L", buyer.get("b2b", {}).get("tau_L", 0.08))),
                "eta": _to_float(inp.get("eta", buyer.get("b2b", {}).get("eta", 0.75))),
                "eps": str(inp.get("eps", "1e-3")),
            })
    df_links = pd.DataFrame(link_rows, columns=LINK_COLS)

    # -------- households groups table --------
    groups_rows = []
    for g in (s.get("households", {}) or {}).get("groups", []) or []:
        groups_rows.append({
            "name": g.get("name", "group"),
            "budget_share": _to_float(g.get("budget_share", 1.0)),
            "sigma_top": _to_float(g.get("sigma_top", 2.0)),
            "industries_csv": ", ".join(list(g.get("industries", []) or [])),
            "weights_csv": ", ".join([str(x) for x in (g.get("weights", []) or [])]),
        })
    df_groups = pd.DataFrame(groups_rows, columns=GROUP_COLS)

    return df_inds, df_links, df_groups


def tables_to_scenario(base: dict, df_inds: pd.DataFrame, df_links: pd.DataFrame, df_groups: pd.DataFrame) -> dict:
    """
    Build scenario dict back:
    - industries: from df_inds + keep existing EXOGENOUS inputs from base by industry id
    - endogenous inputs: rebuilt from df_links (buyer -> inputs[])
    - households.groups: rebuilt from df_groups
    """
    s = copy.deepcopy(base)

    # map existing exogenous inputs from base
    base_exo_inputs: Dict[str, List[dict]] = {}
    for ind in base.get("industries", []) or []:
        iid = ind.get("id")
        exo = [inp for inp in (ind.get("inputs", []) or []) if inp.get("type") == "exogenous"]
        base_exo_inputs[iid] = copy.deepcopy(exo)

    # build industries dict
    industries: List[dict] = []
    ids = []

    for _, r in df_inds.iterrows():
        iid = str(r.get("id", "")).strip()
        if not iid:
            continue
        ids.append(iid)

        mode = str(r.get("mode", "B2C")).strip() or "B2C"
        ind = {
            "id": iid,
            "mode": mode,
            "sigma_firms": _to_float(r.get("sigma_firms", 6.0)),
            "cbar": _to_float(r.get("cbar", 1.0)),
            "labor_share": _to_float(r.get("labor_share", 0.5)),
            "rent_share": _to_float(r.get("rent_share", 0.1)),
            "entry_cost": _to_float(r.get("entry_cost", 10.0)),
            "fixed_cost": _to_float(r.get("fixed_cost", 0.0)),
            "firms_init": _to_int(r.get("firms_init", 0)),
            "labor_demand_coeff": _to_float(r.get("labor_demand_coeff", 0.02)),
            "space_demand_coeff": _to_float(r.get("space_demand_coeff", 0.01)),
        }

        tauH = _to_float(r.get("tau_H", 0.0))
        if tauH > 0:
            ind["tau_H"] = tauH

        # b2b defaults
        ind["b2b"] = {
            "nu": _to_float(r.get("b2b_nu", 4.0)),
            "tau_L": _to_float(r.get("b2b_tau_L", 0.08)),
            "eta": _to_float(r.get("b2b_eta", 0.75)),
            "eps": str(r.get("b2b_eps", "1e-3")),
        }

        # inputs = exogenous kept + endogenous will be appended below
        ind["inputs"] = base_exo_inputs.get(iid, [])

        industries.append(ind)

    # attach endogenous inputs from links
    links = df_links.copy()
    if len(links) > 0:
        # normalize
        for c in LINK_COLS:
            if c not in links.columns:
                links[c] = ""
        links["buyer"] = links["buyer"].astype(str).str.strip()
        links["supplier"] = links["supplier"].astype(str).str.strip()
        links["name"] = links["name"].astype(str).str.strip()

        by_buyer = {}
        for _, r in links.iterrows():
            buyer = str(r.get("buyer", "")).strip()
            supplier = str(r.get("supplier", "")).strip()
            name = str(r.get("name", "input")).strip() or "input"
            if not buyer or not supplier:
                continue

            inp = {
                "name": name,
                "type": "endogenous",
                "from_industry": supplier,
                "a": _to_float(r.get("a", 0.2)),
                "nu": _to_float(r.get("nu", 4.0)),
                "tau_L": _to_float(r.get("tau_L", 0.08)),
                "eta": _to_float(r.get("eta", 0.75)),
                "eps": str(r.get("eps", "1e-3")),
            }
            by_buyer.setdefault(buyer, []).append(inp)

        # add to industries list
        for ind in industries:
            iid = ind["id"]
            ind["inputs"] = (ind.get("inputs", []) or []) + by_buyer.get(iid, [])

    # households groups
    groups: List[dict] = []
    df_groups = df_groups.copy()
    for _, r in df_groups.iterrows():
        name = str(r.get("name", "group")).strip() or "group"
        budget_share = _to_float(r.get("budget_share", 0.0))
        sigma_top = _to_float(r.get("sigma_top", 2.0))
        inds_csv = str(r.get("industries_csv", "")).strip()
        w_csv = str(r.get("weights_csv", "")).strip()

        inds = _csv_list(inds_csv)
        weights = _csv_floats(w_csv)
        if len(weights) != len(inds):
            # fallback equal weights
            weights = [1.0] * len(inds)

        if inds:
            groups.append({
                "name": name,
                "budget_share": budget_share,
                "sigma_top": sigma_top,
                "industries": inds,
                "weights": weights,
            })

    s["industries"] = industries
    s["households"] = {"groups": groups}

    return s


# ---------------------------
# Validation
# ---------------------------

def validate_scenario(s: dict) -> List[str]:
    errs = []
    inds = [ind.get("id") for ind in (s.get("industries", []) or [])]
    inds = [str(x).strip() for x in inds if str(x).strip()]
    if len(set(inds)) != len(inds):
        errs.append("Industry id должны быть уникальны (есть дубликаты).")

    ind_set = set(inds)

    # validate links (endogenous inputs)
    for ind in (s.get("industries", []) or []):
        buyer = ind.get("id")
        for inp in (ind.get("inputs", []) or []):
            if inp.get("type") != "endogenous":
                continue
            sup = str(inp.get("from_industry", "")).strip()
            if sup and sup not in ind_set:
                errs.append(f"Связь buyer={buyer} ссылается на несуществующую supplier отрасль: {sup}")

            a = _to_float(inp.get("a", 0.0))
            if a < 0:
                errs.append(f"Связь buyer={buyer} имеет отрицательный a={a}")

    # validate households
    groups = (s.get("households", {}) or {}).get("groups", []) or []
    bs = [_to_float(g.get("budget_share", 0.0)) for g in groups]
    sumb = sum(bs)
    if groups and not (0.99 <= sumb <= 1.01):
        errs.append(f"Сумма budget_share по группам должна быть ~1. Сейчас: {sumb:.4f}")

    for g in groups:
        inds_g = g.get("industries", []) or []
        for iid in inds_g:
            if str(iid).strip() not in ind_set:
                errs.append(f"Household group '{g.get('name')}' ссылается на несуществующую отрасль: {iid}")

        w = g.get("weights", []) or []
        if len(w) != len(inds_g):
            errs.append(f"Household group '{g.get('name')}' weights не совпадает по длине с industries.")

    return errs
