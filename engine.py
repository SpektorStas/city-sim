from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .geometry import make_mask, apply_blocked_cells
from .roads import grid_edges, build_adj, bfs_dist
from .markets import wage_from_labor_market, rent_from_market


@dataclass
class Firm:
    industry: str
    y: int
    x: int
    capital: float


class Simulation:
    """
    v0.4:
    - v0.3 + non-random entry (location choice) + relocation
    - Entry: choose location maximizing expected discounted profit, enter if disc_profit - entry_cost >= 0
    - Relocation: move if disc_profit_new - relocation_cost > disc_profit_current + relocation_min_gain
    """

    def __init__(self, scenario: dict):
        self.cfg = scenario
        self.rng = np.random.default_rng(int(scenario.get("seed", 0)))

        g = scenario["grid"]
        self.w = int(g["w"])
        self.h = int(g["h"])
        self.walk_radius = float(g.get("walk_radius", 2.0))

        shape = scenario["city_shape"]
        mask = make_mask(self.w, self.h, shape["preset"], shape.get("params", {}))
        mask = apply_blocked_cells(mask, scenario.get("obstacles", {}).get("blocked_cells", []))
        self.mask = mask

        self.N = self._init_population(scenario["population"])

        roads = scenario["roads"]
        edges = grid_edges(self.w, self.h, roads.get("mode", "grid4"))
        self.adj = build_adj(
            self.mask,
            edges,
            keep_prob=float(roads.get("connectivity", 1.0)),
            rng=self.rng,
            blocked_roads=scenario.get("obstacles", {}).get("blocked_roads", []),
        )

        self.industries_list = scenario["industries"]
        self.industries = {ind["id"]: ind for ind in self.industries_list}

        self.firms: list[Firm] = []
        for ind in self.industries_list:
            for _ in range(int(ind.get("firms_init", 0))):
                y, x = self._random_cell()
                self.firms.append(
                    Firm(industry=ind["id"], y=y, x=x, capital=float(ind.get("entry_cost", 10.0)))
                )

        self.wage = np.ones((self.h, self.w), dtype=np.float32)
        self.rent = np.zeros((self.h, self.w), dtype=np.float32)

        self.t = 0
        self.cum_open = 0
        self.cum_close = 0

        self.history = {
            "avg_wage": [],
            "avg_income_pc": [],
            "wage_fund": [],
            "n_firms": [],
            "opens": [],
            "closes": [],
            "cum_open": [],
            "cum_close": [],
            "relocations": [],
            "total_output": [],
            "total_output_B2C": [],
            "total_output_B2B": [],
            "avg_profit_margin": [],
            "avg_input_price": [],
            "profit_margin_by_industry": {},
        }
        for iid in self.industries.keys():
            self.history["profit_margin_by_industry"][iid] = []

    # ---------- utilities ----------
    def _random_cell(self):
        ys, xs = np.where(self.mask)
        i = self.rng.integers(0, len(ys))
        return int(ys[i]), int(xs[i])

    def _sample_cells(self, k: int) -> list[tuple[int, int]]:
        ys, xs = np.where(self.mask)
        if len(ys) == 0:
            return []
        idx = self.rng.choice(len(ys), size=min(k, len(ys)), replace=False)
        return [(int(ys[i]), int(xs[i])) for i in idx]

    def _init_population(self, pcfg: dict) -> np.ndarray:
        total = float(pcfg["total"])
        dist = pcfg.get("distribution", "uniform")
        h, w = self.h, self.w
        base = np.zeros((h, w), dtype=np.float32)

        if dist == "uniform":
            base[self.mask] = 1.0
        elif dist == "gaussian":
            cy, cx = pcfg["params"]["center"]
            sigma = float(pcfg["params"]["sigma"])
            yy, xx = np.mgrid[0:h, 0:w]
            base = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
            base[~self.mask] = 0.0
        else:
            raise ValueError(dist)

        s = base.sum()
        if s <= 0:
            return base
        return base * (total / s)

    def _precompute_road_from_cell(self, y: int, x: int) -> np.ndarray:
        """Road distances from (y,x) to all cells."""
        n = self.h * self.w

        def idx(yy, xx):
            return yy * self.w + xx

        dist = bfs_dist(self.adj, [idx(y, x)], n).reshape(self.h, self.w).astype(np.float32)
        return dist

    def _discount_factor_sum(self, delta: float, horizon: int) -> float:
        if abs(delta - 1.0) < 1e-9:
            return float(horizon)
        return float((1.0 - delta**horizon) / (1.0 - delta))

    def _inputs_of(self, industry_id: str) -> list[dict]:
        return list(self.industries[industry_id].get("inputs", []) or [])

    def _compute_exogenous_input_price(self, total_demand: float, supply: dict) -> float:
        kappa = float(supply["kappa"])
        lam = float(supply["lambda"])
        p = (total_demand - kappa) / (lam + 1e-9)
        return float(max(0.0, p))

    # ---------- expected profit evaluators ----------
    def _mc_at(self, ind: dict, y: int, x: int, wage: np.ndarray, rent: np.ndarray, pin_cell: dict[str, np.ndarray]) -> float:
        labor_share = float(ind["labor_share"])
        rent_share = float(ind["rent_share"])
        cbar = float(ind["cbar"])
        wloc = float(wage[y, x])
        rloc = float(rent[y, x] + 1e-6)
        pinloc = float(pin_cell[ind["id"]][y, x])
        mc = cbar * (wloc ** labor_share) * (rloc ** rent_share) * (pinloc ** (1.0 - labor_share - rent_share))
        return float(mc)

    def _price_from_mc(self, ind: dict, mc: float) -> float:
        sigma_f = float(ind["sigma_firms"])
        mu = sigma_f / (sigma_f - 1.0)
        return float(mu * mc)

    def _expected_b2c_profit_at(
        self,
        iid: str,
        y: int,
        x: int,
        wage: np.ndarray,
        rent: np.ndarray,
        pin_cell: dict[str, np.ndarray],
        # market snapshot (computed in step)
        E_ind_cell: np.ndarray,            # expenditure on industry iid by cell
        P_ind_cell: np.ndarray,            # current industry price index (without entrant)
        existing_firm_prices: list[float], # producer prices of incumbent firms in iid
        existing_firm_locs: list[tuple[int,int]],
        firm_road_cache: dict[tuple[int,int], np.ndarray],  # distances from incumbents (and candidate) to cells
    ) -> tuple[float, float, float]:
        """
        Return (profit, revenue, quantity) for an entrant at (y,x) in B2C industry iid
        under CES competition with incumbents. Uses exact P_new = (P_old^(1-s)+p_c^(1-s))^(1/(1-s)).
        """
        ind = self.industries[iid]
        mc = self._mc_at(ind, y, x, wage, rent, pin_cell)
        p = self._price_from_mc(ind, mc)
        fixed_cost = float(ind.get("fixed_cost", 0.0))

        sigma = float(ind["sigma_firms"])
        tauH = float(ind.get("tau_H", 0.0))

        # distance to cells (walk vs road)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        euc = np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32)
        road = firm_road_cache[(y, x)]
        dH = np.where(euc <= self.walk_radius, euc, road)

        ptilde = p + tauH * dH  # delivered price

        # P_new
        if np.isinf(P_ind_cell).all():
            # no incumbents -> P_new is entrant only
            P_new = ptilde
        else:
            P_new = np.power(np.power(P_ind_cell, 1.0 - sigma) + np.power(ptilde, 1.0 - sigma), 1.0 / (1.0 - sigma))

        # q = (E/P_new)*(ptilde/P_new)^(-sigma) * 1/ptilde
        q_cell = (E_ind_cell / (P_new + 1e-12)) * np.power(ptilde / (P_new + 1e-12), -sigma) * (1.0 / (ptilde + 1e-12))
        Q = float(np.nansum(q_cell))
        revenue = float(p * Q)
        var_cost = float(mc * Q)
        profit = (revenue - var_cost) - fixed_cost  # logistics for B2C: none
        return profit, revenue, Q

    def _expected_b2b_profit_at(
        self,
        sid: str,
        y: int,
        x: int,
        wage: np.ndarray,
        rent: np.ndarray,
        pin_cell: dict[str, np.ndarray],
        M_total_cell: np.ndarray,          # aggregate intermediate demand for supplier industry sid by cell
        incumbent_prices: list[float],
        incumbent_locs: list[tuple[int,int]],
        nu: float,
        tauL: float,
        firm_road_cache: dict[tuple[int,int], np.ndarray],
    ) -> tuple[float, float, float]:
        """
        MVP approximation for B2B location choice:
        - ignore smearing fixed point when choosing location
        - use CES share based on delivered price = p + tauL*dist
        """
        ind = self.industries[sid]
        mc = self._mc_at(ind, y, x, wage, rent, pin_cell)
        p = self._price_from_mc(ind, mc)
        fixed_cost = float(ind.get("fixed_cost", 0.0))

        road = firm_road_cache[(y, x)]
        hatp = p + tauL * road

        if len(incumbent_prices) == 0:
            share = np.ones_like(M_total_cell)
        else:
            # denom: sum incumbents (p_i + tauL*d_i)^(-nu) + entrant
            num_e = np.power(hatp + 1e-12, -nu)
            den = num_e.copy()
            for (pi, loc) in zip(incumbent_prices, incumbent_locs):
                di = firm_road_cache[loc]
                den += np.power(pi + tauL * di + 1e-12, -nu)
            share = num_e / (den + 1e-12)

        Q = float(np.nansum(share * M_total_cell))
        revenue = float(p * Q)
        # approximate logistics cost as tauL*dist*Q (no smearing) for choice stage
        log_cost = float(np.nansum(tauL * road * share * M_total_cell))
        profit = (revenue - mc * Q) - fixed_cost - log_cost
        return profit, revenue, Q

    # ---------- main step ----------
    def step(self, iters_outer: int = 6, iters_b2b: int = 20):
        dyn = self.cfg.get("dynamics", {})
        delta = float(dyn.get("delta", 0.97))
        horizon = int(dyn.get("horizon", 12))
        shock_std = float(dyn.get("shock_std", 0.15))
        allow_reloc = bool(dyn.get("allow_relocation", False))
        reloc_eq_entry = bool(dyn.get("relocation_cost_equals_entry", True))
        reloc_check_prob = float(dyn.get("relocation_check_prob", 0.35))
        reloc_min_gain = float(dyn.get("relocation_min_gain", 0.0))

        entry_attempts = int(dyn.get("entry_attempts_per_industry", 2))
        loc_candidates = int(dyn.get("location_candidates", 40))
        entry_safety = float(dyn.get("entry_margin_safety", 0.0))

        disc_sum = self._discount_factor_sum(delta, horizon)

        labor_cfg = self.cfg["factor_markets"]["labor_supply"]["params"]
        rent_cfg = self.cfg["factor_markets"]["rent_supply"]["params"]

        F = len(self.firms)

        # firm indices by industry
        firms_by_ind: dict[str, list[int]] = {}
        for i, f in enumerate(self.firms):
            firms_by_ind.setdefault(f.industry, []).append(i)

        wage = self.wage.copy()
        rent = self.rent.copy()

        pin_cell = {iid: np.ones((self.h, self.w), dtype=np.float32) for iid in self.industries.keys()}
        flows = {}  # B2B smearing fixed point state

        # caches for profit accounting
        firm_mc = np.zeros(F, dtype=np.float32)
        firm_p = np.zeros(F, dtype=np.float32)
        Q_b2c = np.zeros(F, dtype=np.float32)
        Q_b2b = np.zeros(F, dtype=np.float32)
        logcost = np.zeros(F, dtype=np.float32)

        # We will also collect market snapshot needed for location choice:
        # - P_ind_cell for B2C industries
        # - E_ind_cell (expenditure on each B2C industry by cell)
        P_ind_cell_map: dict[str, np.ndarray] = {}
        E_ind_cell_map: dict[str, np.ndarray] = {}
        # aggregate M_total for each supplier industry sid (for B2B location choice)
        M_total_for_supplier: dict[str, np.ndarray] = {}

        # Road distance cache by location (incumbents + candidates will be added on demand)
        road_cache: dict[tuple[int,int], np.ndarray] = {}

        def get_road(yx: tuple[int,int]) -> np.ndarray:
            if yx not in road_cache:
                road_cache[yx] = self._precompute_road_from_cell(yx[0], yx[1])
            return road_cache[yx]

        # Pre-cache incumbent roads (needed a lot)
        for f in self.firms:
            get_road((f.y, f.x))

        # ---------- OUTER FIXED POINT (like v0.3) ----------
        for _ in range(iters_outer):
            # 1) mc and p
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                mc = self._mc_at(ind, f.y, f.x, wage, rent, pin_cell)
                firm_mc[i] = mc
                firm_p[i] = self._price_from_mc(ind, mc)

            # 2) B2C industry price indices P_ind_cell (with incumbents)
            # delivered_b2c for incumbents
            delivered_b2c = np.full((F, self.h, self.w), np.inf, dtype=np.float32)
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                if ind.get("mode", "B2C") != "B2C":
                    continue
                tauH = float(ind.get("tau_H", 0.0))
                euc = np.sqrt((yy - f.y) ** 2 + (xx - f.x) ** 2).astype(np.float32)
                road = get_road((f.y, f.x))
                dH = np.where(euc <= self.walk_radius, euc, road)
                delivered_b2c[i] = firm_p[i] + tauH * dH

            P_ind_cell_map = {iid: np.full((self.h, self.w), np.inf, dtype=np.float32) for iid in self.industries.keys()}
            for iid, ind in self.industries.items():
                if ind.get("mode", "B2C") != "B2C":
                    continue
                idxs = firms_by_ind.get(iid, [])
                if not idxs:
                    continue
                sigma = float(ind["sigma_firms"])
                P = np.power(delivered_b2c[idxs], 1.0 - sigma).sum(axis=0)
                P_ind_cell_map[iid] = np.power(P, 1.0 / (1.0 - sigma)).astype(np.float32)

            # 3) household incomes and expenditure allocation (compute E_ind_cell_map)
            Y = wage * 1.0
            E_ind_cell_map = {iid: np.zeros((self.h, self.w), dtype=np.float32) for iid in self.industries.keys()}

            # group-level allocation
            for group in self.cfg["households"]["groups"]:
                share = float(group["budget_share"])
                sigma_top = float(group["sigma_top"])
                inds = list(group["industries"])
                weights = np.array(group.get("weights", [1.0] * len(inds)), dtype=np.float32)

                Pvec = np.stack([P_ind_cell_map[iid] for iid in inds], axis=0)  # (G,h,w)

                if abs(sigma_top - 1.0) < 1e-6:
                    S = (weights / weights.sum())[:, None, None]
                else:
                    num = weights[:, None, None] * np.power(Pvec, 1.0 - sigma_top)
                    den = num.sum(axis=0) + 1e-12
                    S = num / den

                Eg = share * Y
                for k, iid in enumerate(inds):
                    E_ind_cell_map[iid] += Eg * S[k]

            # 4) B2C firm quantities
            Q_b2c[:] = 0.0
            for iid, ind in self.industries.items():
                if ind.get("mode", "B2C") != "B2C":
                    continue
                idxs = firms_by_ind.get(iid, [])
                if not idxs:
                    continue
                sigma = float(ind["sigma_firms"])
                Pcell = P_ind_cell_map[iid]
                Ecell = E_ind_cell_map[iid]
                for i in idxs:
                    ptilde = delivered_b2c[i]
                    q = (Ecell / (Pcell + 1e-12)) * np.power(ptilde / (Pcell + 1e-12), -sigma) * (1.0 / (ptilde + 1e-12))
                    Q_b2c[i] += float(q.sum())

            # 5) output by industry&cell (B2C only as buyers for now)
            out_by_ind_cell = {iid: np.zeros((self.h, self.w), dtype=np.float32) for iid in self.industries.keys()}
            for i, f in enumerate(self.firms):
                if self.industries[f.industry].get("mode", "B2C") == "B2C":
                    out_by_ind_cell[f.industry][f.y, f.x] += Q_b2c[i]

            # 6) Exogenous input prices and base pin
            exo_prices = {}
            for iid in self.industries.keys():
                for inp in self._inputs_of(iid):
                    if inp["type"] != "exogenous":
                        continue
                    a = float(inp["a"])
                    total_d = float(a * out_by_ind_cell[iid].sum())
                    exo_prices[(iid, inp["name"])] = self._compute_exogenous_input_price(total_d, inp["supply"])

            for iid in self.industries.keys():
                pin = np.ones((self.h, self.w), dtype=np.float32)
                exo_inps = [inp for inp in self._inputs_of(iid) if inp["type"] == "exogenous"]
                if exo_inps:
                    vals = []
                    weights = []
                    for inp in exo_inps:
                        vals.append(exo_prices[(iid, inp["name"])])
                        weights.append(float(inp["a"]))
                    vals = np.array(vals, dtype=np.float32)
                    weights = np.array(weights, dtype=np.float32)
                    wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
                    gmean = float(np.exp(np.sum((weights / wsum) * np.log(vals + 1e-12))))
                    pin *= gmean
                pin_cell[iid] = pin

            # 7) Endogenous inputs: B2B flows + logistics (as v0.3)
            Q_b2b[:] = 0.0
            logcost[:] = 0.0

            # also build aggregate demand maps per supplier industry (for location choice later)
            M_total_for_supplier = {iid: np.zeros((self.h, self.w), dtype=np.float32) for iid in self.industries.keys()}

            for buyer_iid in self.industries.keys():
                endo_inps = [inp for inp in self._inputs_of(buyer_iid) if inp["type"] == "endogenous"]
                if not endo_inps:
                    continue
                buyer_out_cell = out_by_ind_cell[buyer_iid]

                for inp in endo_inps:
                    sid = inp["from_industry"]
                    suppliers = firms_by_ind.get(sid, [])
                    if not suppliers:
                        continue

                    a = float(inp["a"])
                    nu = float(inp.get("nu", self.industries[sid].get("b2b", {}).get("nu", 4.0)))
                    tauL = float(inp.get("tau_L", self.industries[sid].get("b2b", {}).get("tau_L", 0.08)))
                    eta = float(inp.get("eta", self.industries[sid].get("b2b", {}).get("eta", 0.75)))
                    eps = float(inp.get("eps", self.industries[sid].get("b2b", {}).get("eps", 1e-3)))

                    M = a * buyer_out_cell
                    M_total_for_supplier[sid] += M

                    key = (sid, buyer_iid, inp["name"])
                    if key not in flows:
                        flows[key] = np.zeros((len(suppliers), self.h, self.w), dtype=np.float32)
                        if len(suppliers) > 0:
                            flows[key][:] = (M[None, :, :] / len(suppliers))

                    Q = flows[key]

                    for _k in range(iters_b2b):
                        # delivered includes smearing term (eta)
                        # surcharge = tauL * d * (Q+eps)^(eta-1)
                        d = np.stack([get_road((self.firms[i].y, self.firms[i].x)) for i in suppliers], axis=0)
                        surcharge = tauL * d * np.power(Q + eps, eta - 1.0)
                        hatp = firm_p[suppliers][:, None, None] + surcharge

                        num = np.power(hatp + 1e-12, -nu)
                        den = num.sum(axis=0) + 1e-12
                        share = num / den

                        Q_new = share * M[None, :, :]
                        Q = 0.6 * Q + 0.4 * Q_new

                    flows[key] = Q

                    # supplier outputs + logistics costs
                    sold = Q.reshape(len(suppliers), -1).sum(axis=1).astype(np.float32)
                    for local_i, firm_i in enumerate(suppliers):
                        Q_b2b[firm_i] += float(sold[local_i])
                        d_i = get_road((self.firms[firm_i].y, self.firms[firm_i].x))
                        lc = tauL * d_i * np.power(Q[local_i] + eps, eta)
                        logcost[firm_i] += float(np.nansum(lc))

                    # buyer-side input price index (used in pin)
                    d = np.stack([get_road((self.firms[i].y, self.firms[i].x)) for i in suppliers], axis=0)
                    surcharge = tauL * d * np.power(Q + eps, eta - 1.0)
                    hatp = firm_p[suppliers][:, None, None] + surcharge
                    P_in = np.power(np.power(hatp + 1e-12, 1.0 - nu).sum(axis=0) + 1e-12, 1.0 / (1.0 - nu)).astype(np.float32)

                    sum_a = float(sum(float(x["a"]) for x in endo_inps))
                    wgt = a / (sum_a + 1e-12)
                    pin_cell[buyer_iid] *= np.power(P_in, wgt)

            # 8) factor markets update
            Ld = np.zeros((self.h, self.w), dtype=np.float32)
            Hd = np.zeros((self.h, self.w), dtype=np.float32)
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                out_i = Q_b2c[i] if ind.get("mode", "B2C") == "B2C" else Q_b2b[i]
                Ld[f.y, f.x] += float(ind.get("labor_demand_coeff", 0.02)) * out_i
                Hd[f.y, f.x] += float(ind.get("space_demand_coeff", 0.01)) * out_i

            wage = wage_from_labor_market(Ld, self.N, labor_cfg)
            rent = rent_from_market(Hd, rent_cfg)

        # ---------- realized accounting ----------
        # household income proxies
        ell_bar = float(labor_cfg.get("ell_bar", 1.0))
        Lcap = self.N * ell_bar
        b = float(labor_cfg["b"])
        w_bar = float(labor_cfg["w_bar"])
        beta = float(labor_cfg["beta"])
        share = np.clip((wage - b) / (w_bar - b + 1e-9), 0.0, 1.0)
        employed = Lcap * np.power(share, beta)
        wage_fund = float(np.sum(wage * employed))
        avg_income_pc = float(wage_fund / (np.sum(self.N) + 1e-9))

        # firm profits
        firm_rev = np.zeros(F, dtype=np.float32)
        firm_profit = np.zeros(F, dtype=np.float32)
        for i, f in enumerate(self.firms):
            ind = self.industries[f.industry]
            fixed_cost = float(ind.get("fixed_cost", 0.0))
            out_i = Q_b2c[i] if ind.get("mode", "B2C") == "B2C" else Q_b2b[i]
            rev = float(firm_p[i] * out_i)
            firm_rev[i] = rev
            var_cost = float(firm_mc[i] * out_i)
            shock = float(self.rng.normal(0.0, shock_std * max(1.0, fixed_cost)))
            firm_profit[i] = (rev - var_cost) - fixed_cost - float(logcost[i]) + shock

        # industry margins
        margin_by_ind = {}
        for iid in self.industries.keys():
            idxs = [i for i, f in enumerate(self.firms) if f.industry == iid]
            if not idxs:
                margin_by_ind[iid] = np.nan
                continue
            rev = float(np.sum(firm_rev[idxs]))
            prof = float(np.sum(firm_profit[idxs]))
            margin_by_ind[iid] = float(prof / (rev + 1e-9))

        avg_profit_margin = float(np.nanmean(list(margin_by_ind.values()))) if margin_by_ind else float("nan")
        avg_pin = float(np.mean([pin_cell[iid][self.mask].mean() for iid in pin_cell.keys()]))

        # ---------- EXIT ----------
        closes = 0
        survivors: list[Firm] = []
        surv_profit: list[float] = []
        surv_rev: list[float] = []

        for i, f in enumerate(self.firms):
            pi = float(firm_profit[i])
            disc_pi = pi * disc_sum
            if (pi < 0.0) and (disc_pi < 0.0):
                closes += 1
                continue
            survivors.append(f)
            surv_profit.append(pi)
            surv_rev.append(float(firm_rev[i]))

        self.firms = survivors

        # ---------- RELOCATION ----------
        relocations = 0
        if allow_reloc and len(self.firms) > 0:
            # rebuild indices and snapshot incumbent lists
            firms_by_ind = {}
            for f in self.firms:
                firms_by_ind.setdefault(f.industry, []).append(f)

            candidates = self._sample_cells(loc_candidates)

            for f in self.firms:
                if self.rng.random() > reloc_check_prob:
                    continue

                ind = self.industries[f.industry]
                reloc_cost = float(ind.get("entry_cost", 0.0)) if reloc_eq_entry else float(ind.get("relocation_cost", 0.0))

                # compute current expected profit proxy at current location (no shock)
                cur_pi_exp = None
                if ind.get("mode", "B2C") == "B2C":
                    Pcell = P_ind_cell_map[f.industry]
                    Ecell = E_ind_cell_map[f.industry]
                    incumb = [(ff.y, ff.x) for ff in self.firms if ff.industry == f.industry and not (ff is f)]
                    inc_prices = []
                    for ff in self.firms:
                        if ff.industry == f.industry and not (ff is f):
                            mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                            inc_prices.append(self._price_from_mc(ind, mc0))
                            get_road((ff.y, ff.x))
                    get_road((f.y, f.x))
                    pi, _, _ = self._expected_b2c_profit_at(
                        f.industry, f.y, f.x, wage, rent, pin_cell,
                        Ecell, Pcell, inc_prices, incumb, road_cache
                    )
                    cur_pi_exp = float(pi)
                else:
                    # supplier
                    M = M_total_for_supplier.get(f.industry, np.zeros((self.h, self.w), dtype=np.float32))
                    b2b = ind.get("b2b", {})
                    nu = float(b2b.get("nu", 4.0))
                    tauL = float(b2b.get("tau_L", 0.08))
                    incumb = [(ff.y, ff.x) for ff in self.firms if ff.industry == f.industry and not (ff is f)]
                    inc_prices = []
                    for ff in self.firms:
                        if ff.industry == f.industry and not (ff is f):
                            mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                            inc_prices.append(self._price_from_mc(ind, mc0))
                            get_road((ff.y, ff.x))
                    get_road((f.y, f.x))
                    pi, _, _ = self._expected_b2b_profit_at(
                        f.industry, f.y, f.x, wage, rent, pin_cell,
                        M, inc_prices, incumb, nu, tauL, road_cache
                    )
                    cur_pi_exp = float(pi)

                cur_disc = cur_pi_exp * disc_sum

                # search best candidate
                best = (f.y, f.x)
                best_disc = cur_disc

                for (yyc, xxc) in candidates:
                    get_road((yyc, xxc))
                    if ind.get("mode", "B2C") == "B2C":
                        Pcell = P_ind_cell_map[f.industry]
                        Ecell = E_ind_cell_map[f.industry]
                        incumb = [(ff.y, ff.x) for ff in self.firms if ff.industry == f.industry and not (ff is f)]
                        inc_prices = []
                        for ff in self.firms:
                            if ff.industry == f.industry and not (ff is f):
                                mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                                inc_prices.append(self._price_from_mc(ind, mc0))
                        pi, _, _ = self._expected_b2c_profit_at(
                            f.industry, yyc, xxc, wage, rent, pin_cell,
                            Ecell, Pcell, inc_prices, incumb, road_cache
                        )
                    else:
                        M = M_total_for_supplier.get(f.industry, np.zeros((self.h, self.w), dtype=np.float32))
                        b2b = ind.get("b2b", {})
                        nu = float(b2b.get("nu", 4.0))
                        tauL = float(b2b.get("tau_L", 0.08))
                        incumb = [(ff.y, ff.x) for ff in self.firms if ff.industry == f.industry and not (ff is f)]
                        inc_prices = []
                        for ff in self.firms:
                            if ff.industry == f.industry and not (ff is f):
                                mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                                inc_prices.append(self._price_from_mc(ind, mc0))
                        pi, _, _ = self._expected_b2b_profit_at(
                            f.industry, yyc, xxc, wage, rent, pin_cell,
                            M, inc_prices, incumb, nu, tauL, road_cache
                        )

                    disc = float(pi) * disc_sum - reloc_cost
                    if disc > best_disc + reloc_min_gain:
                        best_disc = disc
                        best = (yyc, xxc)

                if best != (f.y, f.x):
                    f.y, f.x = best
                    relocations += 1

        # ---------- ENTRY (location choice) ----------
        opens = 0
        candidates = self._sample_cells(loc_candidates)

        for ind in self.industries_list:
            iid = ind["id"]
            entry_cost = float(ind.get("entry_cost", 0.0))

            for _ in range(entry_attempts):
                # pick best location by expected discounted profit
                best_loc = None
                best_disc = -np.inf

                if ind.get("mode", "B2C") == "B2C":
                    Pcell = P_ind_cell_map.get(iid, np.full((self.h, self.w), np.inf, dtype=np.float32))
                    Ecell = E_ind_cell_map.get(iid, np.zeros((self.h, self.w), dtype=np.float32))

                    incumb = [(f.y, f.x) for f in self.firms if f.industry == iid]
                    inc_prices = []
                    for ff in self.firms:
                        if ff.industry == iid:
                            mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                            inc_prices.append(self._price_from_mc(ind, mc0))
                            get_road((ff.y, ff.x))

                    for (yyc, xxc) in candidates:
                        get_road((yyc, xxc))
                        pi, _, _ = self._expected_b2c_profit_at(
                            iid, yyc, xxc, wage, rent, pin_cell,
                            Ecell, Pcell, inc_prices, incumb, road_cache
                        )
                        disc = float(pi) * disc_sum
                        if disc > best_disc:
                            best_disc = disc
                            best_loc = (yyc, xxc)

                else:
                    # B2B supplier: use aggregate input demand map
                    M = M_total_for_supplier.get(iid, np.zeros((self.h, self.w), dtype=np.float32))
                    b2b = ind.get("b2b", {})
                    nu = float(b2b.get("nu", 4.0))
                    tauL = float(b2b.get("tau_L", 0.08))

                    incumb = [(f.y, f.x) for f in self.firms if f.industry == iid]
                    inc_prices = []
                    for ff in self.firms:
                        if ff.industry == iid:
                            mc0 = self._mc_at(ind, ff.y, ff.x, wage, rent, pin_cell)
                            inc_prices.append(self._price_from_mc(ind, mc0))
                            get_road((ff.y, ff.x))

                    for (yyc, xxc) in candidates:
                        get_road((yyc, xxc))
                        pi, _, _ = self._expected_b2b_profit_at(
                            iid, yyc, xxc, wage, rent, pin_cell,
                            M, inc_prices, incumb, nu, tauL, road_cache
                        )
                        disc = float(pi) * disc_sum
                        if disc > best_disc:
                            best_disc = disc
                            best_loc = (yyc, xxc)

                if best_loc is None:
                    continue

                # entry condition (with optional safety margin)
                if best_disc >= entry_cost * (1.0 + entry_safety):
                    self.firms.append(Firm(industry=iid, y=best_loc[0], x=best_loc[1], capital=entry_cost))
                    opens += 1

        # ---------- COMMIT ----------
        self.wage = wage
        self.rent = rent
        self.t += 1
        self.cum_open += opens
        self.cum_close += closes

        # totals (for charts)
        total_output = float(np.sum(Q_b2c) + np.sum(Q_b2b))
        out_b2c = float(np.sum(Q_b2c))
        out_b2b = float(np.sum(Q_b2b))

        self.history["avg_wage"].append(float(np.mean(self.wage[self.mask])))
        self.history["avg_income_pc"].append(avg_income_pc)
        self.history["wage_fund"].append(wage_fund)
        self.history["n_firms"].append(len(self.firms))
        self.history["opens"].append(opens)
        self.history["closes"].append(closes)
        self.history["cum_open"].append(self.cum_open)
        self.history["cum_close"].append(self.cum_close)
        self.history["relocations"].append(relocations)

        self.history["total_output"].append(total_output)
        self.history["total_output_B2C"].append(out_b2c)
        self.history["total_output_B2B"].append(out_b2b)
        self.history["avg_profit_margin"].append(avg_profit_margin)
        self.history["avg_input_price"].append(avg_pin)

        for iid in self.industries.keys():
            self.history["profit_margin_by_industry"][iid].append(float(margin_by_ind.get(iid, np.nan)))

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
        return self.history
