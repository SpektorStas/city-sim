from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .geometry import make_mask, apply_blocked_cells
from .roads import grid_edges, build_adj, bfs_dist
from .markets import wage_from_labor_market, rent_from_market
from .demand import ces_price_index, ces_expenditure_shares, firm_shares

@dataclass
class Firm:
    industry: str
    y: int
    x: int
    capital: float

class Simulation:
    def __init__(self, scenario: dict):
        self.cfg = scenario
        self.rng = np.random.default_rng(int(scenario.get("seed", 0)))

        g = scenario["grid"]
        self.w = int(g["w"]); self.h = int(g["h"])
        self.walk_radius = float(g.get("walk_radius", 2.0))

        shape = scenario["city_shape"]
        mask = make_mask(self.w, self.h, shape["preset"], shape.get("params", {}))
        mask = apply_blocked_cells(mask, scenario.get("obstacles", {}).get("blocked_cells", []))
        self.mask = mask

        # population
        self.N = self._init_population(scenario["population"])

        # roads
        roads = scenario["roads"]
        edges = grid_edges(self.w, self.h, roads.get("mode", "grid4"))
        self.adj = build_adj(
            self.mask,
            edges,
            keep_prob=float(roads.get("connectivity", 1.0)),
            rng=self.rng,
            blocked_roads=scenario.get("obstacles", {}).get("blocked_roads", [])
        )

        # init firms
        self.industries = {ind["id"]: ind for ind in scenario["industries"]}
        self.firms: list[Firm] = []
        for ind in scenario["industries"]:
            for _ in range(int(ind.get("firms_init", 0))):
                y, x = self._random_cell()
                self.firms.append(Firm(industry=ind["id"], y=y, x=x, capital=float(ind.get("entry_cost", 10.0))))

        # state arrays
        self.wage = np.ones((self.h, self.w), dtype=np.float32)
        self.rent = np.zeros((self.h, self.w), dtype=np.float32)

        self.t = 0
        self.history = {"avg_wage": [], "n_firms": [], "total_output": []}

    def _random_cell(self):
        ys, xs = np.where(self.mask)
        i = self.rng.integers(0, len(ys))
        return int(ys[i]), int(xs[i])

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
            base = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma)).astype(np.float32)
            base[~self.mask] = 0.0
        else:
            raise ValueError(dist)

        s = base.sum()
        if s <= 0:
            return base
        return base * (total / s)

    def step(self, iters_prices: int = 6):
        # fixed-point: wages/rents <-> outputs
        labor_cfg = self.cfg["factor_markets"]["labor_supply"]["params"]
        rent_cfg = self.cfg["factor_markets"]["rent_supply"]["params"]

        # start from current wage/rent
        wage = self.wage.copy()
        rent = self.rent.copy()

        for _ in range(iters_prices):
            # 1) compute firm prices from mc
            firm_price = np.zeros(len(self.firms), dtype=np.float32)
            firm_mc = np.zeros(len(self.firms), dtype=np.float32)
            firm_output = np.zeros(len(self.firms), dtype=np.float32)

            # exogenous input price per industry (single market)
            pin = {ind_id: 1.0 for ind_id in self.industries.keys()}

            # We'll solve pin from D= a*Q after we get Q; for stability do one pass with pin=1
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                mc = float(ind["cbar"]) * (float(wage[f.y, f.x]) ** float(ind["labor_share"])) * (float(rent[f.y, f.x] + 1e-6) ** float(ind["rent_share"])) * (pin[f.industry] ** (1.0 - float(ind["labor_share"]) - float(ind["rent_share"])))
                firm_mc[i] = mc
                sigma_f = float(ind["sigma_firms"])
                mu = sigma_f / (sigma_f - 1.0)
                firm_price[i] = mu * mc

            # 2) demand from households with industry groups (subs/compl)
            Y = wage * 1.0  # ell will be implied by market clearing; MVP: income proportional to wage
            total_output = 0.0

            # precompute road distance from each firm to all cells (BFS per firm)
            n = self.h * self.w
            def idx(y, x): return y * self.w + x
            firm_road = []
            for f in self.firms:
                dist = bfs_dist(self.adj, [idx(f.y, f.x)], n).reshape(self.h, self.w)
                firm_road.append(dist)
            firm_road = np.stack(firm_road, axis=0)  # (F,h,w)

            # Euclidean distances (for walk radius) per firm (cheap)
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            firm_euc = np.zeros((len(self.firms), self.h, self.w), dtype=np.float32)
            for i, f in enumerate(self.firms):
                firm_euc[i] = np.sqrt((yy - f.y)**2 + (xx - f.x)**2).astype(np.float32)

            # delivered prices to households per firm
            delivered = np.zeros_like(firm_euc)
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                tauH = float(ind.get("tau_H", 0.0))
                dH = np.where(firm_euc[i] <= self.walk_radius, firm_euc[i], firm_road[i])
                delivered[i] = firm_price[i] + tauH * dH

            # compute industry price indices per cell
            ind_ids = list(self.industries.keys())
            ind_price_index = {ind_id: np.full((self.h, self.w), np.inf, dtype=np.float32) for ind_id in ind_ids}

            for ind_id in ind_ids:
                firm_idx = [i for i, f in enumerate(self.firms) if f.industry == ind_id]
                if not firm_idx:
                    continue
                # CES price index over firms per cell
                sigma = float(self.industries[ind_id]["sigma_firms"])
                P = np.power(delivered[firm_idx], 1.0 - sigma).sum(axis=0)
                ind_price_index[ind_id] = np.power(P, 1.0 / (1.0 - sigma)).astype(np.float32)

            # allocate budgets across industries using household groups
            hh = self.cfg["households"]["groups"]
            # reset outputs
            firm_output[:] = 0.0
            for group in hh:
                share = float(group["budget_share"])
                sigma_top = float(group["sigma_top"])
                inds = group["industries"]
                weights = np.array(group.get("weights", [1.0]*len(inds)), dtype=np.float32)

                Pvec = np.stack([ind_price_index[iid] for iid in inds], axis=0)  # (G,h,w)
                # expenditure shares per cell
                # shares_g(i) ∝ w_i * P_i^(1-sigma)
                if abs(sigma_top - 1.0) < 1e-6:
                    S = weights[:, None, None] / weights.sum()
                else:
                    num = weights[:, None, None] * np.power(Pvec, 1.0 - sigma_top)
                    den = num.sum(axis=0) + 1e-12
                    S = num / den

                Eg = share * Y  # (h,w)
                for k, ind_id in enumerate(inds):
                    E_ind = Eg * S[k]  # money spent on this industry at cell
                    # now allocate within industry across firms by CES on delivered price
                    firm_idx = [i for i, f in enumerate(self.firms) if f.industry == ind_id]
                    if not firm_idx:
                        continue
                    sigma_f = float(self.industries[ind_id]["sigma_firms"])
                    Pcell = ind_price_index[ind_id]
                    # quantity demanded for each firm: q = (E/P)*(p/P)^(-sigma) * 1/p  (with delivered price)
                    for i in firm_idx:
                        ptilde = delivered[i]
                        q = (E_ind / (Pcell + 1e-12)) * np.power(ptilde / (Pcell + 1e-12), -sigma_f) * (1.0 / (ptilde + 1e-12))
                        firm_output[i] += q.sum()  # aggregate over cells

            total_output = float(firm_output.sum())

            # 3) labor & space demand from outputs => update wages/rents
            Ld = np.zeros((self.h, self.w), dtype=np.float32)
            Hd = np.zeros((self.h, self.w), dtype=np.float32)
            for i, f in enumerate(self.firms):
                ind = self.industries[f.industry]
                Ld[f.y, f.x] += float(ind.get("labor_demand_coeff", 0.02)) * firm_output[i]
                Hd[f.y, f.x] += float(ind.get("space_demand_coeff", 0.01)) * firm_output[i]

            wage = wage_from_labor_market(Ld, self.N, labor_cfg)
            rent = rent_from_market(Hd, rent_cfg)

        # store
        self.wage = wage
        self.rent = rent
        self.t += 1

        self.history["avg_wage"].append(float(np.mean(wage[self.mask])))
        self.history["n_firms"].append(len(self.firms))
        self.history["total_output"].append(float(total_output))

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
        return self.history
