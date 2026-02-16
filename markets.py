from __future__ import annotations
import numpy as np

def wage_from_labor_market(Ld: np.ndarray, N: np.ndarray, params: dict) -> np.ndarray:
    # shifted power supply: Ls = N*ell_bar*((w-b)/(w_bar-b))^beta
    b = float(params["b"])
    w_bar = float(params["w_bar"])
    beta = float(params["beta"])
    ell_bar = float(params["ell_bar"])

    cap = N * ell_bar
    # avoid division by 0
    share = np.where(cap > 0, np.clip(Ld / (cap + 1e-9), 0.0, 1.0), 0.0)
    w = b + (w_bar - b) * np.power(share, 1.0 / beta)
    return w

def rent_from_market(Hd: np.ndarray, params: dict) -> np.ndarray:
    # linear supply: Hs = kappa + lambda*R  => R = (Hd-kappa)/lambda
    kappa = float(params["kappa"])
    lam = float(params["lambda"])
    R = (Hd - kappa) / (lam + 1e-9)
    return np.clip(R, 0.0, None)
