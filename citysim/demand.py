from __future__ import annotations
import numpy as np

def ces_price_index(prices: np.ndarray, sigma: float) -> float:
    # prices: (n,) positive
    if len(prices) == 0:
        return np.inf
    if abs(sigma - 1.0) < 1e-6:
        return float(np.exp(np.mean(np.log(prices))))
    p = np.power(prices, 1.0 - sigma).sum()
    return float(np.power(p, 1.0 / (1.0 - sigma)))

def ces_expenditure_shares(P: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    # shares across "goods" given their price indices P and weights
    if abs(sigma - 1.0) < 1e-6:
        s = weights / weights.sum()
        return s
    num = weights * np.power(P, 1.0 - sigma)
    den = num.sum()
    if den <= 0:
        return np.zeros_like(P)
    return num / den

def firm_shares(delivered_prices: np.ndarray, sigma: float) -> np.ndarray:
    # shares across firms within an industry
    if len(delivered_prices) == 0:
        return delivered_prices
    num = np.power(delivered_prices, -sigma)
    den = num.sum()
    return num / (den + 1e-12)
