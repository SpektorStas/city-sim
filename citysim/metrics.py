from __future__ import annotations
import numpy as np

def compactness(mask: np.ndarray) -> float:
    # простая метрика: 4*pi*Area/Perimeter^2 (нормирована, 1 — круг)
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    h, w = mask.shape
    per = 0.0
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                y2, x2 = y+dy, x+dx
                if not (0 <= y2 < h and 0 <= x2 < w) or not mask[y2, x2]:
                    per += 1.0
    return float(4.0 * np.pi * area / (per * per + 1e-9))
