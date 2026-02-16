from __future__ import annotations
import numpy as np

def make_mask(w: int, h: int, preset: str, params: dict) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)

    yy, xx = np.mgrid[0:h, 0:w]
    if preset == "circle":
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = float(params.get("radius", min(w, h) * 0.4))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    elif preset == "rectangle":
        rw = int(params.get("width", w))
        rh = int(params.get("height", h))
        x0 = (w - rw) // 2
        y0 = (h - rh) // 2
        mask[:] = False
        mask[y0:y0+rh, x0:x0+rw] = True
    elif preset == "custom":
        # custom будет через внешний файл/ввод позже
        pass
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return mask

def apply_blocked_cells(mask: np.ndarray, blocked: list[list[int]]) -> np.ndarray:
    out = mask.copy()
    for (y, x) in blocked:
        if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
            out[y, x] = False
    return out
