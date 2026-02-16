from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _idx_to_xy(idx: int, w: int) -> tuple[int, int]:
    y = idx // w
    x = idx % w
    return y, x


def adj_to_segments(adj, w: int, h: int):
    """
    Generic: adj[node] -> iterable of neighbor nodes.
    Returns unique undirected segments: ((x1,y1),(x2,y2))
    """
    segs = []
    seen = set()

    # allow dict or list
    items = adj.items() if hasattr(adj, "items") else enumerate(adj)

    for u, neigh in items:
        if neigh is None:
            continue
        for v in neigh:
            a = int(u)
            b = int(v)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            y1, x1 = _idx_to_xy(a, w)
            y2, x2 = _idx_to_xy(b, w)
            # only draw if inside bounds
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                segs.append(((x1, y1), (x2, y2)))
    return segs


def plot_city(sim, show_population=True, show_roads=True, show_firms=True, roads_stride: int = 1):
    """
    Returns matplotlib fig.
    roads_stride: draw every N-th segment to reduce clutter on big grids.
    """
    fig, ax = plt.subplots()

    # base mask background
    base = np.where(sim.mask, 0.15, np.nan)
    ax.imshow(base)

    if show_population:
        pop = np.where(sim.mask, sim.N, np.nan)
        im = ax.imshow(pop, alpha=0.75)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_roads and getattr(sim, "adj", None) is not None:
        segs = adj_to_segments(sim.adj, sim.w, sim.h)
        if roads_stride < 1:
            roads_stride = 1
        for i, ((x1, y1), (x2, y2)) in enumerate(segs):
            if i % roads_stride != 0:
                continue
            ax.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.6)

    if show_firms:
        # group by industry -> different marker
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]
        inds = sorted({f.industry for f in sim.firms})
        for j, iid in enumerate(inds):
            ys = [f.y for f in sim.firms if f.industry == iid]
            xs = [f.x for f in sim.firms if f.industry == iid]
            ax.scatter(xs, ys, s=18, marker=markers[j % len(markers)], label=iid)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax.set_title("City: homes(population) + roads + firms")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
