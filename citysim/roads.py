from __future__ import annotations
import numpy as np
from collections import deque

def grid_edges(w: int, h: int, mode: str) -> list[tuple[int,int,int,int]]:
    edges = []
    nbrs = [(1,0),(-1,0),(0,1),(0,-1)]
    if mode == "grid8":
        nbrs += [(1,1),(1,-1),(-1,1),(-1,-1)]
    for y in range(h):
        for x in range(w):
            for dy, dx in nbrs:
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    if (y, x) < (y2, x2):
                        edges.append((y, x, y2, x2))
    return edges

def build_adj(mask: np.ndarray, edges: list[tuple[int,int,int,int]], keep_prob: float, rng: np.random.Generator,
              blocked_roads: list[list[list[int]]] | None = None) -> list[list[int]]:
    h, w = mask.shape
    n = h * w
    adj = [[] for _ in range(n)]

    blocked_set = set()
    if blocked_roads:
        for a, b in blocked_roads:
            ay, ax = a
            by, bx = b
            blocked_set.add(((ay, ax), (by, bx)))
            blocked_set.add(((by, bx), (ay, ax)))

    def idx(y, x): return y * w + x

    for (y1, x1, y2, x2) in edges:
        if not (mask[y1, x1] and mask[y2, x2]):
            continue
        if ((y1, x1), (y2, x2)) in blocked_set:
            continue
        if rng.random() > keep_prob:
            continue
        i, j = idx(y1, x1), idx(y2, x2)
        adj[i].append(j)
        adj[j].append(i)

    return adj

def bfs_dist(adj: list[list[int]], sources: list[int], n: int) -> np.ndarray:
    dist = np.full(n, np.inf, dtype=np.float32)
    q = deque()
    for s in sources:
        dist[s] = 0.0
        q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = du + 1.0
                q.append(v)
    return dist
