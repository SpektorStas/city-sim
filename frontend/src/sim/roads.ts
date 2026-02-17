import * as PIXI from "pixi.js";
import type { SimState } from "../../sim/state";

function idxToXY(idx: number, w: number) {
  const y = Math.floor(idx / w);
  const x = idx % w;
  return { x, y };
}

export function createRoadsLayer(initialCellSize: number) {
  const container = new PIXI.Container();
  const g = new PIXI.Graphics();
  container.addChild(g);

  let cellSize = initialCellSize;
  let cached = "";

  function setCellSize(v: number) {
    cellSize = v;
    cached = "";
  }

  function render(st: SimState) {
    const adj: any = (st as any).adj;
    if (!adj) return;

    // simple cache key: size + firms count (roads static usually)
    const key = `${st.w}x${st.h}`;
    if (key === cached) return;
    cached = key;

    g.clear();
    g.alpha = 0.8;

    const items = typeof adj.entries === "function" ? adj.entries() : Object.entries(adj);

    const seen = new Set<string>();
    for (const [uRaw, neigh] of items as any) {
      const u = typeof uRaw === "string" ? parseInt(uRaw, 10) : (uRaw as number);
      if (!neigh) continue;

      for (const vRaw of neigh as any) {
        const v = typeof vRaw === "string" ? parseInt(vRaw, 10) : (vRaw as number);
        if (v === u) continue;
        const a = Math.min(u, v), b = Math.max(u, v);
        const sk = `${a}-${b}`;
        if (seen.has(sk)) continue;
        seen.add(sk);

        const p1 = idxToXY(a, st.w);
        const p2 = idxToXY(b, st.w);

        // center of cell
        const x1 = p1.x * cellSize + cellSize * 0.5;
        const y1 = p1.y * cellSize + cellSize * 0.5;
        const x2 = p2.x * cellSize + cellSize * 0.5;
        const y2 = p2.y * cellSize + cellSize * 0.5;

        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
      }
    }

    g.stroke({ width: Math.max(1, cellSize * 0.12), color: 0x93c5fd, alpha: 0.45 });
  }

  return { container, render, setCellSize };
}
