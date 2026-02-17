import * as PIXI from "pixi.js";
import type { SimState } from "../../sim/state";

const COLORS = [
  0xfca5a5, 0xfdba74, 0xfde047, 0x86efac, 0x93c5fd,
  0xc4b5fd, 0xf0abfc, 0xa7f3d0, 0xfecaca, 0xbfdbfe
];

export function createFirmsLayer(initialCellSize: number) {
  const container = new PIXI.Container();

  let cellSize = initialCellSize;

  // pool for performance
  const pool: PIXI.Graphics[] = [];

  function setCellSize(v: number) {
    cellSize = v;
  }

  function getDot(i: number) {
    while (pool.length <= i) {
      const g = new PIXI.Graphics();
      container.addChild(g);
      pool.push(g);
    }
    return pool[i];
  }

  function render(st: SimState) {
    const inds = Array.from(new Set(st.firms.map(f => f.industry))).sort();
    const indIndex = new Map<string, number>();
    inds.forEach((id, i) => indIndex.set(id, i));

    const r = Math.max(2, cellSize * 0.28);

    for (let i = 0; i < st.firms.length; i++) {
      const f = st.firms[i];
      const g = getDot(i);

      const c = COLORS[(indIndex.get(f.industry) ?? 0) % COLORS.length];

      const cx = f.x * cellSize + cellSize * 0.5;
      const cy = f.y * cellSize + cellSize * 0.5;

      g.clear();
      g.circle(cx, cy, r);
      g.fill({ color: c, alpha: 0.95 });
      g.stroke({ width: Math.max(1, r * 0.25), color: 0x0b1020, alpha: 0.9 });
    }

    // hide unused pooled dots
    for (let i = st.firms.length; i < pool.length; i++) {
      pool[i].visible = false;
    }
    for (let i = 0; i < st.firms.length; i++) {
      pool[i].visible = true;
    }
  }

  return { container, render, setCellSize };
}
