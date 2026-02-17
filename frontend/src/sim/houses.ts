import * as PIXI from "pixi.js";
import type { SimState } from "../../sim/state";

export function createHousesLayer(initialCellSize: number) {
  const container = new PIXI.Container();
  const g = new PIXI.Graphics();
  container.addChild(g);

  let cellSize = initialCellSize;
  let lastHash = -1;
  let tick = 0;

  function setCellSize(v: number) {
    cellSize = v;
    lastHash = -1;
  }

  function render(st: SimState) {
    // update every 5 ticks (adjust)
    tick++;
    if (tick % 5 !== 0 && lastHash !== -1) return;

    // hash = population total (cheap)
    let sum = 0;
    for (let i = 0; i < st.N.length; i += 37) sum += st.N[i]; // sparse sample
    const h = (sum * 1000) | 0;
    if (h === lastHash) return;
    lastHash = h;

    // find max N for normalization
    let maxN = 0;
    for (let i = 0; i < st.N.length; i++) {
      if (st.mask[i]) {
        const v = st.N[i];
        if (v > maxN) maxN = v;
      }
    }
    maxN = Math.max(maxN, 1e-9);

    g.clear();

    // draw as alpha intensity
    for (let y = 0; y < st.h; y++) {
      for (let x = 0; x < st.w; x++) {
        const idx = y * st.w + x;
        if (!st.mask[idx]) continue;

        const v = st.N[idx] / maxN; // 0..1
        const a = Math.min(0.85, 0.08 + 0.75 * v);

        // fill square (homes)
        g.rect(x * cellSize, y * cellSize, cellSize, cellSize);
        g.fill({ color: 0x86efac, alpha: a }); // mint-ish
      }
    }
  }

  return { container, render, setCellSize };
}
