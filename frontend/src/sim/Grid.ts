import * as PIXI from "pixi.js";
import type { SimState } from "../../sim/state";

export function createGridLayer(initialCellSize: number) {
  const container = new PIXI.Container();
  const g = new PIXI.Graphics();
  container.addChild(g);

  let cellSize = initialCellSize;
  let cachedW = -1;
  let cachedH = -1;

  function setCellSize(v: number) {
    cellSize = v;
    cachedW = -1;
    cachedH = -1;
  }

  function render(st: SimState) {
    if (st.w === cachedW && st.h === cachedH) return;

    cachedW = st.w;
    cachedH = st.h;

    g.clear();
    g.alpha = 0.25;

    // draw border + light grid
    g.rect(0, 0, st.w * cellSize, st.h * cellSize);
    g.stroke({ width: 1, color: 0x2a3558 });

    // grid lines (sparse for perf)
    const step = Math.max(1, Math.floor(20 / cellSize)); // fewer lines when zoomed out
    for (let x = 0; x <= st.w; x += step) {
      g.moveTo(x * cellSize, 0);
      g.lineTo(x * cellSize, st.h * cellSize);
    }
    for (let y = 0; y <= st.h; y += step) {
      g.moveTo(0, y * cellSize);
      g.lineTo(st.w * cellSize, y * cellSize);
    }
    g.stroke({ width: 1, color: 0x1b2442 });
  }

  return { container, render, setCellSize };
}
