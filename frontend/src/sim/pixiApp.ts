import * as PIXI from "pixi.js";
import type { SimState } from "../sim/state";
import { createGridLayer } from "./layers/grid";
import { createHousesLayer } from "./layers/houses";
import { createRoadsLayer } from "./layers/roads";
import { createFirmsLayer } from "./layers/firms";

export type PixiHandles = {
  app: PIXI.Application;
  stage: PIXI.Container;
  destroy: () => void;
  render: (st: SimState) => void;
  setViewport: (cellSize: number) => void;
};

export async function createPixiApp(
  host: HTMLDivElement,
  opts?: { cellSize?: number; background?: number }
): Promise<PixiHandles> {
  const cellSize = opts?.cellSize ?? 10;
  const background = opts?.background ?? 0x0b1020;

  const app = new PIXI.Application();
  await app.init({
    background,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
    resizeTo: host,
  });

  host.appendChild(app.canvas);

  const stage = app.stage;

  // layers
  const gridLayer = createGridLayer(cellSize);
  const housesLayer = createHousesLayer(cellSize);
  const roadsLayer = createRoadsLayer(cellSize);
  const firmsLayer = createFirmsLayer(cellSize);

  stage.addChild(gridLayer.container);
  stage.addChild(roadsLayer.container);
  stage.addChild(housesLayer.container);
  stage.addChild(firmsLayer.container);

  // simple zoom by changing cellSize
  function setViewport(newCellSize: number) {
    gridLayer.setCellSize(newCellSize);
    housesLayer.setCellSize(newCellSize);
    roadsLayer.setCellSize(newCellSize);
    firmsLayer.setCellSize(newCellSize);
  }

  function render(st: SimState) {
    gridLayer.render(st);
    housesLayer.render(st);
    roadsLayer.render(st);
    firmsLayer.render(st);
  }

  function destroy() {
    app.destroy(true);
    host.innerHTML = "";
  }

  return { app, stage, destroy, render, setViewport };
}
