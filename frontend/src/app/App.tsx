import { useMemo, useRef, useState } from "react";
import { PixiView } from "./PixiView";
import { initSim, step } from "../sim/engine";
import type { Scenario } from "../sim/types";
import type { SimState } from "../sim/state";

const demoScenario: Scenario = {
  grid: { w: 60, h: 40, walkRadius: 2 },
  cityShape: { preset: "rectangle", params: {} },
  roads: { mode: "grid4", connectivity: 1.0 },
  obstacles: { blockedCells: [] },
  population: { total: 2_000_000, distribution: "center" },
  industries: [
    { id: "retail", mode: "B2C", entryCost: 10, fixedCost: 0.2, laborShare: 1.0, rentShare: 1.0, baseCost: 1.0, priceMarkup: 0.25 },
    { id: "cafe", mode: "B2C", entryCost: 12, fixedCost: 0.25, laborShare: 1.0, rentShare: 1.0, baseCost: 1.2, priceMarkup: 0.3 },
  ],
  links: [],
  households: {
    groups: [
      { name: "all", budgetShare: 1.0, industries: ["retail", "cafe"], weights: [1.0, 0.9], sigmaTop: 2.0, tauH: 0.2 } as any,
    ],
  },
  dynamics: { shockStd: 0.05, exitGrace: 10, exitThreshold: 0, relocateThreshold: 0.05, candidatesK: 25 },
};

export default function App() {
  const [speed, setSpeed] = useState(1);        // 1..20
  const [playing, setPlaying] = useState(true);
  const [cellSize, setCellSize] = useState(12);

  const scenario = useMemo(() => demoScenario, []);
  const [sim, setSim] = useState<SimState>(() => initSim(scenario));

  const accRef = useRef(0);
  const lastRef = useRef<number | null>(null);

  // animation loop (React state update throttled)
  const frameRef = useRef<number | null>(null);

  function reset() {
    setSim(initSim(scenario));
  }

  function tick(ts: number) {
    if (lastRef.current == null) lastRef.current = ts;
    const dt = (ts - lastRef.current) / 1000;
    lastRef.current = ts;

    if (playing) {
      accRef.current += dt;

      const simStep = 0.1; // 10 steps/sec
      const maxStepsPerFrame = 8;

      let stepsDone = 0;
      while (accRef.current >= simStep && stepsDone < maxStepsPerFrame) {
        // IMPORTANT: we mutate a copy reference for perf; then setState once below
        // We'll do shallow clone to trigger React update:
        const next = { ...sim, firms: [...sim.firms], metrics: { ...sim.metrics } };
        step(scenario, next, simStep);
        setSim(next);
        accRef.current -= simStep;
        stepsDone++;
      }
    }

    frameRef.current = requestAnimationFrame(tick);
  }

  // start loop once
  if (frameRef.current == null) {
    frameRef.current = requestAnimationFrame(tick);
  }

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        <button onClick={() => setPlaying(p => !p)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => {
          // single step
          const next = { ...sim, firms: [...sim.firms], metrics: { ...sim.metrics } };
          step(scenario, next, 0.1);
          setSim(next);
        }}>Step</button>
        <button onClick={reset}>Reset</button>

        <label style={{ marginLeft: 12 }}>
          Speed
          <input
            type="range"
            min={1}
            max={20}
            value={speed}
            onChange={(e) => setSpeed(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8 }}
          />
        </label>

        <label style={{ marginLeft: 12 }}>
          Zoom
          <input
            type="range"
            min={6}
            max={20}
            value={cellSize}
            onChange={(e) => setCellSize(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8 }}
          />
        </label>

        <div style={{ marginLeft: "auto", opacity: 0.8 }}>
          Firms: {sim.firms.length} | t={sim.t}
        </div>
      </div>

      <PixiView state={sim} cellSize={cellSize} />
    </div>
  );
}
