import { useEffect, useRef } from "react";
import type { SimState } from "../sim/state";
import { createPixiApp, type PixiHandles } from "../render/pixiApp";

export function PixiView({ state, cellSize }: { state: SimState; cellSize: number }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const pixiRef = useRef<PixiHandles | null>(null);

  useEffect(() => {
    let mounted = true;

    (async () => {
      if (!hostRef.current) return;
      const pixi = await createPixiApp(hostRef.current, { cellSize });
      if (!mounted) { pixi.destroy(); return; }
      pixiRef.current = pixi;
      pixi.render(state);
    })();

    return () => {
      mounted = false;
      pixiRef.current?.destroy();
      pixiRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!pixiRef.current) return;
    pixiRef.current.setViewport(cellSize);
    pixiRef.current.render(state);
  }, [state, cellSize]);

  return <div ref={hostRef} style={{ width: "100%", height: "70vh", borderRadius: 12, overflow: "hidden" }} />;
}
