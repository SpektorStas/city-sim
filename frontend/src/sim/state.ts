import { IndustryId } from "./types";

export type Firm = {
  id: number;
  industry: IndustryId;
  x: number;
  y: number;
  age: number;
  capital: number;
  lastProfit: number;
  negStreak: number;
};

export type SimState = {
  t: number;
  w: number;
  h: number;
  mask: Uint8Array;     // 1=город, 0=вне
  N: Float32Array;      // население по клеткам
  rent: Float32Array;   // рента
  wage: Float32Array;   // зарплата (пока можно константу/простую зависимость)
  firms: Firm[];
  metrics: {
    avgIncomePc: number[];
    avgProfitMargin: number[];
    nFirms: number[];
    opens: number[];
    closes: number[];
    relocations: number[];
    profitMarginByIndustry: Record<string, number[]>;
  };
};
