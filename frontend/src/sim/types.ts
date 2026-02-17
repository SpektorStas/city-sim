export type Vec2 = { x: number; y: number };

export type IndustryId = string;

export type Industry = {
  id: IndustryId;
  mode: "B2C" | "B2B";
  entryCost: number;
  fixedCost: number;
  laborShare: number;
  rentShare: number;
  baseCost: number;      // упрощённая себестоимость
  priceMarkup: number;   // упрощённая наценка
};

export type Link = {
  buyer: IndustryId;
  supplier: IndustryId;
  a: number;            // интенсивность потребления ресурса
  tauL: number;         // логистика (удорожание на дистанции)
};

export type HouseholdGroup = {
  name: string;
  budgetShare: number;
  industries: IndustryId[];
  weights: number[];    // предпочтения
  tauH: number;         // транспортные издержки
};

export type CityShape = {
  preset: "rectangle" | "circle" | "ring";
  params: Record<string, number>;
};

export type Roads = {
  mode: "grid4" | "grid8";
  connectivity: number; // доля ребёр
};

export type Obstacles = {
  blockedCells: Array<[number, number]>;
};

export type Scenario = {
  grid: { w: number; h: number; walkRadius: number };
  cityShape: CityShape;
  roads: Roads;
  obstacles: Obstacles;
  population: { total: number; distribution: "uniform" | "center" | "polycentric" };
  industries: Industry[];
  links: Link[];
  households: { groups: HouseholdGroup[] };
  dynamics: {
    shockStd: number;
    exitGrace: number;        // шагов до разрешения выхода
    exitThreshold: number;    // прибыль < 0
    relocateThreshold: number;// улучшение прибыли для релокации
    candidatesK: number;      // сколько клеток пробовать при выборе места
  };
};
