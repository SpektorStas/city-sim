import { Scenario } from "./types";
import { SimState, Firm } from "./state";
import { makeCityMask, makePopulation, buildRoadGraph, roadDistance } from "./scenario";
import { clamp, randn } from "./stability";

let firmId = 1;

export function initSim(s: Scenario): SimState {
  const { w, h } = s.grid;
  const mask = makeCityMask(w, h, s.cityShape, s.obstacles);
  const N = makePopulation(w, h, mask, s.population);

  const rent = new Float32Array(w * h);
  const wage = new Float32Array(w * h);

  // init baseline rents/wages
  for (let i = 0; i < w * h; i++) {
    rent[i] = mask[i] ? 1.0 : 0.0;
    wage[i] = mask[i] ? 1.0 : 0.0;
  }

  const firms: Firm[] = [];
  for (const ind of s.industries) {
    const initCount = Math.max(0, Math.floor((ind as any).firmsInit ?? 0));
    for (let k = 0; k < initCount; k++) firms.push(spawnFirm(s, ind.id, mask, w, h));
  }

  return {
    t: 0,
    w, h,
    mask,
    N,
    rent,
    wage,
    firms,
    metrics: {
      avgIncomePc: [],
      avgProfitMargin: [],
      nFirms: [],
      opens: [],
      closes: [],
      relocations: [],
      profitMarginByIndustry: {},
    },
  };
}

function spawnFirm(s: Scenario, industryId: string, mask: Uint8Array, w: number, h: number): Firm {
  const { x, y } = pickBestLocation(s, industryId, mask, w, h, null);
  return { id: firmId++, industry: industryId, x, y, age: 0, capital: 0, lastProfit: 0, negStreak: 0 };
}

function pickBestLocation(
  s: Scenario,
  industryId: string,
  mask: Uint8Array,
  w: number,
  h: number,
  current: { x: number; y: number } | null
) {
  // пробуем K случайных клеток в пределах маски
  const K = Math.max(5, s.dynamics.candidatesK);
  let best = current ?? { x: 0, y: 0 };
  let bestScore = -1e18;

  for (let k = 0; k < K; k++) {
    const idx = randomCityCell(mask);
    const x = idx % w;
    const y = Math.floor(idx / w);
    const score = locationScoreSimple(s, industryId, x, y, w, h, mask);
    if (score > bestScore) { bestScore = score; best = { x, y }; }
  }
  return best;
}

function randomCityCell(mask: Uint8Array) {
  // простая выборка: пытаться несколько раз, потом линейный поиск
  for (let i = 0; i < 50; i++) {
    const idx = (Math.random() * mask.length) | 0;
    if (mask[idx]) return idx;
  }
  for (let i = 0; i < mask.length; i++) if (mask[i]) return i;
  return 0;
}

function locationScoreSimple(s: Scenario, industryId: string, x: number, y: number, w: number, h: number, mask: Uint8Array) {
  // proxy спроса: сумма населения в радиусе R / (1 + tauH * dist)
  const R = 6; // позже вынесем
  let demand = 0;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
    const idx = yy * w + xx;
    if (!mask[idx]) continue;
    const dist = Math.hypot(dx, dy);
    demand += 1.0 / (1.0 + 0.2 * dist); // устойчивая форма
  }
  // издержки (упрощённо)
  const rent = 1.0 + 0.01 * demand;   // чем спрос выше, тем рента выше (пока так)
  return demand - rent;
}

export function step(s: Scenario, st: SimState, dt: number) {
  st.t += 1;

  let opens = 0, closes = 0, reloc = 0;

  // 1) обновить рынки факторов (пока простые)
  // rent зависит от плотности фирм + населения
  updateFactorMarkets(s, st);

  // 2) фирмы: прибыль, выход, релокация, вход
  const byIndProfit: Record<string, { profit: number; revenue: number }[]> = {};

  for (const f of st.firms) {
    const { profit, revenue } = firmProfitSimple(s, st, f);
    f.lastProfit = profit;
    f.age += 1;

    byIndProfit[f.industry] ??= [];
    byIndProfit[f.industry].push({ profit, revenue });

    if (profit < 0) f.negStreak += 1;
    else f.negStreak = 0;
  }

  // exit
  st.firms = st.firms.filter((f) => {
    if (f.age < s.dynamics.exitGrace) return true;
    if (f.negStreak >= 3) { closes++; return false; }
    return true;
  });

  // relocation
  for (const f of st.firms) {
    if (f.age < s.dynamics.exitGrace) continue;
    const cur = f.lastProfit;
    const best = pickBestLocation(s, f.industry, st.mask, st.w, st.h, { x: f.x, y: f.y });
    const tmpFirm = { ...f, ...best };
    const alt = firmProfitSimple(s, st, tmpFirm).profit;
    if (alt > cur + s.dynamics.relocateThreshold) {
      f.x = best.x; f.y = best.y;
      reloc++;
    }
  }

  // entry (упрощённо): раз в несколько шагов пробуем войти в отрасли
  if (st.t % 5 === 0) {
    for (const ind of s.industries) {
      // шанс входа зависит от "средней прибыли" отрасли
      const arr = byIndProfit[ind.id] ?? [];
      const avgP = arr.length ? arr.reduce((a, b) => a + b.profit, 0) / arr.length : 0;
      if (avgP > 0.2 && Math.random() < 0.3) {
        st.firms.push(spawnFirm(s, ind.id, st.mask, st.w, st.h));
        opens++;
      }
    }
  }

  // 3) метрики
  recordMetrics(s, st, opens, closes, reloc, byIndProfit);
}

function updateFactorMarkets(s: Scenario, st: SimState) {
  const { w, h } = st;
  const firmCount = new Float32Array(w * h);
  for (const f of st.firms) firmCount[f.y * w + f.x] += 1;

  for (let i = 0; i < w * h; i++) {
    if (!st.mask[i]) { st.rent[i] = 0; st.wage[i] = 0; continue; }
    // простая устойчивость
    st.rent[i] = 1.0 + 0.15 * firmCount[i] + 0.00001 * st.N[i];
    st.wage[i] = 1.0 + 0.05 * firmCount[i];
  }
}

function firmProfitSimple(s: Scenario, st: SimState, f: Firm) {
  // выручка ~ локальный спрос от населения вокруг
  const demand = localDemand(st, f.x, f.y, 6);
  const ind = s.industries.find((x) => x.id === f.industry)!;
  const price = ind.baseCost * (1 + ind.priceMarkup);
  const revenue = price * demand;

  const idx = f.y * st.w + f.x;
  const cost = ind.fixedCost + st.rent[idx] * ind.rentShare + st.wage[idx] * ind.laborShare + ind.baseCost * demand;

  // шок
  const shock = randn(0, s.dynamics.shockStd);
  const profit = revenue - cost + shock;

  return { profit, revenue };
}

function localDemand(st: SimState, x: number, y: number, R: number) {
  const { w, h } = st;
  let d = 0;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
    const idx = yy * w + xx;
    if (!st.mask[idx]) continue;
    const dist = Math.hypot(dx, dy);
    d += st.N[idx] / (1.0 + 0.3 * dist);
  }
  return d * 1e-4; // масштаб
}

function recordMetrics(s: Scenario, st: SimState, opens: number, closes: number, reloc: number, byIndProfit: any) {
  const n = st.firms.length;
  st.metrics.nFirms.push(n);
  st.metrics.opens.push(opens);
  st.metrics.closes.push(closes);
  st.metrics.relocations.push(reloc);

  // доход населения — proxy: avg wage
  let wageSum = 0, count = 0;
  for (let i = 0; i < st.w * st.h; i++) if (st.mask[i]) { wageSum += st.wage[i]; count++; }
  const avgW = count ? wageSum / count : 0;
  st.metrics.avgIncomePc.push(avgW);

  // маржинальность
  let mSum = 0, mCnt = 0;
  for (const ind of s.industries) {
    const arr = byIndProfit[ind.id] ?? [];
    if (!arr.length) continue;
    const rev = arr.reduce((a: number, b: any) => a + b.revenue, 0);
    const prof = arr.reduce((a: number, b: any) => a + b.profit, 0);
    const margin = rev > 1e-9 ? prof / rev : 0;
    st.metrics.profitMarginByIndustry[ind.id] ??= [];
    st.metrics.profitMarginByIndustry[ind.id].push(margin);

    mSum += margin; mCnt += 1;
  }
  st.metrics.avgProfitMargin.push(mCnt ? mSum / mCnt : 0);
}
