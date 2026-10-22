// How many lanes of one warp collide when 32 bytes update a 256-bin histogram,
// and what that costs each strategy. The collisions are counted on random
// bytes drawn here with a fixed seed. The counts are statistics of that draw,
// not measurements of a GPU.

import { useMemo, useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";

const BINS = 256;
const WARP = 32;
const BANKS = 32; // shared memory: bank = word address mod 32 (docs/FACTS.md, section 3.2)
const HOT_BIN = 77;
const WARPS = 400; // warps drawn per setting

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Stats = {
  sample: number[]; // the first warp's 32 bytes
  distinct: number; // mean distinct bins per warp
  sameAddress: number; // mean size of the largest group of lanes on one bin
  perBank: number; // mean largest number of lanes on one shared-memory bank
  distinctPerBank: number; // mean largest number of distinct bins on one bank
};

function simulate(hotPercent: number): Stats {
  const rand = mulberry32(2024);
  const p = hotPercent / 100;
  let distinct = 0;
  let sameAddress = 0;
  let perBank = 0;
  let distinctPerBank = 0;
  let sample: number[] = [];
  for (let w = 0; w < WARPS; w++) {
    const lanes: number[] = [];
    for (let l = 0; l < WARP; l++) lanes.push(rand() < p ? HOT_BIN : Math.floor(rand() * BINS));
    if (w === 0) sample = lanes;
    const counts = new Map<number, number>();
    for (const b of lanes) counts.set(b, (counts.get(b) ?? 0) + 1);
    const bankLanes = new Array<number>(BANKS).fill(0);
    const bankDistinct = new Array<number>(BANKS).fill(0);
    for (const b of lanes) bankLanes[b % BANKS] += 1;
    for (const b of counts.keys()) bankDistinct[b % BANKS] += 1;
    distinct += counts.size;
    sameAddress += Math.max(...counts.values());
    perBank += Math.max(...bankLanes);
    distinctPerBank += Math.max(...bankDistinct);
  }
  return {
    sample,
    distinct: distinct / WARPS,
    sameAddress: sameAddress / WARPS,
    perBank: perBank / WARPS,
    distinctPerBank: distinctPerBank / WARPS,
  };
}

export default function HistogramContention() {
  const [ref, width] = useWidth();
  const [hot, setHot] = useState(50);
  const stats = useMemo(() => simulate(hot), [hot]);

  const rows = [
    { name: "Global atomics", ops: WARP, depth: stats.sameAddress, tone: color.bad, where: "L2" },
    { name: "Shared, privatized", ops: WARP, depth: stats.perBank, tone: color.warn, where: "shared" },
    { name: "Warp-aggregated", ops: stats.distinct, depth: stats.distinctPerBank, tone: color.ok, where: "shared" },
  ];

  const pad = 10;
  const cell = (width - 2 * pad) / WARP;
  const showLabels = cell >= 20;
  const barX = pad;
  const barW = width - 2 * pad;
  const scale = (v: number) => (v / WARP) * barW;
  const blockH = 64;
  const stripH = 40;
  const height = stripH + 26 + rows.length * blockH + 4;
  const counts = new Map<number, number>();
  for (const b of stats.sample) counts.set(b, (counts.get(b) ?? 0) + 1);

  return (
    <Figure
      title="Collisions in one warp's histogram update"
      controls={<Slider label="Bytes equal to one value" min={0} max={100} step={5} value={hot} onChange={setHot} format={(v) => `${v}%`} />}
      readout={
        <>
          <Stat label="Distinct bins per warp" value={stats.distinct.toFixed(1)} tone={color.ok} />
          <Stat label="Largest same-bin group" value={stats.sameAddress.toFixed(1)} tone={color.bad} />
          <Stat label="Largest bank group" value={stats.perBank.toFixed(1)} tone={color.warn} />
          <Stat label="Hot-bin share" value={`${hot}%`} />
        </>
      }
      caption={
        <>
          The top strip is one warp&apos;s 32 bytes. Cells that share a bin are shaded. The bars compare, per warp-wide update, the
          atomics each strategy issues and the deepest queue of lanes that wait on one address or one bank. The draw is 400 warps with
          a fixed seed.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Collision statistics for ${hot}% of bytes equal to one value: ${stats.distinct.toFixed(1)} distinct bins per warp, largest same-bin group ${stats.sameAddress.toFixed(1)}.`}>
          <text x={pad} y={12} fontSize={12} style={{ fill: color.muted }}>
            One warp: the bin each of 32 lanes updates
          </text>
          {stats.sample.map((b, l) => {
            const dup = (counts.get(b) ?? 0) > 1;
            return (
              <g key={l}>
                <rect
                  x={pad + l * cell + 1}
                  y={18}
                  width={Math.max(2, cell - 2)}
                  height={stripH - 18}
                  rx={3}
                  style={{ fill: dup ? wash(color.warn, 40) : wash(color.ok, 28), stroke: dup ? color.warn : color.line }}
                />
                {showLabels && (
                  <text x={pad + l * cell + cell / 2} y={18 + (stripH - 18) / 2 + 4} textAnchor="middle" fontSize={10} style={{ fill: color.fg }}>
                    {b}
                  </text>
                )}
              </g>
            );
          })}
          {rows.map((r, i) => {
            const y0 = stripH + 26 + i * blockH;
            return (
              <g key={r.name}>
                <text x={pad} y={y0 - 6} fontSize={12} style={{ fill: color.fg, fontWeight: 600 }}>
                  {r.name}
                </text>
                <rect x={barX} y={y0} width={barW} height={14} rx={3} style={{ fill: color.well }} />
                <rect x={barX} y={y0} width={Math.max(2, scale(r.ops))} height={14} rx={3} style={{ fill: wash(r.tone, 55) }} />
                <text x={barX + 6} y={y0 + 11} fontSize={10} style={{ fill: color.fg }}>
                  {`${r.ops.toFixed(1)} atomics per warp update (${r.where})`}
                </text>
                <rect x={barX} y={y0 + 20} width={barW} height={14} rx={3} style={{ fill: color.well }} />
                <rect x={barX} y={y0 + 20} width={Math.max(2, scale(r.depth))} height={14} rx={3} style={{ fill: r.tone }} />
                <text x={barX + 6} y={y0 + 31} fontSize={10} style={{ fill: color.fg }}>
                  {`deepest queue: ${r.depth.toFixed(1)} lanes`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Lane shares its bin", tone: color.warn },
          { label: "Lane alone on its bin", tone: color.ok },
        ]}
      />
    </Figure>
  );
}
