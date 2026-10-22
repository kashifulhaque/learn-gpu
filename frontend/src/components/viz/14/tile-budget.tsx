import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants, from docs/FACTS.md section 2: 108 SMs; 163 KiB of shared memory
// per block (164 KiB per SM); 255 registers per thread; 32 threads per warp;
// the ridge point 312e12 / 1.935e12 = 161.2 FLOP per byte against the rated
// bandwidth. Tiles are fp16: 2 bytes per element. The matmul is 4096^3.
const SMS = 108;
const SMEM_LIMIT_KIB = 163;
const REG_LIMIT = 255;
const WARP = 32;
const RIDGE = 312e12 / 1.935e12;
const N = 4096;

type Tile = "128x256x64" | "128x128x32" | "64x256x32" | "128x64x32" | "64x32x32";
const TILES: readonly { value: Tile; label: string; bm: number; bn: number; bk: number }[] = [
  { value: "128x256x64", label: "128 x 256 x 64", bm: 128, bn: 256, bk: 64 },
  { value: "128x128x32", label: "128 x 128 x 32", bm: 128, bn: 128, bk: 32 },
  { value: "64x256x32", label: "64 x 256 x 32", bm: 64, bn: 256, bk: 32 },
  { value: "128x64x32", label: "128 x 64 x 32", bm: 128, bn: 64, bk: 32 },
  { value: "64x32x32", label: "64 x 32 x 32", bm: 64, bn: 32, bk: 32 },
];

export default function TileBudget() {
  const [tile, setTile] = useState<Tile>("128x256x64");
  const [stages, setStages] = useState(3);
  const [warps, setWarps] = useState(8);
  const [ref, width] = useWidth<HTMLDivElement>();

  const t = TILES.find((x) => x.value === tile)!;
  const smemKiB = ((t.bm * t.bk + t.bk * t.bn) * 2 * stages) / 1024;
  const fits = smemKiB <= SMEM_LIMIT_KIB;
  const accRegs = (t.bm * t.bn) / (WARP * warps);
  const regsOk = accRegs <= REG_LIMIT;
  const intensity = (t.bm * t.bn) / (t.bm + t.bn); // FLOP per byte loaded, fp16
  const minHit = intensity >= RIDGE ? 0 : 1 - intensity / RIDGE;
  const tiles = Math.ceil(N / t.bm) * Math.ceil(N / t.bn);
  const waves = Math.ceil(tiles / SMS);
  const efficiency = tiles / (waves * SMS);

  const labelW = 120;
  const barW = Math.max(120, width - labelW - 90);
  const rowH = 22;
  const bars = [
    { label: "Shared memory", value: smemKiB, max: 2 * SMEM_LIMIT_KIB, limit: SMEM_LIMIT_KIB, unit: "KiB", ok: fits },
    { label: "Accumulator regs", value: accRegs, max: 2 * REG_LIMIT, limit: REG_LIMIT, unit: "per thread", ok: regsOk },
    { label: "FLOP per byte", value: intensity, max: 2 * RIDGE, limit: RIDGE, unit: "FLOP/B", ok: intensity >= RIDGE },
  ];
  const height = bars.length * (rowH + 10) + 8;

  return (
    <Figure
      title="What one tile costs, and what it buys"
      caption={
        <>
          A tile has to fit in shared memory and registers, and its arithmetic intensity, BLOCK_M x BLOCK_N / (BLOCK_M +
          BLOCK_N) FLOP per byte for fp16, says how much reuse it gives each byte it loads. A tile below the ridge point
          of 161 FLOP per byte needs the L2 to supply the rest. The wave count assumes one program per SM.
        </>
      }
      controls={
        <>
          <Segmented label="Tile (M x N x K)" value={tile} options={TILES} onChange={setTile} />
          <Slider label="num_stages" value={stages} values={[2, 3, 4, 5]} onChange={setStages} />
          <Slider label="num_warps" value={warps} values={[2, 4, 8]} onChange={setWarps} />
        </>
      }
      readout={
        <>
          <Stat label="Shared memory" value={`${smemKiB.toFixed(0)} KiB ${fits ? "fits" : "too big"}`} tone={fits ? color.ok : color.bad} />
          <Stat label="Accumulator registers" value={`${accRegs.toFixed(0)} per thread`} tone={regsOk ? color.ok : color.bad} />
          <Stat label="Intensity" value={`${intensity.toFixed(1)} FLOP/B`} tone={color.a} />
          <Stat label="L2 hit needed" value={`${(100 * minHit).toFixed(0)}%`} />
          <Stat label="Tiles, 4096 x 4096" value={tiles} />
          <Stat label="Waves, fill" value={`${waves}, ${(100 * efficiency).toFixed(1)}%`} tone={color.c} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Shared memory, registers, and intensity of a tile against their limits">
          {bars.map((b, i) => {
            const y = 4 + i * (rowH + 10);
            const w = Math.min(barW, (b.value / b.max) * barW);
            const limitX = labelW + (b.limit / b.max) * barW;
            return (
              <g key={b.label}>
                <text x={0} y={y + rowH - 7} style={{ fill: color.muted, fontSize: 11 }}>
                  {b.label}
                </text>
                <rect x={labelW} y={y} width={barW} height={rowH} rx={3} style={{ fill: wash(color.muted, 12) }} />
                <rect x={labelW} y={y} width={w} height={rowH} rx={3} style={{ fill: b.ok ? color.ok : color.bad, opacity: 0.8 }} />
                <line x1={limitX} x2={limitX} y1={y - 2} y2={y + rowH + 2} style={{ stroke: color.fg, strokeWidth: 1.5 }} />
                <text x={labelW + barW + 8} y={y + rowH - 7} style={{ fill: color.muted, fontSize: 11 }}>
                  {b.value.toFixed(b.unit === "FLOP/B" ? 1 : 0)} {b.unit === "per thread" ? "" : b.unit}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "within the limit, or above the ridge", tone: color.ok },
          { label: "over the limit, or below the ridge", tone: color.bad },
        ]}
      />
    </Figure>
  );
}
