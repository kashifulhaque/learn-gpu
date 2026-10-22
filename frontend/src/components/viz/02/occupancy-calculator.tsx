import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Limits for compute capability 8.0 (A100), docs/FACTS.md sections 4.2 and 4.3.
const REGS_PER_SM = 65536;
const MAX_THREADS_PER_SM = 2048;
const MAX_WARPS = MAX_THREADS_PER_SM / 32; // 64
const MAX_BLOCKS = 32;
const SMEM_PER_SM = 164 * 1024;
const SMEM_PER_BLOCK_MAX = 163 * 1024;
// Allocation granularities. FACTS.md section 11, item 13, marks these as
// common knowledge from the retired calculator spreadsheet: a good
// approximation, to be confirmed with cudaOccupancyMaxActiveBlocksPerMultiprocessor.
const REG_UNIT = 256;
const SUBPARTITIONS = 4;
const SMEM_UNIT = 128;
const SMEM_RESERVED = 1024;

const THREAD_STOPS = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024] as const;

type LimitName = "registers" | "shared memory" | "blocks" | "warps";

function roundUp(value: number, unit: number): number {
  return Math.ceil(value / unit) * unit;
}

export function calculate(threads: number, regs: number, smemBytes: number) {
  const warpsPerBlock = Math.ceil(threads / 32);
  const regsPerWarp = roundUp(Math.max(regs, 1) * 32, REG_UNIT);
  const warpsByRegs = Math.floor(REGS_PER_SM / SUBPARTITIONS / regsPerWarp) * SUBPARTITIONS;
  const smemCharged = roundUp(smemBytes, SMEM_UNIT) + SMEM_RESERVED;
  const limits: Record<LimitName, number> = {
    registers: Math.floor(warpsByRegs / warpsPerBlock),
    "shared memory": smemBytes > SMEM_PER_BLOCK_MAX ? 0 : Math.floor(SMEM_PER_SM / smemCharged),
    blocks: MAX_BLOCKS,
    warps: Math.floor(MAX_WARPS / warpsPerBlock),
  };
  const blocks = Math.min(...Object.values(limits));
  const order: LimitName[] = ["registers", "shared memory", "blocks", "warps"];
  const limiter = order.find((name) => limits[name] === blocks) ?? "warps";
  const warps = blocks * warpsPerBlock;
  return { limits, blocks, limiter, warps, occupancy: warps / MAX_WARPS, warpsPerBlock };
}

export default function OccupancyCalculator() {
  const [threads, setThreads] = useState<number>(256);
  const [regs, setRegs] = useState<number>(64);
  const [smemKib, setSmemKib] = useState<number>(48);
  const [ref, width] = useWidth<HTMLDivElement>(560);

  const r = calculate(threads, regs, smemKib * 1024);

  const pad = 10;
  const labelW = 110;
  const barW = Math.max(100, width - labelW - 2 * pad - 50);
  const barH = 18;
  const rowGap = 28;
  const names: LimitName[] = ["warps", "blocks", "registers", "shared memory"];
  const maxLimit = Math.max(8, ...names.map((n) => r.limits[n]).filter((v) => v <= MAX_BLOCKS));
  const barsH = names.length * rowGap;

  // 64 warp slots in 4 rows of 16, one row per sub-partition's worth of slots.
  const cols = 16;
  const cell = Math.max(8, Math.min(26, Math.floor((width - 2 * pad - (cols - 1) * 3) / cols)));
  const gridTop = pad + barsH + 34;
  const height = gridTop + 4 * (cell + 3) + pad;

  return (
    <Figure
      title="Occupancy calculator for compute capability 8.0"
      caption="Blocks per SM is the smallest of four limits, and the bar that sets it is outlined. The 4-warp register granularity and the 128-byte and 1 KiB shared-memory rules are the common occupancy-calculator values, an approximation that the lab cross-checks against the CUDA occupancy API."
      controls={
        <>
          <Slider label="Threads per block" value={threads} values={THREAD_STOPS} onChange={setThreads} />
          <Slider label="Registers per thread" value={regs} min={16} max={255} onChange={setRegs} />
          <Slider label="Shared memory per block" value={smemKib} min={0} max={163} format={(v) => `${v} KiB`} onChange={setSmemKib} />
        </>
      }
      readout={
        <>
          <Stat label="Blocks per SM" value={r.blocks} />
          <Stat label="Resident warps" value={`${r.warps} of ${MAX_WARPS}`} />
          <Stat label="Occupancy" value={`${(r.occupancy * 100).toFixed(1)}%`} tone={r.occupancy >= 0.5 ? color.ok : color.warn} />
          <Stat label="Limited by" value={r.blocks === 0 ? "cannot launch" : r.limiter} tone={color.accent} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Bars for the four occupancy limits and a grid of the 64 warp slots on one SM">
          {names.map((name, i) => {
            const y = pad + i * rowGap;
            const value = r.limits[name];
            const shown = Math.min(value, maxLimit);
            const w = (shown / maxLimit) * barW;
            const isLimiter = r.blocks === value && name === r.limiter;
            return (
              <g key={name}>
                <text x={pad} y={y + 13} fontSize={12} style={{ fill: color.muted }}>
                  {name}
                </text>
                <rect x={pad + labelW} y={y} width={barW} height={barH} rx={3} style={{ fill: color.well }} />
                <rect
                  x={pad + labelW}
                  y={y}
                  width={w}
                  height={barH}
                  rx={3}
                  style={{ fill: wash(isLimiter ? color.accent : color.b, 70), stroke: isLimiter ? color.accent : "none", strokeWidth: 2 }}
                />
                <text x={pad + labelW + barW + 6} y={y + 13} fontSize={12} style={{ fill: color.fg }}>
                  {value}
                </text>
              </g>
            );
          })}
          <text x={pad} y={pad + barsH + 16} fontSize={12} style={{ fill: color.muted }}>
            Blocks each resource allows. Warp slots on the SM (filled = resident):
          </text>
          {Array.from({ length: MAX_WARPS }, (_, k) => {
            const row = Math.floor(k / cols);
            const col = k % cols;
            const filled = k < r.warps;
            const block = Math.floor(k / r.warpsPerBlock);
            return (
              <rect
                key={k}
                x={pad + col * (cell + 3)}
                y={gridTop + row * (cell + 3)}
                width={cell}
                height={cell}
                rx={2}
                style={{ fill: filled ? wash(block % 2 === 0 ? color.a : color.c, 80) : color.well, stroke: color.line }}
              />
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
