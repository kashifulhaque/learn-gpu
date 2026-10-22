import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// A100 limits, from docs/FACTS.md (compute capability 8.0 table).
const REGISTERS_PER_SM = 65536;
const MAX_REGISTERS_PER_THREAD = 255;
const MAX_THREADS_PER_SM = 2048;
const MAX_BLOCKS_PER_SM = 32;
const WARP = 32;
const CHUNK = 4096; // the chunk size the lab's kernels use for long rows

const LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/** Registers a program needs at minimum, one fp32 register per element a thread holds, and what that implies. */
function plan(n: number, warps: number, block: number) {
  const threads = warps * WARP;
  const perThread = Math.max(1, Math.ceil(block / threads));
  const regsPerBlock = perThread * threads;
  const byRegisters = Math.floor(REGISTERS_PER_SM / regsPerBlock);
  const byThreads = Math.floor(MAX_THREADS_PER_SM / threads);
  const blocks = perThread > MAX_REGISTERS_PER_THREAD ? 0 : Math.min(byRegisters, byThreads, MAX_BLOCKS_PER_SM);
  return { threads, perThread, regsPerBlock, blocks, rowBytes: 4 * n };
}

export default function RowPerProgram() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [n, setN] = useState<number>(16384);
  const [warps, setWarps] = useState<number>(8);

  const whole = plan(n, warps, nextPow2(n));
  const chunked = plan(n, warps, Math.min(CHUNK, nextPow2(n)));
  const barW = Math.max(width - 24, 160);

  const drawBar = (label: string, p: ReturnType<typeof plan>, y: number) => {
    const per = (p.regsPerBlock / REGISTERS_PER_SM) * barW;
    const spill = p.perThread > MAX_REGISTERS_PER_THREAD;
    return (
      <g>
        <text x={12} y={y - 6} fontSize={11} style={{ fill: color.muted }}>
          {label}
        </text>
        <rect x={12} y={y} width={barW} height={22} rx={3} style={{ fill: color.well, stroke: color.line }} />
        {spill ? (
          <rect x={12} y={y} width={barW} height={22} rx={3} style={{ fill: wash(color.bad, 35), stroke: color.bad }} />
        ) : (
          Array.from({ length: p.blocks }, (_, i) => (
            <rect key={i} x={12 + i * per + 1} y={y + 1} width={Math.max(per - 2, 1)} height={20} rx={2} style={{ fill: wash(color.a, 70) }} />
          ))
        )}
        <text x={12 + barW / 2} y={y + 15} textAnchor="middle" fontSize={11} style={{ fill: color.fg }}>
          {spill ? `needs ${p.perThread} registers per thread: over the 255 limit` : `${p.blocks} program${p.blocks === 1 ? "" : "s"} per SM`}
        </text>
      </g>
    );
  };

  return (
    <Figure
      title="One row per program, or a loop over chunks"
      controls={
        <>
          <Slider label="Row length N" value={n} values={LENGTHS} onChange={setN} />
          <Segmented
            label="Warps per program"
            value={warps}
            options={[1, 2, 4, 8, 16].map((w) => ({ value: w, label: String(w) }))}
            onChange={setWarps}
          />
        </>
      }
      readout={
        <>
          <Stat label="Row in fp32" value={formatBytes(whole.rowBytes, 0)} />
          <Stat label="Whole row: registers per thread" value={`at least ${whole.perThread}`} tone={whole.perThread > MAX_REGISTERS_PER_THREAD ? color.bad : color.a} />
          <Stat label="Whole row: programs per SM" value={whole.blocks} />
          <Stat label={`Chunks of ${CHUNK}: registers per thread`} value={`at least ${chunked.perThread}`} />
          <Stat label="Chunks: programs per SM" value={chunked.blocks} />
        </>
      }
      caption="Each bar is one SM's 65,536-register file. A program that holds its whole row in registers needs at least one fp32 register per element, so a long row crowds out other programs and then exceeds 255 registers per thread. A loop over 4096-element chunks keeps the cost flat."
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={118} role="img" aria-label="Register file occupancy for a whole-row program and a chunked program">
          {drawBar(`Whole row in one block (${nextPow2(n).toLocaleString("en-US")} lanes)`, whole, 22)}
          {drawBar(`Loop over chunks of ${Math.min(CHUNK, nextPow2(n)).toLocaleString("en-US")} lanes`, chunked, 80)}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Registers of one resident program", tone: color.a },
          { label: "Over the per-thread limit", tone: color.bad },
        ]}
      />
    </Figure>
  );
}
