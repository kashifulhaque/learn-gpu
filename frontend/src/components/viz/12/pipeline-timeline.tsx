import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// A timing model of one thread block's main loop, not a measurement.
//
// Constants, each from the chapter:
// - Load latency default 466 cycles: global-memory latency measured on an
//   A100 PCIe 40GB (Luo et al., docs/FACTS.md 3.3). The slider moves it, since
//   a tile that hits in L2 sees about 261 cycles and a busy memory system more.
// - Compute per k-tile default 512 cycles: one 128 x 128 x 32 tile is
//   2 * 128 * 128 * 32 = 1,048,576 FLOP, and an SM's tensor cores do 2,048
//   FLOP per clock (docs/FACTS.md 2.2): 1,048,576 / 2,048 = 512.
// - Bytes per stage 18,944: a 128 x 40 A tile plus a 32 x 136 B tile in bf16.
// - 167,936 bytes of shared memory per SM (164 KiB) and 1,024 reserved per block.
const STAGE_BYTES = 18944;
const SM_SMEM = 164 * 1024;
const RESERVED = 1024;
const TILES = 24; // k-tiles the utilization is averaged over
const SHOWN = 8; // k-tiles drawn

type Bar = { tile: number; start: number; end: number };

function simulate(stages: number, latency: number, compute: number) {
  const issue: number[] = [];
  const loadDone: number[] = [];
  const start: number[] = [];
  const end: number[] = [];
  for (let t = 0; t < TILES; t++) {
    if (stages === 1) {
      issue[t] = t === 0 ? 0 : end[t - 1];
    } else if (t < stages - 1) {
      issue[t] = 0;
    } else {
      issue[t] = start[t - stages + 1];
    }
    loadDone[t] = issue[t] + latency;
    start[t] = Math.max(t === 0 ? 0 : end[t - 1], loadDone[t]);
    end[t] = start[t] + compute;
  }
  const loads: Bar[] = issue.map((s, t) => ({ tile: t, start: s, end: loadDone[t] }));
  const computes: Bar[] = start.map((s, t) => ({ tile: t, start: s, end: end[t] }));
  const busy = (TILES * compute) / end[TILES - 1];
  return { loads, computes, busy };
}

export default function PipelineTimeline() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [stages, setStages] = useState(3);
  const [latency, setLatency] = useState(466);
  const [compute, setCompute] = useState(512);

  const { loads, computes, busy } = simulate(stages, latency, compute);
  const shownEnd = computes[SHOWN - 1].end;
  const left = 96;
  const right = 12;
  const plotW = Math.max(120, width - left - right);
  const x = (cycles: number) => left + (cycles / shownEnd) * plotW;
  const lanes = Math.max(1, stages - 1);
  const laneH = 14;
  const loadTop = 28;
  const loadH = lanes * laneH + 6;
  const compTop = loadTop + loadH + 18;
  const height = compTop + 34 + 24;

  const smem = stages * STAGE_BYTES;
  const blocks = Math.max(0, Math.floor(SM_SMEM / (smem + RESERVED)));

  return (
    <Figure
      title="Pipeline depth against tensor-core idle time"
      controls={
        <>
          <Slider label="Stages" value={stages} values={[1, 2, 3, 4, 5, 6]} onChange={setStages} />
          <Slider label="Load latency (cycles)" value={latency} min={100} max={1500} step={10} onChange={setLatency} />
          <Slider label="Compute per k-tile (cycles)" value={compute} min={128} max={1024} step={16} onChange={setCompute} />
        </>
      }
      readout={
        <>
          <Stat label="Tensor cores busy (model)" value={`${(busy * 100).toFixed(0)}%`} tone={color.b} />
          <Stat label="Shared memory per block" value={formatBytes(smem)} />
          <Stat label="Blocks that fit per SM" value={blocks} tone={blocks === 0 ? color.bad : undefined} />
        </>
      }
      caption={
        <>
          A model of one block with {stages} stage{stages === 1 ? "" : "s"}, a load latency of {latency} cycles, and{" "}
          {compute} cycles of tensor-core work per k-tile. Defaults: 466 cycles is the measured global-memory latency
          on an A100 PCIe 40GB, and 512 is 1,048,576 FLOP at 2,048 FLOP per clock. The model ignores bandwidth limits,
          so it shows the floor on idle time, not a prediction.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Timeline of loads and computes for ${stages} stages`}>
          <text x={left - 8} y={loadTop + 12} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
            cp.async loads
          </text>
          <text x={left - 8} y={compTop + 20} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
            tensor cores
          </text>
          <rect x={left} y={loadTop} width={plotW} height={loadH} rx={3} style={{ fill: color.well }} />
          <rect x={left} y={compTop} width={plotW} height={28} rx={3} style={{ fill: color.well }} />
          {loads.slice(0, SHOWN).map((bar) => {
            const lane = bar.tile % lanes;
            const x0 = x(bar.start);
            const w = Math.max(1, x(Math.min(bar.end, shownEnd)) - x0);
            if (bar.start >= shownEnd) return null;
            return (
              <g key={`l${bar.tile}`}>
                <rect
                  x={x0}
                  y={loadTop + 3 + lane * laneH}
                  width={w}
                  height={laneH - 2}
                  rx={2}
                  style={{ fill: wash(color.warn, 45), stroke: color.warn }}
                />
                <text x={x0 + 3} y={loadTop + 3 + lane * laneH + 10} style={{ fill: color.fg, fontSize: 9 }}>
                  {bar.tile}
                </text>
              </g>
            );
          })}
          {computes.slice(0, SHOWN).map((bar) => {
            const x0 = x(bar.start);
            const w = Math.max(1, x(bar.end) - x0);
            return (
              <g key={`c${bar.tile}`}>
                <rect
                  x={x0}
                  y={compTop + 2}
                  width={w}
                  height={24}
                  rx={2}
                  style={{ fill: wash(color.b, 40), stroke: color.b }}
                />
                {w > 14 && (
                  <text x={x0 + w / 2} y={compTop + 18} textAnchor="middle" style={{ fill: color.fg, fontSize: 10 }}>
                    {bar.tile}
                  </text>
                )}
              </g>
            );
          })}
          <text x={left} y={compTop + 46} style={{ fill: color.subtle, fontSize: 10 }}>
            0
          </text>
          <text x={left + plotW} y={compTop + 46} textAnchor="end" style={{ fill: color.subtle, fontSize: 10 }}>
            {Math.round(shownEnd).toLocaleString("en-US")} cycles (first {SHOWN} k-tiles)
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Copy of k-tile in flight", tone: color.warn },
          { label: "MMAs on k-tile", tone: color.b },
        ]}
      />
    </Figure>
  );
}

