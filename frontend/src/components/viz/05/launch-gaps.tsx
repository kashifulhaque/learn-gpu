import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// The constants come from NVIDIA's CUDA graphs article, measured on a V100 with a
// 500,000-element kernel that executes for 2.9 microseconds (docs/FACTS.md 4.5):
//   9.6 us per kernel with a sync after every launch  -> 6.7 us of overhead (D)
//   3.8 us per kernel with launches overlapped        -> a floor of 3.8 us (D)
//   3.4 us per kernel inside a graph of 20 kernels    -> 0.5 us of overhead (D)
// The lab A100 queues a launch in about 5 us (learn-inference chapter 12). The
// shape of the figure transfers to the A100; these numbers are a V100's, and this
// figure is a model, not a measurement.
const SYNC_OVERHEAD_US = 9.6 - 2.9;
const OVERLAP_FLOOR_US = 3.8;
const GRAPH_OVERHEAD_US = 3.4 - 2.9;
const KERNELS = 8;
const DURATIONS = [2.9, 5, 10, 20, 50] as const;

type Mode = "sync" | "overlap" | "graph";

function period(mode: Mode, d: number): number {
  if (mode === "sync") return d + SYNC_OVERHEAD_US;
  if (mode === "overlap") return Math.max(d, OVERLAP_FLOOR_US);
  return d + GRAPH_OVERHEAD_US;
}

export default function LaunchGaps() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [d, setD] = useState<number>(2.9);
  const [mode, setMode] = useState<Mode>("sync");

  const lx = 64;
  const pw = Math.max(140, width - lx - 16);
  // One time axis for every mode at this duration, so switching modes shows the gap shrinking.
  const axis = KERNELS * period("sync", d);
  const sx = (us: number) => lx + (us / axis) * pw;
  const p = period(mode, d);
  const total = KERNELS * p;
  const busy = d / p;
  const laneY = 40;
  const laneH = 40;

  const boxes = Array.from({ length: KERNELS }, (_, i) => i);
  const modeName = mode === "sync" ? "Sync after each launch" : mode === "overlap" ? "Launches overlapped" : "One CUDA graph";

  return (
    <Figure
      title="Gaps between kernels"
      caption="Each green box is one kernel; the space between boxes is the GPU waiting. With a sync after every launch, a short kernel leaves the GPU idle most of the time. Constants are NVIDIA's V100 measurements, so read the shape, not the microseconds."
      controls={
        <>
          <Slider label="Kernel time" values={DURATIONS} value={d} onChange={setD} format={(v) => `${v} µs`} />
          <Segmented
            label="Launch style"
            value={mode}
            onChange={setMode}
            options={[
              { value: "sync", label: "Sync each launch" },
              { value: "overlap", label: "Overlapped" },
              { value: "graph", label: "CUDA graph" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Per kernel" value={`${p.toFixed(1)} µs`} tone={color.c} />
          <Stat label="GPU busy" value={`${(busy * 100).toFixed(0)}%`} tone={color.a} />
          <Stat label={`${KERNELS} kernels take`} value={`${total.toFixed(1)} µs`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={laneY + laneH + 56} role="img" aria-label={`${modeName}: eight kernels of ${d} microseconds on a GPU lane, with gaps between them`}>
          <text x={lx - 8} y={laneY + laneH / 2 + 4} textAnchor="end" fontSize={12} style={{ fill: color.muted }}>
            GPU
          </text>
          <rect x={lx} y={laneY} width={pw} height={laneH} rx={4} style={{ fill: color.well }} />
          {boxes.map((i) => (
            <rect
              key={i}
              x={sx(i * p)}
              y={laneY}
              width={Math.max(2, sx(d) - sx(0))}
              height={laneH}
              rx={2}
              style={{ fill: wash(color.a, 55), stroke: color.a }}
            />
          ))}
          <line x1={sx(0)} x2={sx(total)} y1={laneY + laneH + 16} y2={laneY + laneH + 16} strokeWidth={4} style={{ stroke: color.c }} />
          <text x={lx} y={laneY + laneH + 38} fontSize={11} style={{ fill: color.muted }}>
            {modeName}: {total.toFixed(1)} µs for {KERNELS} kernels, axis spans {axis.toFixed(1)} µs
          </text>
          <text x={lx} y={20} fontSize={11} style={{ fill: color.muted }}>
            kernel {d} µs, period {p.toFixed(1)} µs
          </text>
        </svg>
      </div>
    </Figure>
  );
}
