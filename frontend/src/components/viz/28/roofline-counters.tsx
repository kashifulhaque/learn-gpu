import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// Peaks for the A100 80GB PCIe, from docs/FACTS.md: 19.5 TFLOP/s FP32 on the
// CUDA cores and 1935 GB/s HBM2e. The kernels are the synthesized report in
// content/labs/28-profile-reports/fixture_ncu.py, and every number below is
// printed by scripts/figures/28-reading-profiler-reports.py from its counters.
const PEAK_GFLOPS = 19500;
const BANDWIDTH_GBS = 1935;
const RIDGE = PEAK_GFLOPS / BANDWIDTH_GBS;

type Row = {
  name: string;
  label: string;
  bytes: number;
  flops: number;
  ms: number;
  ai: number | null;
  gflops: number;
  gbs: number;
  pct: number;
};

// ROOFLINE: peak 19.5 TFLOP/s FP32, 1935 GB/s, ridge 10.08 FLOP/B
const ROOFLINE: Row[] = [
  { name: "vec_add", label: "memory-bound", bytes: 805306368, flops: 67108864, ms: 0.6291, ai: 0.0833, gflops: 106.7, gbs: 1280.0, pct: 66.1 },
  { name: "sgemm_tiled", label: "compute-bound", bytes: 201326592, flops: 137455730688, ms: 8.4849, ai: 682.75, gflops: 16200.0, gbs: 23.7, pct: 83.1 },
  { name: "copy_strided", label: "uncoalesced", bytes: 603979776, flops: 0, ms: 0.5033, ai: 0.0, gflops: 0.0, gbs: 1200.0, pct: 62.0 },
  { name: "transpose_naive_smem", label: "bank-conflicts", bytes: 536870912, flops: 0, ms: 0.93, ai: 0.0, gflops: 0.0, gbs: 577.3, pct: 29.8 },
  { name: "reduce_tail", label: "tail-effect", bytes: 83886080, flops: 12994560000, ms: 2.0, ai: 154.9072, gflops: 6497.3, gbs: 41.9, pct: 33.3 },
  { name: "scan_lookback", label: "latency-bound", bytes: 1073741824, flops: 0, ms: 3.9636, ai: 0.0, gflops: 0.0, gbs: 270.9, pct: 14.0 },
];

const X_MIN = 0.01;
const X_MAX = 1000;
const Y_MIN = 10;
const Y_MAX = 30000;

export default function RooflineCounters() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [name, setName] = useState("sgemm_tiled");
  const row = ROOFLINE.find((r) => r.name === name) ?? ROOFLINE[0];

  const w = Math.max(width, 320);
  const h = 300;
  const padL = 52;
  const padR = 16;
  const padT = 12;
  const padB = 36;
  const x = (v: number) => padL + ((Math.log10(v) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (w - padL - padR);
  const y = (v: number) => padT + (1 - (Math.log10(v) - Math.log10(Y_MIN)) / (Math.log10(Y_MAX) - Math.log10(Y_MIN))) * (h - padT - padB);

  const roof = `M ${x(X_MIN)} ${y(BANDWIDTH_GBS * X_MIN)} L ${x(RIDGE)} ${y(PEAK_GFLOPS)} L ${x(X_MAX)} ${y(PEAK_GFLOPS)}`;
  const xTicks = [0.01, 0.1, 1, 10, 100, 1000];
  const yTicks = [10, 100, 1000, 10000];
  const plotted = ROOFLINE.filter((r) => r.ai !== null && r.ai > 0);
  const hasPoint = row.ai !== null && row.ai > 0;

  return (
    <Figure
      title="A kernel on the roofline, from raw counters"
      caption="Each point is FLOPs and DRAM bytes from the counters of the recorded report, over its duration. The roof is the A100 80GB PCIe FP32 peak and its 1935 GB/s HBM2e rating. A kernel with no FLOPs has no place on a log axis, so its readout shows the bandwidth share instead."
      controls={
        <Segmented
          label="Kernel"
          value={name}
          onChange={setName}
          options={ROOFLINE.map((r) => ({ value: r.name, label: r.name }))}
        />
      }
      readout={
        <>
          <Stat label="DRAM bytes" value={row.bytes.toLocaleString("en-US")} />
          <Stat label="FP32 FLOPs" value={row.flops.toLocaleString("en-US")} />
          <Stat label="FLOP per byte" value={hasPoint ? String(row.ai) : "0"} tone={color.a} />
          <Stat label="GFLOP/s" value={row.gflops.toLocaleString("en-US")} tone={color.b} />
          <Stat label="GB/s" value={row.gbs.toLocaleString("en-US")} tone={color.c} />
          <Stat label="Diagnosis" value={row.label} />
        </>
      }
    >
      <div ref={ref}>
        <Legend
          items={[
            { label: "Roof: 1935 GB/s slope, then 19.5 TFLOP/s", tone: color.fg },
            { label: "Selected kernel", tone: color.accent },
            { label: "Other kernels", tone: color.faint },
          ]}
        />
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Roofline with the ridge at ${RIDGE.toFixed(2)} FLOP per byte. ${row.name} does ${hasPoint ? row.ai : 0} FLOP per byte and reaches ${row.gflops} GFLOP per second.`}
        >
          {xTicks.map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={padT} y2={h - padB} style={{ stroke: color.line }} />
              <text x={x(t)} y={h - padB + 14} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                {t}
              </text>
            </g>
          ))}
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={padL} x2={w - padR} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={padL - 6} y={y(t) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {t.toLocaleString("en-US")}
              </text>
            </g>
          ))}
          <text x={(padL + w - padR) / 2} y={h - 4} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            FLOP per DRAM byte
          </text>
          <text x={12} y={padT + 8} fontSize={11} style={{ fill: color.muted }}>
            GFLOP/s
          </text>
          <path d={roof} fill="none" strokeWidth={2} style={{ stroke: color.fg }} />
          <line x1={x(RIDGE)} x2={x(RIDGE)} y1={y(PEAK_GFLOPS)} y2={h - padB} strokeDasharray="4 4" style={{ stroke: color.faint }} />
          <text x={x(RIDGE) + 4} y={h - padB - 6} fontSize={11} style={{ fill: color.muted }}>
            ridge {RIDGE.toFixed(2)}
          </text>
          {plotted.map((r) => {
            const on = r.name === name;
            return (
              <g key={r.name}>
                <circle
                  cx={x(r.ai as number)}
                  cy={y(r.gflops)}
                  r={on ? 7 : 5}
                  style={{ fill: on ? color.accent : wash(color.faint, 70), stroke: color.paper }}
                />
                <text
                  x={x(r.ai as number) + (x(r.ai as number) > w - 110 ? -9 : 9)}
                  y={y(r.gflops) - 8}
                  fontSize={11}
                  textAnchor={x(r.ai as number) > w - 110 ? "end" : "start"}
                  style={{ fill: on ? color.fg : color.muted }}
                >
                  {r.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
