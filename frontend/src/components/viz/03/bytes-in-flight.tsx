import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants, all from docs/FACTS.md.
const SMS = 108; // section 2.1
const RATING_GBS = 1935; // A100 80GB PCIe rated bandwidth, section 2.1
const COPY_CEILING_GBS = 1275; // measured copy, learn-inference chapter 10, section 10
const LATENCY_CYCLES = 466.3; // HBM latency, A100 PCIe 40GB, Luo et al., section 3.3
const CLOCK_GHZ = 1.41; // boost clock, section 2.1
const LATENCY_NS = LATENCY_CYCLES / CLOCK_GHZ; // 330.7 ns
// Little's law: bytes in flight needed to sustain the rating, per SM.
const NEEDED_PER_SM = (RATING_GBS * 1e9 * LATENCY_NS * 1e-9) / SMS; // about 5,925 B

const THREADS = [128, 256, 512, 1024, 1536, 2048] as const;

type Load = 4 | 8 | 16;

export default function BytesInFlight() {
  const [threads, setThreads] = useState<number>(1024);
  const [load, setLoad] = useState<Load>(4);
  const [depth, setDepth] = useState<number>(1);
  const [ref, width] = useWidth<HTMLDivElement>(560);

  const perSm = threads * load * depth;
  const predicted = Math.min(RATING_GBS, (perSm * SMS) / (LATENCY_NS * 1e-9) / 1e9);
  const needThreads = Math.ceil(NEEDED_PER_SM / (load * depth));

  const pad = 10;
  const labelW = 150;
  const barW = Math.max(120, width - labelW - 2 * pad - 70);
  const barH = 22;
  const scale = (v: number, max: number) => (Math.min(v, max) / max) * barW;
  const maxA = NEEDED_PER_SM * 1.25;
  const maxB = RATING_GBS * 1.05;
  const height = pad * 2 + 2 * 56;

  const row = (y: number, label: string, value: number, max: number, mark: number, markLabel: string, tone: string, text: string) => (
    <g>
      <text x={pad} y={y + 15} fontSize={12} style={{ fill: color.muted }}>
        {label}
      </text>
      <rect x={pad + labelW} y={y} width={barW} height={barH} rx={3} style={{ fill: color.well }} />
      <rect x={pad + labelW} y={y} width={scale(value, max)} height={barH} rx={3} style={{ fill: wash(tone, 75) }} />
      <rect x={pad + labelW + scale(mark, max) - 1} y={y - 4} width={2} height={barH + 8} style={{ fill: color.fg }} />
      <text x={pad + labelW + scale(mark, max)} y={y + barH + 17} fontSize={11} textAnchor="middle" style={{ fill: color.subtle }}>
        {markLabel}
      </text>
      <text x={pad + labelW + barW + 6} y={y + 15} fontSize={12} style={{ fill: color.fg }}>
        {text}
      </text>
    </g>
  );

  return (
    <Figure
      title="Bytes in flight against the HBM rating"
      caption="A prediction from Little's law, not a measurement: bandwidth is at most the bytes in flight divided by the 466.3-cycle HBM latency (330.7 ns) that Luo et al. measured on an A100 PCIe 40GB. The black marks are the bytes per SM needed to sustain 1935 GB/s and the rating."
      controls={
        <>
          <Slider label="Resident threads per SM" value={threads} values={THREADS} onChange={setThreads} />
          <Segmented
            label="Bytes per load"
            value={load}
            options={[
              { value: 4, label: "4 (float)" },
              { value: 8, label: "8 (float2)" },
              { value: 16, label: "16 (float4)" },
            ]}
            onChange={setLoad}
          />
          <Slider label="Independent loads per thread" value={depth} min={1} max={4} onChange={setDepth} />
        </>
      }
      readout={
        <>
          <Stat label="In flight per SM" value={`${perSm.toLocaleString("en-US")} B`} />
          <Stat label="Needed per SM" value={`${Math.round(NEEDED_PER_SM).toLocaleString("en-US")} B`} />
          <Stat label="Predicted ceiling" value={`${Math.round(predicted).toLocaleString("en-US")} GB/s`} tone={predicted >= RATING_GBS ? color.ok : color.warn} />
          <Stat label="Threads needed per SM" value={`${needThreads} of 2048`} />
          <Stat label="Measured copy (reference)" value={`${COPY_CEILING_GBS} GB/s`} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Bars for bytes in flight per SM and predicted bandwidth">
          {row(pad, "Bytes in flight per SM", perSm, maxA, NEEDED_PER_SM, "needed", color.a, `${perSm.toLocaleString("en-US")} B`)}
          {row(pad + 56, "Predicted GB/s", predicted, maxB, RATING_GBS, "rating", color.b, `${Math.round(predicted)}`)}
        </svg>
      </div>
    </Figure>
  );
}
