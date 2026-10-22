import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";
import { expertLoads, tileCount } from "./loads";

// Constants. The layer routes TOKENS tokens to the top-2 of E experts, so it
// has TOKENS * 2 rows to multiply. The ridge point is 312 TFLOP/s over
// 1935 GB/s = 161 FLOP per byte (docs/FACTS.md section 2); an expert's weights
// are read once per GEMM, so its intensity is about its row count in FLOP per
// byte (2 * n * d * f FLOP over 2 * d * f bytes for bf16 weights).
const TOKENS = 4096;
const TOP_K = 2;
const RIDGE = 161;
const BAR_AREA = 150;

export default function RoutingLoad() {
  const [experts, setExperts] = useState(32);
  const [skew, setSkew] = useState(1);
  const [blockM, setBlockM] = useState(64);
  const [ref, width] = useWidth<HTMLDivElement>();

  const rows = TOKENS * TOP_K;
  const counts = expertLoads(rows, experts, skew);
  const mean = rows / experts;
  const max = Math.max(...counts);
  const empty = counts.filter((c) => c === 0).length;
  const tiles = tileCount(counts, blockM);
  const waste = tiles === 0 ? 0 : 1 - rows / (tiles * blockM);
  const memoryBound = counts.filter((c) => c > 0 && c < RIDGE).length;

  const margin = 8;
  const innerW = Math.max(width, 240) - 2 * margin;
  const slot = innerW / experts;
  const barW = Math.max(slot - 1, 1);
  const scale = (n: number) => (n / max) * (BAR_AREA - 14);
  const ridgeY = BAR_AREA - scale(Math.min(RIDGE, max));

  return (
    <Figure
      title="How many rows each expert receives"
      caption={
        <>
          {TOKENS.toLocaleString("en-US")} tokens routed to the top 2 of the experts give {rows.toLocaleString("en-US")}{" "}
          rows. Bar height is a modelled load, not a measurement: the expected share of expert e is proportional to 1 / (e + 1)
          to the power of the skew. The dashed line is 161 rows, where an expert&apos;s weight-read intensity reaches the A100
          ridge point of 161 FLOP per byte. Bars below it are memory bound.
        </>
      }
      controls={
        <>
          <Slider label="Experts" value={experts} values={[8, 16, 32, 64, 128]} onChange={setExperts} />
          <Slider label="Skew" value={skew} min={0} max={2} step={0.25} onChange={setSkew} format={(v) => v.toFixed(2)} />
          <Slider label="BLOCK_M" value={blockM} values={[16, 32, 64, 128]} onChange={setBlockM} />
        </>
      }
      readout={
        <>
          <Stat label="Mean rows per expert" value={mean.toFixed(0)} />
          <Stat label="Busiest expert" value={`${max} (${(max / mean).toFixed(1)}x mean)`} tone={color.warn} />
          <Stat label="Empty experts" value={empty} />
          <Stat label="Experts below the ridge" value={`${memoryBound} of ${experts}`} tone={color.bad} />
          <Stat label="Row tiles launched" value={tiles} tone={color.a} />
          <Stat label="Padding in those tiles" value={`${(waste * 100).toFixed(1)}%`} tone={color.accent} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg
          width={Math.max(width, 240)}
          height={BAR_AREA + 8}
          role="img"
          aria-label="Bar chart of rows per expert with a dashed line at 161 rows"
        >
          {counts.map((c, e) => (
            <rect
              key={e}
              x={margin + e * slot}
              y={BAR_AREA - scale(c)}
              width={barW}
              height={Math.max(scale(c), c > 0 ? 1 : 0)}
              style={{ fill: c < RIDGE ? color.bad : color.info }}
            />
          ))}
          <line
            x1={margin}
            x2={margin + innerW}
            y1={ridgeY}
            y2={ridgeY}
            style={{ stroke: color.warn, strokeWidth: 1.5, strokeDasharray: "5 4" }}
          />
          <line x1={margin} x2={margin + innerW} y1={BAR_AREA} y2={BAR_AREA} style={{ stroke: wash(color.muted, 60) }} />
        </svg>
      </div>
      <Legend
        items={[
          { label: "expert with 161 rows or more", tone: color.info },
          { label: "expert below 161 rows", tone: color.bad },
          { label: "161 rows (ridge point)", tone: color.warn, dashed: true },
        ]}
      />
    </Figure>
  );
}
