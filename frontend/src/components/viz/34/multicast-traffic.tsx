import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, formatBytes, useWidth } from "../kit";

// A 4096 by 4096 by 4096 bf16 GEMM with 128 by 256 output tiles: 32 by 16 = 512
// blocks. Each block loads a 128 by 4096 slice of A and a 4096 by 256 slice of
// B from L2. A cluster of cm by cn blocks shares one A slice per cluster row
// and one B slice per cluster column. With TMA multicast, each distinct slice
// is loaded once per cluster. Rows are [cm, cn, A base, B base, A multicast,
// B multicast] in MiB, printed by
// scripts/figures/34-clusters-and-distributed-shared-memory.py.

const MIB = 1024 * 1024;
const ROWS: readonly (readonly [number, number, number, number, number, number])[] = [
  [1, 1, 512, 1024, 512, 1024],
  [2, 1, 512, 1024, 512, 512],
  [1, 2, 512, 1024, 256, 1024],
  [2, 2, 512, 1024, 256, 512],
  [4, 1, 512, 1024, 512, 256],
  [4, 2, 512, 1024, 256, 256],
  [2, 4, 512, 1024, 128, 512],
  [4, 4, 512, 1024, 128, 256],
];

export default function MulticastTraffic() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [index, setIndex] = useState(1);
  const [cm, cn, a0, b0, a1, b1] = ROWS[index];
  const base = a0 + b0;
  const multi = a1 + b1;
  const saved = 1 - multi / base;

  const barW = Math.max(120, width - 150);
  const scale = barW / base;
  const bar = (y: number, a: number, b: number, label: string) => (
    <g transform={`translate(0, ${y})`}>
      <text x={0} y={16} fontSize={12} style={{ fill: color.muted }}>
        {label}
      </text>
      <rect x={110} y={2} width={a * scale} height={20} style={{ fill: color.a }} />
      <rect x={110 + a * scale} y={2} width={b * scale} height={20} style={{ fill: color.b }} />
      <text x={116 + (a + b) * scale} y={17} fontSize={12} style={{ fill: color.fg }}>
        {formatBytes((a + b) * MIB, 1)}
      </text>
    </g>
  );

  return (
    <Figure
      title="L2 bytes loaded by a 4096 cubed bf16 GEMM"
      caption="Blocks compute 128 by 256 tiles. A cluster of cm by cn blocks shares each A slice across a cluster row and each B slice across a cluster column. Multicast loads a shared slice once per cluster, so a 2 by 1 cluster halves the B bytes, which is a third of the total."
      controls={
        <Segmented
          label="Cluster shape (M by N)"
          value={index}
          options={ROWS.map((r, i) => ({ value: i, label: `${r[0]}x${r[1]}` }))}
          onChange={setIndex}
        />
      }
      readout={
        <>
          <Stat label="Without multicast" value={formatBytes(base * MIB, 1)} />
          <Stat label="With multicast" value={formatBytes(multi * MIB, 1)} tone={color.a} />
          <Stat label="Saved" value={`${(saved * 100).toFixed(1)}%`} tone={color.ok} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, 260)}
          height={64}
          role="img"
          aria-label={`Cluster ${cm} by ${cn}: ${base} MiB without multicast, ${multi} MiB with multicast.`}
        >
          {bar(0, a0, b0, "no multicast")}
          {bar(30, a1, b1, `${cm} by ${cn} multicast`)}
        </svg>
        <Legend
          items={[
            { label: "A slices", tone: color.a },
            { label: "B slices", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
