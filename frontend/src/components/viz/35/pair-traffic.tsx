import { useState } from "react";
import { Figure, Segmented, Stat, color, formatBytes, useWidth } from "../kit";

// Per-CTA operand bytes for one K = 64 bf16 slice, when each CTA owns 128
// accumulator rows. From scripts/figures/35-blackwell-tcgen05.py, which calls
// the lab's per_cta_operand_bytes and operand_intensity:
// [N, A bytes 1-CTA, B bytes 1-CTA, A bytes pair, B bytes pair, FLOP/byte 1-CTA, FLOP/byte pair]
const ROWS: readonly (readonly [number, number, number, number, number, number, number])[] = [
  [64, 16384, 8192, 16384, 4096, 42.7, 51.2],
  [128, 16384, 16384, 16384, 8192, 64.0, 85.3],
  [192, 16384, 24576, 16384, 12288, 76.8, 109.7],
  [256, 16384, 32768, 16384, 16384, 85.3, 128.0],
];

export default function PairTraffic() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [n, setN] = useState(256);
  const row = ROWS.find((r) => r[0] === n) ?? ROWS[3];
  const [, a1, b1, a2, b2, i1, i2] = row;

  const maxBytes = 49152;
  const barMax = Math.max(160, width - 190);
  const scale = barMax / maxBytes;
  const bars = [
    { label: "One CTA", a: a1, b: b1 },
    { label: "Each CTA of a pair", a: a2, b: b2 },
  ];

  return (
    <Figure
      title="Operand bytes one CTA stages per K = 64 slice"
      caption="Each CTA owns 128 accumulator rows and N columns, in bf16. In a pair, a CTA stages its own 128 rows of A and half of the N rows of B. The B segment halves, and the A segment does not change."
      controls={
        <Segmented
          label="N"
          value={n}
          options={ROWS.map((r) => ({ value: r[0], label: String(r[0]) }))}
          onChange={setN}
        />
      }
      readout={
        <>
          <Stat label="B bytes, one CTA" value={formatBytes(b1, 0)} tone={color.b} />
          <Stat label="B bytes, pair" value={formatBytes(b2, 0)} tone={color.b} />
          <Stat label="FLOP per byte" value={`${i1.toFixed(1)} to ${i2.toFixed(1)}`} tone={color.a} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, 190 + barMax)}
          height={110}
          role="img"
          aria-label={`At N of ${n}, one CTA stages ${a1} bytes of A and ${b1} of B. Each CTA of a pair stages ${a2} of A and ${b2} of B.`}
        >
          {bars.map((bar, i) => (
            <g key={bar.label} transform={`translate(0, ${14 + i * 46})`}>
              <text x={0} y={20} fontSize={12} style={{ fill: color.fg }}>
                {bar.label}
              </text>
              <rect x={150} y={4} width={bar.a * scale} height={28} rx={3} style={{ fill: color.a }} />
              <rect x={150 + bar.a * scale} y={4} width={bar.b * scale} height={28} rx={3} style={{ fill: color.b }} />
              <text x={150 + (bar.a + bar.b) * scale + 6} y={23} fontSize={11} style={{ fill: color.muted }}>
                {formatBytes(bar.a + bar.b, 0)}
              </text>
            </g>
          ))}
          <text x={150} y={106} fontSize={11} style={{ fill: color.a }}>
            A
          </text>
          <text x={166} y={106} fontSize={11} style={{ fill: color.b }}>
            B
          </text>
        </svg>
      </div>
    </Figure>
  );
}
