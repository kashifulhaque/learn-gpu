import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Per head, bfloat16 (2 bytes per element). N is the sequence length, d the head
// dimension, BM the query tile height. Element counts:
//   standard attention: write S, read S, write P, read P = 4 N^2, plus Q, K, V, O = 4 N d
//   FlashAttention, every K and V read from HBM: Q and O once (2 N d), and K and V (2 N d)
//     once per query tile, N / BM times
//   FlashAttention, L2 serves the repeat reads: K and V come from HBM once (2 N d), so 4 N d
// The L2 is 40 MiB on the A100 (docs/FACTS.md section 2.1). The L2 case applies only
// when K and V for the head fit in it, and when the programs that share them run together.
const BYTES = 2;
const BM = 128;
const L2_BYTES = 40 * 1024 * 1024;
const N_STOPS = [512, 1024, 2048, 4096, 8192, 16384] as const;

export default function IoTraffic() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [n, setN] = useState<number>(8192);
  const [d, setD] = useState<number>(128);

  const standard = (4 * n * n + 4 * n * d) * BYTES;
  const noCache = (2 * n * d + (n / BM) * 2 * n * d) * BYTES;
  const withL2 = 4 * n * d * BYTES;
  const kvBytes = 2 * n * d * BYTES;
  const fits = kvBytes <= L2_BYTES;

  const bars = [
    { label: "Standard attention", value: standard, tone: color.d },
    { label: "Flash, no cache help", value: noCache, tone: color.c },
    { label: "Flash, L2 serves K and V", value: withL2, tone: color.a },
  ];
  const max = Math.max(...bars.map((b) => b.value));
  const lx = 150;
  const pw = Math.max(120, width - lx - 90);
  const rowH = 38;
  const h = 20 + rowH * bars.length;

  return (
    <Figure
      title="Bytes moved through HBM, one head"
      caption="The tiled kernel still re-reads K and V once per query tile. It beats the standard path because the N x N matrix is gone, and by more when the L2 absorbs the repeat reads."
      controls={
        <>
          <Slider label="Sequence length N" values={N_STOPS} value={n} onChange={setN} />
          <Segmented
            label="Head dimension d"
            value={d}
            onChange={setD}
            options={[
              { value: 64, label: "64" },
              { value: 128, label: "128" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Standard" value={formatBytes(standard)} tone={color.d} />
          <Stat label="Flash, no cache" value={formatBytes(noCache)} tone={color.c} />
          <Stat label="Flash, L2 absorbs" value={formatBytes(withL2)} tone={color.a} />
          <Stat label={`K and V for the head vs 40 MiB L2`} value={`${formatBytes(kvBytes)} (${fits ? "fits" : "does not fit"})`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="Three horizontal bars comparing bytes moved through HBM by standard attention and by FlashAttention with and without L2 reuse">
          {bars.map((bar, i) => {
            const y = 10 + i * rowH;
            const w = Math.max(2, (bar.value / max) * pw);
            return (
              <g key={bar.label}>
                <text x={lx - 8} y={y + 18} textAnchor="end" fontSize={12} style={{ fill: color.muted }}>
                  {bar.label}
                </text>
                <rect x={lx} y={y} width={w} height={24} rx={3} style={{ fill: wash(bar.tone, 45), stroke: bar.tone }} />
                <text x={lx + w + 6} y={y + 17} fontSize={11} style={{ fill: color.fg }}>
                  {formatBytes(bar.value)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
