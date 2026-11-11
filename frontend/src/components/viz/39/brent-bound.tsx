import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatCount, useWidth } from "../kit";

function short(v: number): string {
  if (v >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1).replace(/\.0$/, "")}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1).replace(/\.0$/, "")}k`;
  return v.toFixed(v < 100 ? 0 : 0);
}

// Brent's bound for a scan or a reduction of n = 2^20 values. The upper bound
// on a greedy schedule is W / p + S additions-steps. Work W and span S:
//   tree reduction  W = 1,048,575   S = 20
//   Kogge-Stone     W = 19,922,945  S = 20
//   Blelloch        W = 2,097,150   S = 40
// scripts/figures/39-memory-consistency-and-atomics.py prints this table from
// the lab's work_span functions. It is arithmetic, not a measurement.

const BOUNDS = [
  { p: 1, reduction: 1048595.0, kogge: 19922965.0, blelloch: 2097190.0 },
  { p: 2, reduction: 524307.5, kogge: 9961492.5, blelloch: 1048615.0 },
  { p: 4, reduction: 262163.75, kogge: 4980756.25, blelloch: 524327.5 },
  { p: 8, reduction: 131091.88, kogge: 2490388.12, blelloch: 262183.75 },
  { p: 16, reduction: 65555.94, kogge: 1245204.06, blelloch: 131111.88 },
  { p: 32, reduction: 32787.97, kogge: 622612.03, blelloch: 65575.94 },
  { p: 64, reduction: 16403.98, kogge: 311316.02, blelloch: 32807.97 },
  { p: 128, reduction: 8211.99, kogge: 155668.01, blelloch: 16423.98 },
  { p: 256, reduction: 4116.0, kogge: 77844.0, blelloch: 8231.99 },
  { p: 512, reduction: 2068.0, kogge: 38932.0, blelloch: 4136.0 },
  { p: 1024, reduction: 1044.0, kogge: 19476.0, blelloch: 2088.0 },
  { p: 2048, reduction: 532.0, kogge: 9748.0, blelloch: 1064.0 },
  { p: 4096, reduction: 276.0, kogge: 4884.0, blelloch: 552.0 },
  { p: 8192, reduction: 148.0, kogge: 2452.0, blelloch: 296.0 },
  { p: 16384, reduction: 84.0, kogge: 1236.0, blelloch: 168.0 },
  { p: 32768, reduction: 52.0, kogge: 628.0, blelloch: 104.0 },
  { p: 65536, reduction: 36.0, kogge: 324.0, blelloch: 72.0 },
  { p: 131072, reduction: 28.0, kogge: 172.0, blelloch: 56.0 },
  { p: 262144, reduction: 24.0, kogge: 96.0, blelloch: 48.0 },
  { p: 524288, reduction: 22.0, kogge: 58.0, blelloch: 44.0 },
  { p: 1048576, reduction: 21.0, kogge: 39.0, blelloch: 42.0 },
];

const SERIES = [
  { key: "reduction", label: "Tree reduction", tone: color.a },
  { key: "kogge", label: "Kogge-Stone scan", tone: color.b },
  { key: "blelloch", label: "Blelloch scan", tone: color.c },
] as const;

const PS = BOUNDS.map((row) => row.p);
const YMIN = 10;
const YMAX = 3e7;

export default function BrentBound() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [p, setP] = useState(1024);

  const w = Math.max(width, 320);
  const h = 260;
  const left = 52;
  const right = 12;
  const top = 12;
  const bottom = 30;
  const x = (e: number) => left + (e / 20) * (w - left - right);
  const y = (v: number) => {
    const t = (Math.log10(v) - Math.log10(YMIN)) / (Math.log10(YMAX) - Math.log10(YMIN));
    return top + (1 - t) * (h - top - bottom);
  };
  const at = BOUNDS[PS.indexOf(p)];
  const exp = Math.log2(p);

  return (
    <Figure
      title="Brent's bound for 2^20 values"
      caption="The bound is work divided by p, plus span, in addition steps. At 1,024 processors the bounds are 1,044, 19,476, and 2,088. Kogge-Stone does 9.5 times Blelloch's work, so it loses until p passes about 891,290, where its span of 20 beats Blelloch's 40."
      controls={
        <Slider
          label="Processors p"
          values={PS}
          value={p}
          onChange={setP}
          format={(v) => formatCount(v)}
        />
      }
      readout={
        <>
          {SERIES.map((s) => (
            <Stat key={s.key} label={s.label} value={short(at[s.key])} tone={s.tone} />
          ))}
        </>
      }
    >
      <div ref={ref}>
        <Legend items={SERIES.map((s) => ({ label: s.label, tone: s.tone }))} />
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Upper bound on steps for three algorithms at ${p} processors: reduction ${at.reduction}, Kogge-Stone ${at.kogge}, Blelloch ${at.blelloch}.`}
        >
          {[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7].map((v) => (
            <g key={v}>
              <line x1={left} x2={w - right} y1={y(v)} y2={y(v)} style={{ stroke: color.line }} />
              <text x={left - 6} y={y(v) + 4} fontSize={10} textAnchor="end" style={{ fill: color.muted }}>
                {short(v)}
              </text>
            </g>
          ))}
          {[0, 5, 10, 15, 20].map((e) => (
            <text key={e} x={x(e)} y={h - 10} fontSize={10} textAnchor="middle" style={{ fill: color.muted }}>
              {`2^${e}`}
            </text>
          ))}
          {SERIES.map((s) => (
            <polyline
              key={s.key}
              fill="none"
              strokeWidth={2}
              style={{ stroke: s.tone }}
              points={BOUNDS.map((row, i) => `${x(i)},${y(row[s.key])}`).join(" ")}
            />
          ))}
          <line x1={x(exp)} x2={x(exp)} y1={top} y2={h - bottom} strokeDasharray="4 3" style={{ stroke: color.faint }} />
          {SERIES.map((s) => (
            <circle key={s.key} cx={x(exp)} cy={y(at[s.key])} r={4} style={{ fill: s.tone }} />
          ))}
        </svg>
      </div>
    </Figure>
  );
}
