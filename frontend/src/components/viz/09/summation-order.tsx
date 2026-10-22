import { useMemo, useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Constants, all from docs/FACTS.md section 6 and the chapter's arithmetic:
//  - The significand carries p bits: bf16 p = 8 (7 stored + 1 implicit), fp16 p = 11,
//    fp32 p = 24. Every addition is rounded to p bits, nearest, ties to even. The exponent
//    range is ignored: the sums here stay far inside every format's range.
//  - The data are n pseudo-random values uniform in [0, 1), the same for every method, so
//    the exact sum is about n / 2. The error is measured against a float64 sum of the
//    rounded inputs.
type Fmt = "bf16" | "fp16" | "fp32";
const PRECISION: Record<Fmt, number> = { bf16: 8, fp16: 11, fp32: 24 };

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Round x to a p-bit significand, to nearest, ties to even. */
function roundP(x: number, p: number): number {
  if (x === 0 || !Number.isFinite(x)) return x;
  if (p >= 24) return Math.fround(x);
  const e = Math.floor(Math.log2(Math.abs(x)));
  const ulp = 2 ** (e - (p - 1));
  const n = x / ulp;
  const base = Math.floor(n);
  const frac = n - base;
  let k = base;
  if (frac > 0.5 || (frac === 0.5 && base % 2 !== 0)) k = base + 1;
  return k * ulp;
}

const SAMPLES = 160;

type Result = {
  exact: number;
  seq: number;
  pair: number;
  kahan: number;
  curve: { i: number; seq: number; exact: number }[];
};

function simulate(n: number, fmt: Fmt): Result {
  const p = PRECISION[fmt];
  const rand = mulberry32(12345);
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = roundP(rand(), p);

  let exact = 0;
  let seq = 0;
  let s = 0;
  let c = 0;
  const curve: Result["curve"] = [];
  const stride = Math.max(1, Math.floor(n / SAMPLES));
  for (let i = 0; i < n; i++) {
    exact += x[i];
    seq = roundP(seq + x[i], p);
    const y = roundP(x[i] - c, p);
    const t = roundP(s + y, p);
    c = roundP(roundP(t - s, p) - y, p);
    s = t;
    if (i % stride === 0 || i === n - 1) curve.push({ i: i + 1, seq, exact });
  }

  let level = Array.from(x);
  while (level.length > 1) {
    const next: number[] = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(i + 1 < level.length ? roundP(level[i] + level[i + 1], p) : level[i]);
    }
    level = next;
  }
  return { exact, seq, pair: level[0], kahan: s, curve };
}

function relErr(v: number, exact: number): number {
  return Math.abs(v - exact) / exact;
}

function fmtErr(e: number): string {
  if (e === 0) return "0";
  return e < 0.001 ? e.toExponential(1) : e.toFixed(4);
}

export default function SummationOrder() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [fmt, setFmt] = useState<Fmt>("bf16");
  const [logN, setLogN] = useState(12); // n = 4096

  const n = 2 ** logN;
  const r = useMemo(() => simulate(n, fmt), [n, fmt]);

  const h = 250;
  const m = { l: 56, r: 14, t: 14, b: 38 };
  const pw = Math.max(160, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const yMax = r.exact * 1.05;
  const x = (i: number) => m.l + (i / n) * pw;
  const y = (v: number) => m.t + ph - (v / yMax) * ph;
  const path = (key: "seq" | "exact") =>
    r.curve.map((pt, idx) => `${idx === 0 ? "M" : "L"}${x(pt.i)},${y(pt[key])}`).join(" ");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * r.exact);

  return (
    <Figure
      title="Three ways to add the same numbers"
      caption="The curve is the running left-to-right sum. Once it outgrows the addends by about 2^p, in a p-bit significand, adding stops changing it. The tree and Kahan sums reach the same total by a different order or by carrying the lost bits."
      controls={
        <>
          <Segmented
            label="Accumulator format"
            value={fmt}
            onChange={setFmt}
            options={[
              { value: "bf16", label: "bf16 (p = 8)" },
              { value: "fp16", label: "fp16 (p = 11)" },
              { value: "fp32", label: "fp32 (p = 24)" },
            ]}
          />
          <Slider label="Terms n" min={6} max={14} step={1} value={logN} onChange={setLogN} format={(v) => `${(2 ** v).toLocaleString("en-US")}`} />
        </>
      }
      readout={
        <>
          <Stat label="Exact sum" value={r.exact.toFixed(1)} />
          <Stat label="Sequential" value={`${r.curve[r.curve.length - 1].seq.toFixed(1)}  (err ${fmtErr(relErr(r.seq, r.exact))})`} tone={color.bad} />
          <Stat label="Pairwise tree" value={`${r.pair.toFixed(1)}  (err ${fmtErr(relErr(r.pair, r.exact))})`} tone={color.b} />
          <Stat label="Kahan" value={`${r.kahan.toFixed(1)}  (err ${fmtErr(relErr(r.kahan, r.exact))})`} tone={color.ok} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={h}
          role="img"
          aria-label={`Running sum of ${n} values in ${fmt}. The exact total is ${r.exact.toFixed(1)}. The sequential sum ends at ${r.seq.toFixed(1)}, the pairwise tree at ${r.pair.toFixed(1)}, and Kahan at ${r.kahan.toFixed(1)}.`}
        >
          {yTicks.map((t, i) => (
            <g key={i}>
              <line x1={m.l} x2={m.l + pw} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {t >= 100 ? t.toFixed(0) : t.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={m.l + pw / 2} y={h - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            terms added so far
          </text>
          <text x={12} y={m.t + ph / 2} fontSize={11} textAnchor="middle" transform={`rotate(-90 12 ${m.t + ph / 2})`} style={{ fill: color.muted }}>
            running sum
          </text>
          <path d={path("exact")} fill="none" strokeWidth={2} strokeDasharray="5 4" style={{ stroke: color.lineStrong }} />
          <path d={path("seq")} fill="none" strokeWidth={2.5} style={{ stroke: color.bad }} />
          <circle cx={x(n)} cy={y(r.pair)} r={5} style={{ fill: color.b }} />
          <circle cx={x(n)} cy={y(r.kahan)} r={5} style={{ fill: "none", stroke: color.ok }} strokeWidth={2.5} />
        </svg>
        <Legend
          items={[
            { label: "Exact running sum", tone: color.lineStrong, dashed: true },
            { label: "Sequential, rounded each step", tone: color.bad },
            { label: "Pairwise tree, final", tone: color.b },
            { label: "Kahan, final", tone: color.ok },
          ]}
        />
      </div>
    </Figure>
  );
}
