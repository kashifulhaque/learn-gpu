import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants, all from docs/FACTS.md section 6 (formats) and the chapter's arithmetic:
//  - A format is (exponent bits e, mantissa bits m). bias = 2^(e-1) - 1. A normal number is
//    2^(E - bias) x (1 + M / 2^m); a subnormal (E = 0) is 2^(1 - bias) x (M / 2^m).
//  - "ieee" formats reserve the top exponent for infinity and NaN (fp16, bf16, tf32, E5M2).
//    E4M3 ("fn") reuses the top exponent for finite values, so its maximum is 2^8 x 1.75 = 448.
//    E2M1 has no infinity or NaN at all, so its maximum is 2^2 x 1.5 = 6.
//  - tf32 is 1 sign, 8 exponent, 10 mantissa bits (19 meaningful bits in a 32-bit container).
type Top = "ieee" | "e4m3" | "finite";
type Format = { id: string; label: string; e: number; m: number; top: Top };

const FORMATS: readonly Format[] = [
  { id: "fp16", label: "fp16", e: 5, m: 10, top: "ieee" },
  { id: "bf16", label: "bf16", e: 8, m: 7, top: "ieee" },
  { id: "tf32", label: "tf32", e: 8, m: 10, top: "ieee" },
  { id: "e5m2", label: "E5M2", e: 5, m: 2, top: "ieee" },
  { id: "e4m3", label: "E4M3", e: 4, m: 3, top: "e4m3" },
  { id: "e2m1", label: "E2M1", e: 2, m: 1, top: "finite" },
];

type Mode = "nearest" | "truncate";

function limits(f: Format) {
  const bias = 2 ** (f.e - 1) - 1;
  const emin = 1 - bias;
  let topExp: number;
  let topSig: number;
  if (f.top === "ieee") {
    topExp = 2 ** f.e - 2 - bias;
    topSig = 2 - 2 ** -f.m;
  } else if (f.top === "e4m3") {
    topExp = 2 ** f.e - 1 - bias;
    topSig = 2 - 2 ** (1 - f.m);
  } else {
    topExp = 2 ** f.e - 1 - bias;
    topSig = 2 - 2 ** -f.m;
  }
  return { bias, emin, max: topSig * 2 ** topExp };
}

type Rounded = {
  lo: number;
  hi: number;
  ulp: number;
  binadeLo: number;
  binadeHi: number;
  value: number;
  overflow: boolean;
  tie: boolean;
};

function round(x: number, f: Format, mode: Mode): Rounded {
  const { emin, max } = limits(f);
  const e = Math.max(Math.floor(Math.log2(x)), emin);
  const ulp = 2 ** (e - f.m);
  const n = x / ulp;
  const base = Math.floor(n);
  const frac = n - base;
  const tie = frac === 0.5;
  let k = base;
  if (mode === "nearest") {
    if (frac > 0.5 || (tie && base % 2 === 1)) k = base + 1;
  }
  const value = k * ulp;
  // Round to nearest overflows once the result passes the largest finite value.
  const overflow = mode === "nearest" ? value > max : false;
  const binadeLo = x >= 2 ** emin ? 2 ** e : 0;
  const binadeHi = 2 ** (e + 1);
  return {
    lo: base * ulp,
    hi: (base + 1) * ulp,
    ulp,
    binadeLo,
    binadeHi,
    value: mode === "truncate" ? Math.min(value, max) : value,
    overflow,
    tie,
  };
}

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(4);
  return Number(v.toPrecision(8)).toString();
}

function bits(q: number, f: Format): string {
  const { bias, emin } = limits(f);
  let E: number;
  let M: number;
  if (q < 2 ** emin) {
    E = 0;
    M = Math.round(q / 2 ** (emin - f.m));
  } else {
    const e = Math.floor(Math.log2(q));
    E = e + bias;
    M = Math.round((q / 2 ** e - 1) * 2 ** f.m);
  }
  const pad = (v: number, w: number) => v.toString(2).padStart(w, "0");
  return `0 ${pad(E, f.e)} ${pad(M, f.m)}`;
}

export default function FormatRounding() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [fid, setFid] = useState<string>("bf16");
  const [mode, setMode] = useState<Mode>("nearest");
  const [k, setK] = useState(8);
  const [f, setF] = useState(4 / 1024); // 257 = (1 + 4/1024) x 2^8, the chapter's tie example

  const format = FORMATS.find((v) => v.id === fid) ?? FORMATS[1];
  const { max } = limits(format);
  const x = (1 + f) * 2 ** k;
  const r = round(x, format, mode);
  const absErr = r.overflow ? Infinity : Math.abs(r.value - x);
  const relErr = r.overflow ? Infinity : absErr / x;

  // Draw ticks only inside the binade, where the spacing is uniform.
  const first = Math.max(Math.round(r.binadeLo / r.ulp), Math.floor(x / r.ulp) - 3);
  const last = Math.min(Math.round(r.binadeHi / r.ulp), Math.floor(x / r.ulp) + 4);
  const ticks: number[] = [];
  for (let i = first; i <= last; i++) ticks.push(i * r.ulp);
  const t0 = ticks[0];
  const t1 = ticks[ticks.length - 1];

  const h = 150;
  const mx = { l: 24, r: 24 };
  const pw = Math.max(160, width - mx.l - mx.r);
  const axisY = 78;
  const px = (v: number) => mx.l + ((v - t0) / Math.max(t1 - t0, r.ulp)) * pw;
  const labelEvery = pw / ticks.length < 44 ? 2 : 1;

  const resultText = r.overflow
    ? format.top === "ieee"
      ? "overflow: infinity"
      : format.top === "e4m3"
        ? "overflow: NaN"
        : `overflow: saturates at ${fmt(max)}`
    : fmt(r.value);

  return (
    <Figure
      title="Rounding one number into a format"
      caption="The marks are the numbers the format can hold near x. Their spacing, one ulp, doubles at every power of two, so the same fraction of 2^k is a bigger or smaller error as k changes."
      controls={
        <>
          <Segmented
            label="Format"
            value={fid}
            onChange={setFid}
            options={FORMATS.map((v) => ({ value: v.id, label: v.label }))}
          />
          <Segmented
            label="Rounding"
            value={mode}
            onChange={setMode}
            options={[
              { value: "nearest", label: "Nearest, ties to even" },
              { value: "truncate", label: "Toward zero" },
            ]}
          />
          <Slider label="Exponent k" min={-12} max={16} step={1} value={k} onChange={setK} format={(v) => `2^${v}`} />
          <Slider
            label="Fraction f"
            min={0}
            max={1023 / 1024}
            step={1 / 1024}
            value={f}
            onChange={setF}
            format={(v) => `${(v * 1024).toFixed(0)} / 1024`}
          />
        </>
      }
      readout={
        <>
          <Stat label="x = (1 + f) 2^k" value={fmt(x)} tone={color.warn} />
          <Stat label="Stored value" value={resultText} tone={r.overflow ? color.bad : color.ok} />
          <Stat label="Relative error" value={r.overflow ? "n/a" : relErr.toExponential(2)} />
          <Stat label="1 ulp here" value={fmt(r.ulp)} tone={color.b} />
          <Stat label="Largest finite" value={fmt(max)} />
          <Stat label="Bits (S E M)" value={r.overflow ? "n/a" : bits(r.value, format)} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={h}
          role="img"
          aria-label={`${format.label} number line near ${fmt(x)}. The nearest stored value is ${resultText}. One ulp is ${fmt(r.ulp)}.`}
        >
          <line x1={mx.l} x2={mx.l + pw} y1={axisY} y2={axisY} style={{ stroke: color.lineStrong }} />
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={px(t)} x2={px(t)} y1={axisY - 10} y2={axisY + 10} style={{ stroke: color.b }} strokeWidth={2} />
              {i % labelEvery === 0 && (
                <text x={px(t)} y={axisY + 28} textAnchor="middle" fontSize={10} style={{ fill: color.muted }}>
                  {fmt(t)}
                </text>
              )}
            </g>
          ))}
          {!r.overflow && (
            <path
              d={`M${px(x)},${axisY - 22} Q${(px(x) + px(r.value)) / 2},${axisY - 52} ${px(r.value)},${axisY - 22}`}
              fill="none"
              strokeWidth={2}
              style={{ stroke: color.c }}
            />
          )}
          <circle cx={px(x)} cy={axisY} r={7} style={{ fill: wash(color.warn, 45), stroke: color.warn }} strokeWidth={2} />
          <text x={px(x)} y={axisY - 30} textAnchor="middle" fontSize={11} style={{ fill: color.warn }}>
            x
          </text>
          {!r.overflow && (
            <>
              <circle cx={px(r.value)} cy={axisY} r={5} style={{ fill: color.ok }} />
              <text x={px(r.value)} y={axisY - 30} textAnchor="middle" fontSize={11} style={{ fill: color.ok }}>
                stored
              </text>
            </>
          )}
          {r.tie && (
            <text x={mx.l + pw / 2} y={h - 8} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              x is exactly halfway between two stored values
            </text>
          )}
        </svg>
        <Legend
          items={[
            { label: "Stored values", tone: color.b },
            { label: "Your x", tone: color.warn },
            { label: "Rounded result", tone: color.ok },
            { label: "Error", tone: color.c },
          ]}
        />
      </div>
    </Figure>
  );
}
