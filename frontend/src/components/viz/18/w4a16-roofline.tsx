import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// Constants, all from docs/FACTS.md and the chapter's arithmetic:
//  - A100 80GB PCIe: 312 TFLOP/s dense fp16 tensor core [A100-DS].
//  - 1275 GB/s measured device-to-device copy (learn-inference chapter 10).
//  - A weight matrix of N = K = 8192: 67,108,864 weights.
//  - fp16 weights: 2 bytes each. W4A16 in the lab's layout: 0.5 byte of codes,
//    plus an fp16 scale and a uint8 zero point per 128 weights = 0.5234375 bytes.
//  - Activations and outputs are fp16: 2 M K + 2 M N bytes.
// The model is a roofline: time = max(bytes / bandwidth, FLOPs / peak). It
// ignores the cost of dequantization and the padding of M up to 16 in the tensor core.
const N = 8192;
const K = 8192;
const BW = 1275e9;
const PEAK = 312e12;
const FP16_BYTES_PER_WEIGHT = 2;
const W4_BYTES_PER_WEIGHT = 0.5 + 3 / 128;
const M_STOPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] as const;

function times(m: number, bytesPerWeight: number) {
  const bytes = N * K * bytesPerWeight + 2 * m * K + 2 * m * N;
  const memory = bytes / BW;
  const compute = (2 * m * N * K) / PEAK;
  return { memory, compute, time: Math.max(memory, compute), bound: memory >= compute ? "memory" : "compute" };
}

const speedup = (m: number) => times(m, FP16_BYTES_PER_WEIGHT).time / times(m, W4_BYTES_PER_WEIGHT).time;

export default function W4a16Roofline() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [m, setM] = useState<number>(16);

  const h = 240;
  const pad = { l: 44, r: 14, t: 14, b: 40 };
  const pw = Math.max(120, width - pad.l - pad.r);
  const ph = h - pad.t - pad.b;
  const x = (v: number) => pad.l + (Math.log2(v) / Math.log2(1024)) * pw;
  const y = (s: number) => pad.t + ph - (s / 4.2) * ph;

  const curve: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const v = 2 ** ((i / 100) * 10);
    curve.push(`${i === 0 ? "M" : "L"}${x(v).toFixed(1)},${y(speedup(v)).toFixed(1)}`);
  }

  const a = times(m, FP16_BYTES_PER_WEIGHT);
  const b = times(m, W4_BYTES_PER_WEIGHT);
  const bound = FP16_BYTES_PER_WEIGHT / W4_BYTES_PER_WEIGHT;
  const crossW4 = 244.7 / (2 / W4_BYTES_PER_WEIGHT); // M at which W4A16 becomes compute bound
  const crossFp16 = 244.7 / (2 / FP16_BYTES_PER_WEIGHT);

  return (
    <Figure
      title="Predicted W4A16 speedup against batch size"
      caption="Roofline prediction for an 8192 x 8192 weight matrix against the 1275 GB/s copy ceiling and 312 TFLOP/s. The gain tracks the byte ratio until the 4-bit kernel turns compute bound at M of about 64, then fades to 1 where fp16 does too. The lab measures the real curve."
      controls={<Slider label="Batch rows M" values={M_STOPS} value={m} onChange={setM} />}
      readout={
        <>
          <Stat label="fp16 time" value={`${(a.time * 1e6).toFixed(1)} us (${a.bound})`} tone={color.muted} />
          <Stat label="W4A16 time" value={`${(b.time * 1e6).toFixed(1)} us (${b.bound})`} tone={color.a} />
          <Stat label="Predicted speedup" value={`${(a.time / b.time).toFixed(2)}x`} tone={color.c} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="Predicted speedup of W4A16 over fp16 on a log scale of batch rows, flat near 3.8 and falling to 1 near M of 256">
          {[0, 1, 2, 3, 4].map((s) => (
            <g key={s}>
              <line x1={pad.l} x2={pad.l + pw} y1={y(s)} y2={y(s)} style={{ stroke: color.line }} />
              <text x={pad.l - 6} y={y(s) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {s}x
              </text>
            </g>
          ))}
          <line x1={pad.l} x2={pad.l + pw} y1={y(bound)} y2={y(bound)} strokeDasharray="5 4" style={{ stroke: color.faint }} />
          <text x={pad.l + pw} y={y(bound) - 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            byte ratio {bound.toFixed(2)}
          </text>
          {[crossW4, crossFp16].map((v, i) => (
            <g key={i}>
              <line x1={x(v)} x2={x(v)} y1={pad.t} y2={pad.t + ph} strokeDasharray="3 3" style={{ stroke: i === 0 ? color.a : color.b }} />
              <text x={x(v) + 4} y={pad.t + 12 + i * 12} fontSize={11} style={{ fill: i === 0 ? color.a : color.b }}>
                {i === 0 ? "W4A16 ridge, M = 64" : "fp16 ridge, M = 245"}
              </text>
            </g>
          ))}
          <path d={curve.join(" ")} fill="none" strokeWidth={2} style={{ stroke: color.c }} />
          <circle cx={x(m)} cy={y(speedup(m))} r={5} style={{ fill: color.c, stroke: color.paper }} strokeWidth={2} />
          {M_STOPS.filter((_, i) => i % 2 === 0).map((v) => (
            <text key={v} x={x(v)} y={h - 22} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              {v}
            </text>
          ))}
          <text x={pad.l + pw / 2} y={h - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            batch rows M (log scale)
          </text>
        </svg>
        <Legend items={[{ label: "predicted speedup", tone: color.c }]} />
      </div>
    </Figure>
  );
}
