import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants, all from docs/FACTS.md:
//  - Peak dense rates of the A100 80GB PCIe: 312 TFLOP/s bf16 tensor, 156 TF32 tensor,
//    19.5 FP32 on the CUDA cores [A100-DS].
//  - 1935 GB/s rated HBM2e bandwidth [A100-DS]; 1275 GB/s measured copy (learn-inference chapter 10).
//  - Kernel intensities are the chapter's arithmetic: a float32 vector add does 1 FLOP per 12 bytes;
//    the lab's k = 64 FMA kernel does 128 FLOP per 8 bytes; an fp16 GEMM of 8192 x 128 x 8192 is
//    124.1 FLOP/byte and of 4096^3 is 4096 / 3 = 1365.3 FLOP/byte at minimum traffic.
const PEAKS = {
  bf16: { label: "bf16 tensor", tflops: 312 },
  tf32: { label: "TF32 tensor", tflops: 156 },
  fp32: { label: "FP32 cores", tflops: 19.5 },
} as const;
type Precision = keyof typeof PEAKS;

const BANDWIDTHS = { measured: 1275, rated: 1935 } as const;
type Ceiling = keyof typeof BANDWIDTHS;

const KERNELS = [
  { name: "vector add", intensity: 1 / 12 },
  { name: "k = 64 FMAs", intensity: 16 },
  { name: "GEMM 8192x128x8192", intensity: 124.1 },
  { name: "GEMM 4096^3", intensity: 4096 / 3 },
] as const;

const X_MIN = -2; // log10 FLOP/byte
const X_MAX = 4;
const Y_MIN = -2; // log10 TFLOP/s
const Y_MAX = 3;

function attainable(intensity: number, tflops: number, gbs: number): number {
  return Math.min(tflops, (gbs * intensity) / 1000); // GB/s times FLOP/byte is GFLOP/s
}

function formatIntensity(i: number): string {
  return i < 1 ? i.toFixed(3) : i < 100 ? i.toFixed(1) : i.toFixed(0);
}

export default function Roofline() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [precision, setPrecision] = useState<Precision>("fp32");
  const [ceiling, setCeiling] = useState<Ceiling>("measured");
  const [logI, setLogI] = useState<number>(Math.log10(16));

  const peak = PEAKS[precision].tflops;
  const bw = BANDWIDTHS[ceiling];
  const ridge = (peak * 1000) / bw;
  const intensity = 10 ** logI;
  const reach = attainable(intensity, peak, bw);
  const bound = intensity < ridge ? "memory" : "compute";

  const h = 280;
  const m = { l: 52, r: 14, t: 14, b: 40 };
  const pw = Math.max(160, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const x = (i: number) => m.l + ((Math.log10(i) - X_MIN) / (X_MAX - X_MIN)) * pw;
  const y = (t: number) => m.t + ph - ((Math.log10(t) - Y_MIN) / (Y_MAX - Y_MIN)) * ph;

  const slanted = `M${x(10 ** X_MIN)},${y(attainable(10 ** X_MIN, peak, bw))} L${x(ridge)},${y(peak)}`;
  const flat = `M${x(ridge)},${y(peak)} L${x(10 ** X_MAX)},${y(peak)}`;
  const xTicks = [-2, -1, 0, 1, 2, 3, 4];
  const yTicks = [-2, -1, 0, 1, 2, 3];

  return (
    <Figure
      title="The A100 roofline"
      caption="Attainable throughput is the smaller of the flat arithmetic roof and the slanted bandwidth roof. Move the intensity to see which side of the ridge a kernel lands on."
      controls={
        <>
          <Segmented
            label="Arithmetic peak"
            value={precision}
            onChange={setPrecision}
            options={[
              { value: "fp32", label: "FP32 19.5" },
              { value: "tf32", label: "TF32 156" },
              { value: "bf16", label: "bf16 312" },
            ]}
          />
          <Segmented
            label="Bandwidth"
            value={ceiling}
            onChange={setCeiling}
            options={[
              { value: "measured", label: "Copy 1275" },
              { value: "rated", label: "Rated 1935" },
            ]}
          />
          <Slider
            label="Arithmetic intensity"
            min={X_MIN}
            max={X_MAX}
            step={0.05}
            value={logI}
            onChange={setLogI}
            format={(v) => `${formatIntensity(10 ** v)} FLOP/B`}
          />
        </>
      }
      readout={
        <>
          <Stat label="Ridge point" value={`${ridge.toFixed(1)} FLOP/B`} tone={color.c} />
          <Stat label="Attainable" value={`${reach >= 1 ? reach.toFixed(1) : reach.toFixed(3)} TFLOP/s`} tone={color.b} />
          <Stat label="Share of the peak" value={`${((reach / peak) * 100).toFixed(1)}%`} />
          <Stat label="Limited by" value={bound} tone={bound === "memory" ? color.warn : color.ok} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={h}
          role="img"
          aria-label={`Log-log roofline for the ${PEAKS[precision].label} peak of ${peak} TFLOP/s and ${bw} GB/s. The ridge point is ${ridge.toFixed(1)} FLOP per byte. The selected intensity of ${formatIntensity(intensity)} is ${bound} bound.`}
        >
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={m.l} x2={m.l + pw} y1={y(10 ** t)} y2={y(10 ** t)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={y(10 ** t) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {10 ** t}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={`x${t}`}>
              <line x1={x(10 ** t)} x2={x(10 ** t)} y1={m.t} y2={m.t + ph} style={{ stroke: color.line }} />
              <text x={x(10 ** t)} y={m.t + ph + 16} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
                {10 ** t}
              </text>
            </g>
          ))}
          <text x={m.l + pw / 2} y={h - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            arithmetic intensity, FLOP per byte (log scale)
          </text>
          <text x={12} y={m.t + ph / 2} fontSize={11} textAnchor="middle" transform={`rotate(-90 12 ${m.t + ph / 2})`} style={{ fill: color.muted }}>
            TFLOP/s (log scale)
          </text>

          <line x1={x(ridge)} x2={x(ridge)} y1={y(peak)} y2={m.t + ph} strokeDasharray="4 4" style={{ stroke: color.c }} />
          <path d={slanted} fill="none" strokeWidth={2.5} style={{ stroke: color.b }} />
          <path d={flat} fill="none" strokeWidth={2.5} style={{ stroke: color.a }} />

          {KERNELS.map((k, index) => (
            <g key={k.name}>
              <circle cx={x(k.intensity)} cy={y(attainable(k.intensity, peak, bw))} r={4} style={{ fill: color.fg }} />
              <text
                x={x(k.intensity) + (k.intensity > 300 ? -7 : 7)}
                y={y(attainable(k.intensity, peak, bw)) + (index % 2 === 0 ? 16 : -8)}
                textAnchor={k.intensity > 300 ? "end" : "start"}
                fontSize={11}
                style={{ fill: color.fg }}
              >
                {k.name}
              </text>
            </g>
          ))}

          <line x1={x(intensity)} x2={x(intensity)} y1={y(reach)} y2={m.t + ph} style={{ stroke: color.accent, strokeDasharray: "3 3" }} />
          <circle cx={x(intensity)} cy={y(reach)} r={7} strokeWidth={2} style={{ fill: wash(color.accent, 35), stroke: color.accent }} />
        </svg>
        <Legend
          items={[
            { label: "Bandwidth roof", tone: color.b },
            { label: "Arithmetic roof", tone: color.a },
            { label: "Ridge point", tone: color.c, dashed: true },
            { label: "Your intensity", tone: color.accent },
          ]}
        />
      </div>
    </Figure>
  );
}
