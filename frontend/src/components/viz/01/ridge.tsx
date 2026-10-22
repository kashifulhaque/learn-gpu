import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Peak = SMs x FLOP per SM per clock x clock, from docs/FACTS.md, section 2.2.
// The clocks are the ones implied by NVIDIA's rated throughput; the B200 tensor
// clock is unconfirmed (section 2.2).
type Precision = "fp32" | "tf32" | "bf16" | "fp8" | "fp4";
type GpuKey = "A100" | "H100" | "B200";

type Gpu = {
  sms: number;
  tensorGhz: number;
  vectorGhz: number;
  bf16PerSmClock: number;
  fp32PerSmClock: number;
  gbs: number; // rated HBM bandwidth
  tone: string;
};

const GPUS: Record<GpuKey, Gpu> = {
  A100: { sms: 108, tensorGhz: 1.41, vectorGhz: 1.41, bf16PerSmClock: 2048, fp32PerSmClock: 128, gbs: 1935, tone: color.info },
  H100: { sms: 132, tensorGhz: 1.83, vectorGhz: 1.98, bf16PerSmClock: 4096, fp32PerSmClock: 256, gbs: 3350, tone: color.warn },
  B200: { sms: 148, tensorGhz: 1.856, vectorGhz: 1.98, bf16PerSmClock: 8192, fp32PerSmClock: 256, gbs: 8000, tone: color.violet },
};

// Each tensor precision relative to bf16, and the first GPU that has it.
const RATIO: Record<Exclude<Precision, "fp32">, number> = { tf32: 0.5, bf16: 1, fp8: 2, fp4: 4 };
const SUPPORT: Record<Precision, GpuKey[]> = {
  fp32: ["A100", "H100", "B200"],
  tf32: ["A100", "H100", "B200"],
  bf16: ["A100", "H100", "B200"],
  fp8: ["H100", "B200"],
  fp4: ["B200"],
};

function perSmClock(g: Gpu, p: Precision): number {
  return p === "fp32" ? g.fp32PerSmClock : g.bf16PerSmClock * RATIO[p];
}
function peakTflops(g: Gpu, p: Precision): number {
  const ghz = p === "fp32" ? g.vectorGhz : g.tensorGhz;
  return (g.sms * perSmClock(g, p) * ghz) / 1e3; // SMs x FLOP/clk x GHz = GFLOP/s, then to TFLOP/s
}

const INTENSITIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] as const;
const X_MIN = 1;
const X_MAX = 4096;
const Y_MIN = 1;
const Y_MAX = 10000;

export default function PeaksRidge() {
  const [precision, setPrecision] = useState<Precision>("bf16");
  const [intensity, setIntensity] = useState<number>(1);
  const [ref, width] = useWidth<HTMLDivElement>(560);

  const keys = SUPPORT[precision];
  const height = Math.max(240, Math.min(340, Math.round(width * 0.55)));
  const m = { l: 48, r: 14, t: 12, b: 34 };
  const pw = width - m.l - m.r;
  const ph = height - m.t - m.b;
  const x = (v: number) => m.l + (Math.log2(v / X_MIN) / Math.log2(X_MAX / X_MIN)) * pw;
  const y = (v: number) => m.t + ph - (Math.log10(v / Y_MIN) / Math.log10(Y_MAX / Y_MIN)) * ph;

  const rows = keys.map((k) => {
    const g = GPUS[k];
    const peak = peakTflops(g, precision);
    const ridge = (peak * 1e3) / g.gbs; // TFLOP/s over GB/s = 1000 FLOP per byte, so scale by 1e3
    const attainable = Math.min(peak, (intensity * g.gbs) / 1e3);
    return { k, g, peak, ridge, attainable };
  });

  const yTicks = [1, 10, 100, 1000, 10000];
  const xTicks = [1, 4, 16, 64, 256, 1024, 4096];

  return (
    <Figure
      title="Rooflines and ridge points, A100 to B200"
      caption="Each line is the attainable TFLOP/s: it climbs with arithmetic intensity until it meets the peak at the ridge point, then goes flat. Peaks are dense, from SMs x FLOP per SM per clock x clock; bandwidths are the rated HBM figures. The B200 uses the HGX values and an unconfirmed implied clock."
      controls={
        <>
          <Segmented
            label="Precision"
            value={precision}
            options={[
              { value: "fp32", label: "FP32" },
              { value: "tf32", label: "TF32" },
              { value: "bf16", label: "BF16" },
              { value: "fp8", label: "FP8" },
              { value: "fp4", label: "FP4" },
            ]}
            onChange={setPrecision}
          />
          <Slider label="Arithmetic intensity" value={intensity} values={INTENSITIES} onChange={setIntensity} format={(v) => `${v} FLOP/B`} />
        </>
      }
      readout={
        <>
          {rows.map((r) => (
            <Stat
              key={r.k}
              label={`${r.k}: peak, ridge`}
              value={`${r.peak.toFixed(r.peak < 100 ? 1 : 0)} TFLOP/s, ${r.ridge.toFixed(r.ridge < 100 ? 1 : 0)}`}
              tone={r.g.tone}
            />
          ))}
          {rows.map((r) => (
            <Stat
              key={`${r.k}-at`}
              label={`${r.k} at ${intensity} FLOP/B`}
              value={`${r.attainable.toFixed(r.attainable < 10 ? 2 : 1)} TFLOP/s (${((r.attainable / r.peak) * 100).toFixed(r.attainable / r.peak < 0.1 ? 2 : 1)}%)`}
              tone={r.g.tone}
            />
          ))}
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Log-log roofline plot of attainable TFLOP/s against arithmetic intensity for the A100, H100, and B200">
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={m.l} x2={m.l + pw} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={y(t) + 4} textAnchor="end">
                {t >= 1000 ? `${t / 1000}k` : t}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={m.t} y2={m.t + ph} style={{ stroke: color.line }} />
              <text x={x(t)} y={m.t + ph + 15} textAnchor="middle">
                {t}
              </text>
            </g>
          ))}
          <text x={m.l + pw / 2} y={height - 4} textAnchor="middle">
            Arithmetic intensity (FLOP per byte)
          </text>
          <text x={12} y={m.t + 8} textAnchor="start">
            TFLOP/s
          </text>
          <line x1={x(intensity)} x2={x(intensity)} y1={m.t} y2={m.t + ph} style={{ stroke: color.fg, strokeDasharray: "4 3" }} />
          {rows.map((r) => {
            const startY = Math.max(Y_MIN, r.g.gbs / 1e3);
            const ridgeIn = r.ridge < X_MAX;
            const d = ridgeIn
              ? `M ${x(1)} ${y(startY)} L ${x(r.ridge)} ${y(r.peak)} L ${x(X_MAX)} ${y(r.peak)}`
              : `M ${x(1)} ${y(startY)} L ${x(X_MAX)} ${y(Math.min(Y_MAX, (X_MAX * r.g.gbs) / 1e3))}`;
            return (
              <g key={r.k}>
                <path d={d} fill="none" style={{ stroke: r.g.tone, strokeWidth: 2 }} />
                {ridgeIn && <circle cx={x(r.ridge)} cy={y(r.peak)} r={4.5} style={{ fill: color.paper, stroke: r.g.tone, strokeWidth: 2 }} />}
                <circle cx={x(intensity)} cy={y(r.attainable)} r={4} style={{ fill: r.g.tone }} />
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          ...keys.map((k) => ({ label: k, tone: GPUS[k].tone })),
          ...(keys.length < 3 ? [{ label: `no ${precision.toUpperCase()} tensor path on ${(["A100", "H100"] as GpuKey[]).filter((k) => !keys.includes(k)).join(" or ")}`, tone: color.faint }] : []),
        ]}
      />
    </Figure>
  );
}
