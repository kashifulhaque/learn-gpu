import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// Constants, from docs/FACTS.md sections 2.1, 8.1 and 8.2. Peak dense BF16
// or FP16 matrix rate in TFLOP/s and HBM bandwidth in GB/s per chip or GPU.
// TPU v5e lists 800 GiB/s, which is 800 x 1.0737 = 858.99 GB/s. The B200 row
// is the HGX per-GPU figure. Ridge point = peak / bandwidth. Weight bytes for
// the decode floor: a 7-billion-parameter bf16 model holds 14 GB.
type Part = { name: string; vendor: "nvidia" | "amd" | "google"; tflops: number; gbs: number };
const PARTS: readonly Part[] = [
  { name: "A100 PCIe", vendor: "nvidia", tflops: 312, gbs: 1935 },
  { name: "H100 SXM5", vendor: "nvidia", tflops: 989.4, gbs: 3350 },
  { name: "B200 (HGX)", vendor: "nvidia", tflops: 2250, gbs: 8000 },
  { name: "MI300X", vendor: "amd", tflops: 1307.4, gbs: 5300 },
  { name: "MI325X", vendor: "amd", tflops: 1307.4, gbs: 6000 },
  { name: "MI355X", vendor: "amd", tflops: 2500, gbs: 8000 },
  { name: "TPU v5e", vendor: "google", tflops: 197, gbs: (800 * 1024 ** 3) / 1e9 },
  { name: "TPU v5p", vendor: "google", tflops: 459, gbs: 2765 },
  { name: "TPU v6e", vendor: "google", tflops: 918, gbs: 1638 },
  { name: "TPU7x", vendor: "google", tflops: 2307, gbs: 7380 },
];
const WEIGHTS_GB = 14;

type Metric = "ridge" | "decode" | "bf16";
const METRICS: readonly { value: Metric; label: string }[] = [
  { value: "ridge", label: "Ridge point" },
  { value: "decode", label: "Decode floor" },
  { value: "bf16", label: "Peak BF16" },
];
const TONE = { nvidia: color.ok, amd: color.bad, google: color.info } as const;

export default function AcceleratorRidge() {
  const [metric, setMetric] = useState<Metric>("ridge");
  const [ref, width] = useWidth<HTMLDivElement>();

  const value = (p: Part) =>
    metric === "ridge" ? p.tflops * 1e12 / (p.gbs * 1e9) : metric === "decode" ? (WEIGHTS_GB / p.gbs) * 1e3 : p.tflops;
  const unit = metric === "ridge" ? "FLOP/B" : metric === "decode" ? "ms" : "TFLOP/s";
  const digits = metric === "bf16" ? 0 : metric === "decode" ? 2 : 0;
  const max = Math.max(...PARTS.map(value));
  const labelW = 84;
  const valW = 92;
  const barW = Math.max(100, width - labelW - valW);
  const rowH = 20;
  const gap = 6;
  const height = PARTS.length * (rowH + gap) + 4;
  const ridge = (p: Part) => (p.tflops * 1e12) / (p.gbs * 1e9);
  const lo = PARTS.reduce((a, b) => (value(a) <= value(b) ? a : b));
  const hi = PARTS.reduce((a, b) => (value(a) >= value(b) ? a : b));

  return (
    <Figure
      title="Ten accelerators on one roofline axis"
      caption={
        <>
          Peak dense BF16 matrix rate divided by HBM bandwidth gives each part's ridge point. The decode floor is the time
          to read 14 GB of weights once at the rated bandwidth, a prediction and not a measurement. Different vendors rate
          peaks under different conditions, so read the bars as shapes, not as a league table.
        </>
      }
      controls={<Segmented label="Show" value={metric} options={METRICS} onChange={setMetric} />}
      readout={
        <>
          <Stat label={`Lowest (${unit})`} value={`${lo.name}: ${value(lo).toFixed(digits === 0 ? 0 : 2)}`} />
          <Stat label={`Highest (${unit})`} value={`${hi.name}: ${value(hi).toFixed(digits === 0 ? 0 : 2)}`} />
          <Stat label="A100 ridge" value={`${ridge(PARTS[0]).toFixed(0)} FLOP/B`} tone={color.a} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Bars of ${unit} for ten accelerators`}>
          {PARTS.map((p, i) => {
            const y = 2 + i * (rowH + gap);
            const v = value(p);
            const w = Math.max(2, (v / max) * barW);
            return (
              <g key={p.name}>
                <text x={0} y={y + rowH - 6} style={{ fill: color.muted, fontSize: 11 }}>
                  {p.name}
                </text>
                <rect x={labelW} y={y} width={barW} height={rowH} rx={3} style={{ fill: wash(color.muted, 10) }} />
                <rect x={labelW} y={y} width={w} height={rowH} rx={3} style={{ fill: TONE[p.vendor], opacity: 0.8 }} />
                <text x={labelW + barW + 6} y={y + rowH - 6} style={{ fill: color.fg, fontSize: 11 }}>
                  {v.toFixed(digits)} {unit}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "NVIDIA", tone: color.ok },
          { label: "AMD", tone: color.bad },
          { label: "Google TPU", tone: color.info },
        ]}
      />
    </Figure>
  );
}
