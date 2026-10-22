import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Link rates are per direction, in GB/s, from docs/FACTS.md section 7.1:
// PCIe Gen4 x16 31.5 (16 GT/s x 128/130 x 16 / 8), NVLink 3 on the A100 300 (600 total),
// NVLink 4 on the H100 450 (900 total), InfiniBand NDR 50 (400 Gb/s / 8).
const LINKS = [
  { value: "pcie4", label: "PCIe Gen4", gbs: 31.5 },
  { value: "nvlink3", label: "NVLink 3", gbs: 300 },
  { value: "nvlink4", label: "NVLink 4", gbs: 450 },
  { value: "ib", label: "IB NDR", gbs: 50 },
] as const;

// The models, as the chapter and the lab write them.
const ring = (p: number, n: number, a: number, b: number) => 2 * (p - 1) * (a + n / p / b);
const tree = (p: number, n: number, a: number, b: number) => 2 * Math.ceil(Math.log2(p)) * a + (2 * n) / b;
const crossover = (p: number, a: number, b: number) => a * b * p * (p - 1 - Math.ceil(Math.log2(p)));

const MIN_LOG = 10; // 1 KiB
const MAX_LOG = 34; // 16 GiB

function fmtTime(s: number): string {
  if (s < 1e-3) return `${(s * 1e6).toFixed(1)} us`;
  if (s < 1) return `${(s * 1e3).toFixed(2)} ms`;
  return `${s.toFixed(2)} s`;
}

export default function AllreduceCost() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [link, setLink] = useState<string>("nvlink3");
  const [p, setP] = useState<number>(8);
  const [alphaUs, setAlphaUs] = useState<number>(5);
  const [logN, setLogN] = useState<number>(30);

  const gbs = LINKS.find((l) => l.value === link)!.gbs;
  const B = gbs * 1e9;
  const a = alphaUs * 1e-6;
  const n = 2 ** logN;
  const tRing = ring(p, n, a, B);
  const tTree = tree(p, n, a, B);
  const x = crossover(p, a, B);
  const winner = tTree < tRing ? "tree" : "ring";
  const busbw = ((n / tRing / 1e9) * (2 * (p - 1))) / p;

  const h = 210;
  const m = { l: 56, r: 12, t: 10, b: 32 };
  const pw = Math.max(140, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const lo = Math.log10(1e-6);
  const hi = Math.log10(10);
  const sx = (lg: number) => m.l + ((lg - MIN_LOG) / (MAX_LOG - MIN_LOG)) * pw;
  const sy = (t: number) => m.t + ph - ((Math.log10(Math.max(t, 1e-6)) - lo) / (hi - lo)) * ph;
  const path = (f: (n: number) => number) => {
    const pts: string[] = [];
    for (let lg = MIN_LOG; lg <= MAX_LOG; lg += 0.5) pts.push(`${lg === MIN_LOG ? "M" : "L"}${sx(lg).toFixed(1)},${sy(f(2 ** lg)).toFixed(1)}`);
    return pts.join(" ");
  };
  const xTicks = [10, 20, 30];
  const yTicks = [-6, -3, 0];

  return (
    <Figure
      title="What an all-reduce costs"
      caption="Time against message size for a ring and an idealized tree, from the alpha-beta model. Small messages pay for latency, so the tree wins. Large messages pay for bandwidth, so the ring wins. The dashed line is the crossover size."
      controls={
        <>
          <Segmented label="Link, per direction" value={link} onChange={setLink} options={LINKS.map((l) => ({ value: l.value, label: l.label }))} />
          <Segmented label="Ranks" value={p} onChange={setP} options={[2, 4, 8, 16, 64].map((v) => ({ value: v, label: String(v) }))} />
          <Slider label="Latency alpha" min={1} max={20} step={1} value={alphaUs} onChange={setAlphaUs} format={(v) => `${v} us`} />
          <Slider label="Message size" min={MIN_LOG} max={MAX_LOG} step={1} value={logN} onChange={setLogN} format={(v) => formatBytes(2 ** v, 0)} />
        </>
      }
      readout={
        <>
          <Stat label={`Link, ${p} ranks`} value={`${gbs} GB/s, alpha ${alphaUs} us`} />
          <Stat label="Ring" value={fmtTime(tRing)} tone={color.a} />
          <Stat label="Tree" value={fmtTime(tTree)} tone={color.b} />
          <Stat label="Faster here" value={winner} tone={color.c} />
          <Stat label="Ring busbw at this size" value={`${busbw.toFixed(1)} GB/s`} />
          <Stat label="Crossover" value={formatBytes(Math.max(x, 0), 1)} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label={`Ring and tree all-reduce time against size for ${p} ranks on ${gbs} GB/s links. At ${formatBytes(n, 0)} the ring takes ${fmtTime(tRing)} and the tree ${fmtTime(tTree)}.`}>
          {yTicks.map((e) => (
            <g key={e}>
              <line x1={m.l} x2={m.l + pw} y1={sy(10 ** e)} y2={sy(10 ** e)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={sy(10 ** e) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {e === -6 ? "1 us" : e === -3 ? "1 ms" : "1 s"}
              </text>
            </g>
          ))}
          {xTicks.map((lg) => (
            <text key={lg} x={sx(lg)} y={h - 12} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              {formatBytes(2 ** lg, 0)}
            </text>
          ))}
          <path d={path((v) => ring(p, v, a, B))} fill="none" strokeWidth={2} style={{ stroke: color.a }} />
          <path d={path((v) => tree(p, v, a, B))} fill="none" strokeWidth={2} style={{ stroke: color.b }} />
          {x > 2 ** MIN_LOG && x < 2 ** MAX_LOG && (
            <line x1={sx(Math.log2(x))} x2={sx(Math.log2(x))} y1={m.t} y2={m.t + ph} style={{ stroke: color.c, strokeDasharray: "4 3" }} />
          )}
          <line x1={sx(logN)} x2={sx(logN)} y1={m.t} y2={m.t + ph} style={{ stroke: color.accent, strokeDasharray: "2 3" }} />
          <circle cx={sx(logN)} cy={sy(tRing)} r={5} strokeWidth={2} style={{ fill: wash(color.a, 35), stroke: color.a }} />
          <circle cx={sx(logN)} cy={sy(tTree)} r={5} strokeWidth={2} style={{ fill: wash(color.b, 35), stroke: color.b }} />
        </svg>
        <Legend
          items={[
            { label: "Ring", tone: color.a },
            { label: "Tree (idealized)", tone: color.b },
            { label: "Crossover", tone: color.c, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
