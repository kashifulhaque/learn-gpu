import { useMemo, useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// Synthetic data: 256 weights drawn from a standard normal with a fixed seed,
// so the drawing is the same on every render. The quantizer is the lab's:
// asymmetric 4-bit codes 0..15, one fp16 scale and one 8-bit zero point per
// group, with 0 included in the range. Bits per weight is 4 + (16 + 8) / g,
// the lab's layout (fp16 scale, uint8 zero point per group).
const COUNT = 256;
const OUTLIER_INDEX = 150;
const OUTLIER_VALUE = 8;

function normals(count: number): number[] {
  let state = 12345;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  const out: number[] = [];
  while (out.length < count) {
    const r = Math.sqrt(-2 * Math.log(uniform()));
    const t = 2 * Math.PI * uniform();
    out.push(r * Math.cos(t), r * Math.sin(t));
  }
  return out.slice(0, count);
}

const BASE = normals(COUNT);

function quantize(weights: number[], group: number) {
  const dq: number[] = new Array(weights.length);
  const steps: number[] = [];
  for (let start = 0; start < weights.length; start += group) {
    const slice = weights.slice(start, start + group);
    const lo = Math.min(0, ...slice);
    const hi = Math.max(0, ...slice);
    const scale = hi - lo > 0 ? (hi - lo) / 15 : 1;
    const zero = Math.min(15, Math.max(0, Math.round(-lo / scale)));
    steps.push(scale);
    slice.forEach((w, i) => {
      const code = Math.min(15, Math.max(0, Math.round(w / scale) + zero));
      dq[start + i] = (code - zero) * scale;
    });
  }
  return { dq, steps };
}

export default function GroupScales() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [group, setGroup] = useState<number>(128);
  const [outlier, setOutlier] = useState<"off" | "on">("on");

  const weights = useMemo(() => {
    const w = BASE.slice();
    if (outlier === "on") w[OUTLIER_INDEX] = OUTLIER_VALUE;
    return w;
  }, [outlier]);

  const { dq, steps } = useMemo(() => quantize(weights, group), [weights, group]);
  const err = weights.map((w, i) => Math.abs(w - dq[i]));
  const norm = Math.sqrt(weights.reduce((s, w) => s + w * w, 0));
  const relErr = Math.sqrt(err.reduce((s, e) => s + e * e, 0)) / norm;
  const bits = 4 + 24 / group;

  const m = { l: 40, r: 12, t: 10, b: 24 };
  const pw = Math.max(120, width - m.l - m.r);
  const topH = 120;
  const gap = 22;
  const botH = 60;
  const h = m.t + topH + gap + botH + m.b;
  const x = (i: number) => m.l + ((i + 0.5) / COUNT) * pw;
  const yTop = (v: number) => m.t + topH / 2 - (v / 9) * (topH / 2);
  const botTop = m.t + topH + gap;
  const yErr = (e: number) => botTop + botH - (Math.min(e, 0.9) / 0.9) * botH;

  const path = (values: number[]) => values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${yTop(v).toFixed(1)}`).join(" ");
  const groups = Array.from({ length: Math.ceil(COUNT / group) }, (_, g) => g);

  return (
    <Figure
      title="One scale per group of weights"
      caption="256 synthetic weights from a standard normal, quantized to 4 bits with one scale and one zero point per group. A single outlier stretches the step of its own group only. Bits per weight include an fp16 scale and an 8-bit zero point per group."
      controls={
        <>
          <Segmented
            label="Group size"
            value={group}
            onChange={setGroup}
            options={[
              { value: 256, label: "256 (whole row)" },
              { value: 128, label: "128" },
              { value: 64, label: "64" },
              { value: 32, label: "32" },
              { value: 16, label: "16" },
            ]}
          />
          <Segmented
            label="Outlier of 8 sigma"
            value={outlier}
            onChange={setOutlier}
            options={[
              { value: "off", label: "Off" },
              { value: "on", label: "On" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Relative error" value={`${(relErr * 100).toFixed(1)}%`} tone={color.a} />
          <Stat label="Largest step" value={Math.max(...steps).toFixed(2)} tone={color.c} />
          <Stat label="Bits per weight" value={bits.toFixed(3)} tone={color.b} />
          <Stat label="Groups" value={String(groups.length)} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="Weights and their 4-bit reconstructions, with the absolute error below, for the chosen group size">
          {groups.map((g) => {
            const x0 = m.l + ((g * group) / COUNT) * pw;
            const x1 = m.l + (Math.min(COUNT, (g + 1) * group) / COUNT) * pw;
            return (
              <rect
                key={g}
                x={x0}
                y={m.t}
                width={x1 - x0}
                height={topH + gap + botH}
                style={{ fill: g % 2 === 0 ? wash(color.tint, 55) : "transparent" }}
              />
            );
          })}
          <line x1={m.l} x2={m.l + pw} y1={yTop(0)} y2={yTop(0)} style={{ stroke: color.line }} />
          {[-4, 4, 8].map((v) => (
            <text key={v} x={m.l - 6} y={yTop(v) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
              {v}
            </text>
          ))}
          <path d={path(weights)} fill="none" strokeWidth={1.2} style={{ stroke: color.muted }} />
          <path d={path(dq)} fill="none" strokeWidth={1.6} style={{ stroke: color.a }} />
          {err.map((e, i) => (
            <line key={i} x1={x(i)} x2={x(i)} y1={botTop + botH} y2={yErr(e)} strokeWidth={Math.max(1, pw / COUNT - 0.5)} style={{ stroke: color.c }} />
          ))}
          <text x={m.l - 6} y={botTop + 10} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            err
          </text>
          <text x={m.l + pw / 2} y={h - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            weight index (shading marks groups)
          </text>
        </svg>
        <Legend
          items={[
            { label: "weight", tone: color.muted },
            { label: "dequantized", tone: color.a },
            { label: "absolute error", tone: color.c },
          ]}
        />
      </div>
    </Figure>
  );
}
