import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Floors in milliseconds, printed by scripts/figures/41-capstone-kernel-leaderboard.py
// from the A100 constants in docs/FACTS.md: 1275 GB/s copy ceiling, 19.5 TFLOP/s
// fp32, 312 TFLOP/s bf16. They are predictions for the ideal fused kernel.
const TASKS = [
  { value: "gray", label: "Grayscale and blur", floor: 0.2105, shape: "4096 by 4096 image" },
  { value: "topk", label: "Top-8", floor: 0.4115, shape: "4096 rows of 32000" },
  { value: "mlp", label: "Fused MLP", floor: 0.2203, shape: "M 4096, d 1024, h 4096" },
] as const;
type TaskKey = (typeof TASKS)[number]["value"];

// score = log(ref / t) / log(ref / floor), clipped to [0, 1]. The linear form is
// (ref - t) / (ref - floor). Times here are multiples of the floor.
function logScore(t: number, ref: number): number {
  return Math.min(1, Math.max(0, Math.log(ref / t) / Math.log(ref)));
}
function linearScore(t: number, ref: number): number {
  return Math.min(1, Math.max(0, (ref - t) / (ref - 1)));
}

export default function ScoreCurve() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [task, setTask] = useState<TaskKey>("gray");
  const [refMult, setRefMult] = useState(4);
  const [timeMult, setTimeMult] = useState(2);

  const info = TASKS.find((t) => t.value === task)!;
  const t = Math.min(timeMult, refMult * 1.25);
  const w = Math.max(width, 320);
  const h = 220;
  const left = 40;
  const right = 12;
  const top = 12;
  const bottom = 30;
  const xMax = refMult * 1.25;
  const x = (m: number) => left + ((m - 1) / (xMax - 1)) * (w - left - right);
  const y = (s: number) => top + (1 - s) * (h - top - bottom);
  const points = (fn: (m: number) => number) =>
    Array.from({ length: 61 }, (_, i) => {
      const m = 1 + (i / 60) * (xMax - 1);
      return `${x(m).toFixed(1)},${y(fn(m)).toFixed(1)}`;
    }).join(" ");

  const sLog = logScore(t, refMult);
  const sLin = linearScore(t, refMult);

  return (
    <Figure
      title="From a time to a score"
      caption="The score is the share of the headroom between the reference and the floor that your kernel closes, on a log scale. Halfway in time ratio, not in milliseconds, scores 0.5. A linear score is higher everywhere between the two ends: at 2x the floor, with a reference at 4x, it gives 0.667 where the log score gives 0.5. The floors are predictions for the ideal fused kernel at the 1275 GB/s copy ceiling and the rated compute peak; the lab measures the real reference."
      controls={
        <>
          <Segmented label="Task" value={task} options={TASKS.map((o) => ({ value: o.value, label: o.label }))} onChange={setTask} />
          <Slider label="Reference time" min={1.5} max={10} step={0.5} value={refMult} onChange={setRefMult} format={(v) => `${v}x floor`} />
          <Slider label="Your time" min={1} max={10} step={0.25} value={timeMult} onChange={setTimeMult} format={(v) => `${v}x floor`} />
        </>
      }
      readout={
        <>
          <Stat label="Floor" value={`${info.floor.toFixed(4)} ms`} />
          <Stat label="Reference" value={`${(info.floor * refMult).toFixed(3)} ms`} tone={color.b} />
          <Stat label="Your time" value={`${(info.floor * t).toFixed(3)} ms`} tone={color.a} />
          <Stat label="Log score" value={sLog.toFixed(3)} tone={color.a} />
          <Stat label="Linear score" value={sLin.toFixed(3)} tone={color.c} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Score against time for ${info.label} on ${info.shape}. At ${t} times the floor with a reference of ${refMult} times the floor, the log score is ${sLog.toFixed(3)} and the linear score is ${sLin.toFixed(3)}.`}
        >
          {[0, 0.5, 1].map((s) => (
            <g key={s}>
              <line x1={left} x2={w - right} y1={y(s)} y2={y(s)} style={{ stroke: color.line }} />
              <text x={left - 6} y={y(s) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {s}
              </text>
            </g>
          ))}
          <polyline points={points((m) => logScore(m, refMult))} fill="none" strokeWidth={2} style={{ stroke: color.a }} />
          <polyline points={points((m) => linearScore(m, refMult))} fill="none" strokeWidth={2} strokeDasharray="5 4" style={{ stroke: color.c }} />
          <line x1={x(t)} x2={x(t)} y1={top} y2={h - bottom} style={{ stroke: color.lineStrong }} />
          <circle cx={x(t)} cy={y(sLog)} r={5} style={{ fill: color.a }} />
          <circle cx={x(t)} cy={y(sLin)} r={4} style={{ fill: color.c }} />
          <text x={left} y={h - 8} fontSize={11} style={{ fill: color.muted }}>
            floor
          </text>
          <text x={x(refMult)} y={h - 8} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            reference
          </text>
          <text x={w - right} y={h - 8} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            time, in multiples of the floor
          </text>
        </svg>
        <Legend
          items={[
            { label: "Log score", tone: color.a },
            { label: "Linear score", tone: color.c, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
