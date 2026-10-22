import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// A schedule simulation, not a measurement.
//
// Model, from the chapter (Narayanan et al. 2021, "Efficient large-scale
// language model training on GPU clusters using Megatron-LM"):
// - p stages, m microbatches. A forward step takes 1 unit and a backward step 2
//   units (the usual factor of two). Communication between stages is free.
// - Bubble time over ideal time = (p - 1) / m, for GPipe and for 1F1B.
// - Interleaving with v chunks per device divides the bubble by v.
// - GPipe stashes activations for all m microbatches; 1F1B for at most p.
const TF = 1;
const TB = 2;

type Op = { kind: "F" | "B"; mb: number };
type Schedule = "gpipe" | "1f1b";

function opsFor(stage: number, p: number, m: number, schedule: Schedule): Op[] {
  const ops: Op[] = [];
  if (schedule === "gpipe") {
    for (let j = 0; j < m; j++) ops.push({ kind: "F", mb: j });
    for (let j = 0; j < m; j++) ops.push({ kind: "B", mb: j });
    return ops;
  }
  const warm = Math.min(p - stage - 1, m);
  for (let j = 0; j < warm; j++) ops.push({ kind: "F", mb: j });
  for (let i = 0; i < m - warm; i++) {
    ops.push({ kind: "F", mb: warm + i });
    ops.push({ kind: "B", mb: i });
  }
  for (let i = m - warm; i < m; i++) ops.push({ kind: "B", mb: i });
  return ops;
}

function simulate(p: number, m: number, schedule: Schedule) {
  const lists = Array.from({ length: p }, (_, s) => opsFor(s, p, m, schedule));
  const index = new Map<string, number>();
  lists.forEach((list, s) => list.forEach((op, i) => index.set(`${s}:${op.kind}:${op.mb}`, i)));
  const memo = new Map<string, { start: number; end: number }>();
  const run = (s: number, i: number): { start: number; end: number } => {
    const key = `${s}:${i}`;
    const hit = memo.get(key);
    if (hit) return hit;
    const op = lists[s][i];
    let ready = i > 0 ? run(s, i - 1).end : 0;
    if (op.kind === "F" && s > 0) {
      ready = Math.max(ready, run(s - 1, index.get(`${s - 1}:F:${op.mb}`)!).end);
    }
    if (op.kind === "B") {
      const dep =
        s < p - 1
          ? run(s + 1, index.get(`${s + 1}:B:${op.mb}`)!).end
          : run(s, index.get(`${s}:F:${op.mb}`)!).end;
      ready = Math.max(ready, dep);
    }
    const out = { start: ready, end: ready + (op.kind === "F" ? TF : TB) };
    memo.set(key, out);
    return out;
  };
  const bars: { stage: number; kind: "F" | "B"; mb: number; start: number; end: number }[] = [];
  let makespan = 0;
  for (let s = 0; s < p; s++) {
    for (let i = 0; i < lists[s].length; i++) {
      const t = run(s, i);
      bars.push({ stage: s, kind: lists[s][i].kind, mb: lists[s][i].mb, ...t });
      makespan = Math.max(makespan, t.end);
    }
  }
  return { bars, makespan };
}

export default function PipelineBubble() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [p, setP] = useState(4);
  const [m, setM] = useState(8);
  const [v, setV] = useState(1);
  const [schedule, setSchedule] = useState<Schedule>("1f1b");

  const { bars, makespan } = simulate(p, m, schedule);
  const busy = m * (TF + TB);
  const idle = 1 - busy / makespan;
  const formula = (p - 1) / m;
  const inflight = schedule === "gpipe" ? m : Math.min(p, m);

  const left = 64;
  const right = 12;
  const plotW = Math.max(120, width - left - right);
  const rowH = 22;
  const gap = 4;
  const x = (t: number) => left + (t / makespan) * plotW;
  const height = 8 + p * (rowH + gap) + 24;

  return (
    <Figure
      title="The pipeline bubble: GPipe against 1F1B"
      controls={
        <>
          <Slider label="Stages p" value={p} min={2} max={8} onChange={setP} />
          <Slider label="Microbatches m" value={m} min={1} max={16} onChange={setM} />
          <Slider label="Interleave v (formula only)" value={v} values={[1, 2, 4]} onChange={setV} />
          <Segmented
            label="Schedule"
            value={schedule}
            onChange={setSchedule}
            options={[
              { value: "gpipe", label: "GPipe" },
              { value: "1f1b", label: "1F1B" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Idle share of wall-clock (simulated)" value={`${(idle * 100).toFixed(1)}%`} tone={color.bad} />
          <Stat label="Bubble over ideal, (p-1)/m" value={`${(formula * 100).toFixed(1)}%`} />
          <Stat label={`With interleave v = ${v}`} value={`${((formula / v) * 100).toFixed(1)}%`} tone={color.ok} />
          <Stat label="Microbatches stashed, peak" value={inflight} tone={color.b} />
        </>
      }
      caption={
        <>
          {p} stages and {m} microbatches under {schedule === "gpipe" ? "GPipe" : "1F1B"}. A forward step takes one
          unit and a backward step two, and communication is free. The simulated idle share is the bubble over the
          wall-clock time, (p-1)/(m+p-1); the bubble over ideal time is (p-1)/m, the figure the Megatron-LM paper
          quotes. The two schedules have the same bubble, and 1F1B stashes at most p microbatches where GPipe stashes m.
          The interleave readout applies the paper's 1/v factor and isn't drawn.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Schedule of ${m} microbatches on ${p} stages`}>
          {Array.from({ length: p }, (_, s) => {
            const y = 8 + s * (rowH + gap);
            return (
              <g key={s}>
                <text x={left - 8} y={y + rowH / 2 + 4} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
                  Stage {s + 1}
                </text>
                <rect x={left} y={y} width={plotW} height={rowH} rx={3} style={{ fill: color.well }} />
              </g>
            );
          })}
          {bars.map((bar) => {
            const y = 8 + bar.stage * (rowH + gap);
            const x0 = x(bar.start);
            const w = Math.max(0.5, x(bar.end) - x0);
            const tone = bar.kind === "F" ? color.b : color.c;
            return (
              <g key={`${bar.stage}-${bar.kind}-${bar.mb}`}>
                <rect
                  x={x0}
                  y={y + 1}
                  width={w}
                  height={rowH - 2}
                  rx={2}
                  style={{ fill: wash(tone, 40), stroke: tone, strokeWidth: 1 }}
                />
                {w > 13 && (
                  <text x={x0 + w / 2} y={y + rowH / 2 + 4} textAnchor="middle" style={{ fill: color.fg, fontSize: 10 }}>
                    {bar.mb + 1}
                  </text>
                )}
              </g>
            );
          })}
          <text x={left} y={height - 6} style={{ fill: color.subtle, fontSize: 10 }}>
            0
          </text>
          <text x={left + plotW} y={height - 6} textAnchor="end" style={{ fill: color.subtle, fontSize: 10 }}>
            {makespan} units (labels are microbatch numbers)
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Forward (1 unit)", tone: color.b },
          { label: "Backward (2 units)", tone: color.c },
        ]}
      />
    </Figure>
  );
}
