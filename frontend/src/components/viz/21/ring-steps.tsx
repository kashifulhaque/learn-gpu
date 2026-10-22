import { useState } from "react";
import { Figure, Segmented, StepControls, Stat, color, useStepper, useWidth, wash } from "../kit";

// A ring all-reduce over p ranks, simulated exactly as the lab's solution runs it.
// Rank r sends to (r + 1) % p. Reduce-scatter, step s: rank r sends chunk (r - s) % p and
// adds the chunk (r - s - 1) % p it receives. All-gather, step s: rank r sends chunk
// (r + 1 - s) % p and overwrites chunk (r - s) % p. Each step moves n / p bytes per rank.
// The cell shows how many ranks' contributions the chunk holds; "all" means fully reduced.

type Cell = { count: number; done: boolean };

function simulate(p: number, steps: number): Cell[][] {
  const grid: Cell[][] = Array.from({ length: p }, () => Array.from({ length: p }, () => ({ count: 1, done: p === 1 })));
  const mod = (a: number) => ((a % p) + p) % p;
  for (let k = 0; k < steps; k++) {
    const next = grid.map((row) => row.map((c) => ({ ...c })));
    if (k < p - 1) {
      const s = k;
      for (let r = 0; r < p; r++) {
        const prev = mod(r - 1);
        const chunk = mod(prev - s); // what rank r - 1 sends in step s, which rank r receives
        next[r][chunk] = { count: grid[r][chunk].count + grid[prev][chunk].count, done: false };
      }
    } else {
      const s = k - (p - 1);
      for (let r = 0; r < p; r++) {
        const prev = mod(r - 1);
        const chunk = mod(prev + 1 - s);
        next[r][chunk] = { count: p, done: true };
      }
    }
    for (let r = 0; r < p; r++) for (let c = 0; c < p; c++) grid[r][c] = next[r][c];
  }
  // A chunk becomes "done" in reduce-scatter's last step at its owner.
  if (steps >= p - 1) {
    for (let r = 0; r < p; r++) grid[r][mod(r + 1)] = { count: p, done: true };
  }
  return grid;
}

export default function RingSteps() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [p, setP] = useState<number>(4);
  const total = 2 * (p - 1);
  const stepper = useStepper(total + 1, 900);
  const done = Math.min(stepper.step, total);
  const grid = simulate(p, done);

  const phase = done === 0 ? "start" : done <= p - 1 ? "reduce-scatter" : "all-gather";
  const sentFraction = (done * 1) / p; // fraction of n each rank has sent so far
  const label = 74;
  const cell = Math.max(28, Math.min(54, Math.floor((width - label - 8) / p)));
  const h = 22 + p * cell + 4;

  const describe = (s: number) => (s === 0 ? "before any step" : s <= p - 1 ? `reduce-scatter step ${s}` : `all-gather step ${s - (p - 1)}`);

  return (
    <Figure
      title="Ring all-reduce, one step at a time"
      caption="Row r is rank r, and each column is one of p chunks of the array. A cell counts how many ranks' contributions that rank's copy of the chunk holds. After p - 1 steps each rank owns one complete chunk, and after 2(p - 1) steps every cell is complete."
      controls={
        <>
          <Segmented
            label="Ranks"
            value={p}
            onChange={(v) => {
              setP(v);
              stepper.reset();
            }}
            options={[3, 4, 5, 6].map((v) => ({ value: v, label: String(v) }))}
          />
          <StepControls stepper={stepper} count={total + 1} describe={describe} />
        </>
      }
      readout={
        <>
          <Stat label="Phase" value={phase} tone={phase === "all-gather" ? color.b : color.a} />
          <Stat label="Steps done" value={`${done} of ${total}`} />
          <Stat label="Sent per rank so far" value={`${done} x n/${p} = ${sentFraction.toFixed(2)} n`} />
          <Stat label="Total at the end" value={`2(p-1)/p = ${((2 * (p - 1)) / p).toFixed(3)} n`} tone={color.c} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label={`Ring all-reduce over ${p} ranks after ${done} of ${total} steps, in the ${phase} phase.`}>
          {Array.from({ length: p }, (_, c) => (
            <text key={c} x={label + c * cell + cell / 2} y={14} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              chunk {c}
            </text>
          ))}
          {grid.map((row, r) => (
            <g key={r} transform={`translate(0, ${22 + r * cell})`}>
              <text x={label - 8} y={cell / 2 + 4} textAnchor="end" fontSize={12} style={{ fill: color.fg }}>
                rank {r}
              </text>
              {row.map((cellState, c) => {
                const full = cellState.count === p;
                const tone = full ? color.a : cellState.count > 1 ? color.b : color.faint;
                return (
                  <g key={c} transform={`translate(${label + c * cell}, 0)`}>
                    <rect
                      x={1}
                      y={1}
                      width={cell - 2}
                      height={cell - 2}
                      rx={3}
                      style={{ fill: wash(tone, full ? 45 : cellState.count > 1 ? 30 : 15), stroke: full ? color.a : color.line }}
                    />
                    <text x={cell / 2} y={cell / 2 + 4} textAnchor="middle" fontSize={12} style={{ fill: color.fg }}>
                      {full ? "all" : cellState.count}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
