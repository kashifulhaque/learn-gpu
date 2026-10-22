// Steps through Kogge-Stone and Blelloch on a small array, so the reader sees
// which cells add into which, and counts the additions each network does.
// Every number is computed here from the input array; none is a measurement.

import { useEffect, useMemo, useState } from "react";
import { Figure, Segmented, Slider, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

type Algo = "ks" | "blelloch";
type Op = { from: number; to: number; kind: "add" | "copy" };
type Frame = { values: number[]; ops: Op[]; label: string };

// The first n of these are the input. The first eight are the chapter's example.
const INPUT = [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 1, 4, 0, 2, 6, 1];

// Kogge-Stone: at distance d, every cell from d onward adds the cell d places to its left.
function koggeStone(input: number[]): Frame[] {
  const n = input.length;
  const frames: Frame[] = [{ values: input.slice(), ops: [], label: "Input" }];
  for (let d = 1; d < n; d *= 2) {
    const prev = frames[frames.length - 1].values;
    const values = prev.slice();
    const ops: Op[] = [];
    for (let t = d; t < n; t++) {
      values[t] = prev[t] + prev[t - d];
      ops.push({ from: t - d, to: t, kind: "add" });
    }
    frames.push({ values, ops, label: `distance ${d}: add the cell ${d} to the left` });
  }
  return frames;
}

// Blelloch: an up-sweep that builds a tree of partial sums, then a down-sweep that pushes prefixes down.
function blelloch(input: number[]): Frame[] {
  const n = input.length;
  const frames: Frame[] = [{ values: input.slice(), ops: [], label: "Input" }];
  let offset = 1;
  for (let d = n >> 1; d > 0; d >>= 1) {
    const values = frames[frames.length - 1].values.slice();
    const ops: Op[] = [];
    for (let tid = 0; tid < d; tid++) {
      const a = offset * (2 * tid + 1) - 1;
      const b = offset * (2 * tid + 2) - 1;
      values[b] += values[a];
      ops.push({ from: a, to: b, kind: "add" });
    }
    frames.push({ values, ops, label: `up-sweep, stride ${offset}` });
    offset *= 2;
  }
  const cleared = frames[frames.length - 1].values.slice();
  cleared[n - 1] = 0;
  frames.push({ values: cleared, ops: [], label: "clear the root" });
  for (let d = 1; d < n; d *= 2) {
    offset >>= 1;
    const values = frames[frames.length - 1].values.slice();
    const ops: Op[] = [];
    for (let tid = 0; tid < d; tid++) {
      const a = offset * (2 * tid + 1) - 1;
      const b = offset * (2 * tid + 2) - 1;
      const tmp = values[a];
      values[a] = values[b];
      values[b] += tmp;
      ops.push({ from: b, to: a, kind: "copy" });
      ops.push({ from: a, to: b, kind: "add" });
    }
    frames.push({ values, ops, label: `down-sweep, stride ${offset}` });
  }
  return frames;
}

const SIZES = [4, 8, 16] as const;
const ROW = 46; // pixels per row
const CELL_H = 26;

export default function ScanNetworks() {
  const [ref, width] = useWidth();
  const [algo, setAlgo] = useState<Algo>("blelloch");
  const [n, setN] = useState<number>(8);

  const frames = useMemo(() => (algo === "ks" ? koggeStone(INPUT.slice(0, n)) : blelloch(INPUT.slice(0, n))), [algo, n]);
  const stepper = useStepper(frames.length, 900);

  // Start on the finished network, so the printed figure shows every step.
  // Play restarts from the input.
  useEffect(() => {
    stepper.setStep(frames.length - 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames]);

  const logn = Math.log2(n);
  const totalAdds = algo === "ks" ? n * logn - n + 1 : 2 * (n - 1);
  const stepsTotal = algo === "ks" ? logn : 2 * logn;
  const addsSoFar = frames.slice(0, stepper.step + 1).reduce((s, f) => s + f.ops.filter((o) => o.kind === "add").length, 0);

  const pad = 10;
  const cw = Math.max(18, (width - 2 * pad) / n);
  const cellW = cw - 4;
  const height = frames.length * ROW + 8;
  const cx = (i: number) => pad + i * cw + cw / 2;
  const cy = (row: number) => 6 + row * ROW + CELL_H / 2;
  const frame = frames[Math.min(stepper.step, frames.length - 1)];
  const final = frames[frames.length - 1].values;

  return (
    <Figure
      title="Two scan networks on one array"
      controls={
        <>
          <Segmented
            label="Algorithm"
            value={algo}
            options={[
              { value: "ks", label: "Kogge-Stone" },
              { value: "blelloch", label: "Blelloch" },
            ]}
            onChange={(v) => setAlgo(v)}
          />
          <Slider label="Elements" values={SIZES} value={n} onChange={setN} />
          <StepControls stepper={stepper} count={frames.length} describe={() => frame.label} />
        </>
      }
      readout={
        <>
          <Stat label="Algorithm" value={algo === "ks" ? "Kogge-Stone (inclusive)" : "Blelloch (exclusive)"} tone={color.a} />
          <Stat label={`Adds for n = ${n}`} value={`${totalAdds} (${addsSoFar} so far)`} tone={color.b} />
          <Stat label="Sequential steps" value={stepsTotal} tone={color.c} />
          <Stat label="One serial loop" value={`${n - 1} adds, ${n - 1} steps`} />
        </>
      }
      caption={
        <>
          Each row is the array after one step, and each arrow is an addition. Kogge-Stone finishes in log2 n steps but does{" "}
          n log2 n - n + 1 adds. Blelloch does 2(n - 1) adds, close to the serial loop, in 2 log2 n steps.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`The array after each step of ${algo === "ks" ? "Kogge-Stone" : "Blelloch"} on ${n} elements. The final row is ${final.join(", ")}.`}>
          {frames.slice(0, stepper.step + 1).map((f, row) => {
            const prev = row > 0 ? frames[row - 1].values : null;
            return (
              <g key={row}>
                {f.ops.map((op, k) => (
                  <line
                    key={k}
                    x1={cx(op.from)}
                    y1={cy(row - 1) + CELL_H / 2}
                    x2={cx(op.to)}
                    y2={cy(row) - CELL_H / 2}
                    style={{ stroke: op.kind === "add" ? color.a : color.c, strokeWidth: 1.5, opacity: 0.85 }}
                    strokeDasharray={op.kind === "copy" ? "4 3" : undefined}
                  />
                ))}
                {f.values.map((v, i) => {
                  const changed = prev !== null && prev[i] !== v;
                  return (
                    <g key={i}>
                      <rect
                        x={cx(i) - cellW / 2}
                        y={cy(row) - CELL_H / 2}
                        width={cellW}
                        height={CELL_H}
                        rx={4}
                        style={{ fill: changed ? wash(color.a, 22) : color.well, stroke: changed ? color.a : color.line }}
                      />
                      <text x={cx(i)} y={cy(row) + 4.5} textAnchor="middle" fontSize={12} style={{ fill: color.fg }}>
                        {v}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
