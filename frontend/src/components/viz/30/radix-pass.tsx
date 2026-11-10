import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// An LSD radix sort of eight 4-bit keys with a 2-bit digit, two passes.
// Every array below is the output of scripts/figures/30-sort-top-k-and-sampling.py.
// The letter after a key is its position in the original input, so the two
// equal keys, 6b and 6g, show whether a pass is stable.

type Item = readonly [number, string];

type Pass = {
  shift: number;
  input: readonly Item[];
  digits: readonly number[];
  counts: readonly number[];
  offsets: readonly number[];
  dest: readonly number[];
};

const START: readonly Item[] = [
  [13, "a"],
  [6, "b"],
  [9, "c"],
  [5, "d"],
  [3, "e"],
  [14, "f"],
  [6, "g"],
  [1, "h"],
];

const PASSES: readonly Pass[] = [
  {
    shift: 0,
    input: START,
    digits: [1, 2, 1, 1, 3, 2, 2, 1],
    counts: [0, 4, 3, 1],
    offsets: [0, 0, 4, 7],
    dest: [0, 4, 1, 2, 7, 5, 6, 3],
  },
  {
    shift: 2,
    input: [
      [13, "a"],
      [9, "c"],
      [5, "d"],
      [1, "h"],
      [6, "b"],
      [14, "f"],
      [6, "g"],
      [3, "e"],
    ],
    digits: [3, 2, 1, 0, 1, 3, 1, 0],
    counts: [2, 3, 1, 2],
    offsets: [0, 2, 5, 6],
    dest: [6, 5, 2, 0, 3, 7, 4, 1],
  },
];

const binary = (key: number) => key.toString(2).padStart(4, "0");

export default function RadixPassFigure() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [passIndex, setPassIndex] = useState(0);
  const [done, setDone] = useState(8);

  const pass = PASSES[passIndex];
  const n = pass.input.length;
  const cell = Math.max(30, Math.min(58, Math.floor((width - 24) / n)));
  const gx = 12;
  const inputY = 20;
  const binY = 112;
  const outY = 196;
  const boxH = 38;
  const svgW = Math.max(width, gx * 2 + n * cell);

  const slots: (Item | null)[] = Array(n).fill(null);
  for (let i = 0; i < done; i++) slots[pass.dest[i]] = pass.input[i];
  const cursor = pass.offsets.map((start, d) => start + pass.digits.slice(0, done).filter((v) => v === d).length);
  const next = done < n ? pass.dest[done] : null;
  const toneOf = (key: number) => (key === 6 ? color.accent : color.b);

  return (
    <Figure
      title="One stable radix pass, then the next"
      caption="Each pass sorts by one 2-bit digit. The pass reads the input from left to right and writes each key at the next free slot of its digit's bin. The two 6s keep their order, b before g, through both passes, and that is why the second pass doesn't undo the first."
      controls={
        <>
          <Segmented
            label="Pass"
            value={passIndex}
            options={[
              { value: 0, label: "1: low digit" },
              { value: 1, label: "2: high digit" },
            ]}
            onChange={(v) => {
              setPassIndex(v);
              setDone(8);
            }}
          />
          <Slider label="Keys scattered" min={0} max={8} value={done} onChange={setDone} />
        </>
      }
      readout={
        <>
          <Stat label="Counts" value={`[${pass.counts.join(", ")}]`} tone={color.b} />
          <Stat label="Exclusive scan" value={`[${pass.offsets.join(", ")}]`} tone={color.a} />
          <Stat label="Next free slots" value={`[${cursor.join(", ")}]`} tone={color.accent} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={svgW}
          height={outY + boxH + 24}
          role="img"
          aria-label={`Radix pass ${passIndex + 1}. Counts ${pass.counts.join(", ")}. Exclusive scan ${pass.offsets.join(", ")}. ${done} of ${n} keys scattered.`}
        >
          <text x={gx} y={12} fontSize={11} style={{ fill: color.muted }}>
            input, in order
          </text>
          {pass.input.map(([key, tag], i) => {
            const shown = i < done;
            const bits = binary(key);
            const lo = pass.shift === 0;
            const digitBits = lo ? bits.slice(2) : bits.slice(0, 2);
            const rest = lo ? bits.slice(0, 2) : bits.slice(2);
            return (
              <g key={`in${i}`} opacity={shown ? 0.45 : 1}>
                <rect
                  x={gx + i * cell}
                  y={inputY}
                  width={cell - 3}
                  height={boxH + 12}
                  rx={4}
                  style={{ fill: wash(toneOf(key), 18), stroke: i === done ? color.fg : "none" }}
                />
                <text x={gx + i * cell + (cell - 3) / 2} y={inputY + 18} fontSize={13} textAnchor="middle" style={{ fill: color.fg }}>
                  {key}
                  {tag}
                </text>
                <text x={gx + i * cell + (cell - 3) / 2} y={inputY + 38} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                  {lo ? (
                    <>
                      {rest}
                      <tspan style={{ fill: color.a, fontWeight: 700 }}>{digitBits}</tspan>
                    </>
                  ) : (
                    <>
                      <tspan style={{ fill: color.a, fontWeight: 700 }}>{digitBits}</tspan>
                      {rest}
                    </>
                  )}
                </text>
              </g>
            );
          })}

          {pass.input.slice(0, done).map(([key], i) => (
            <line
              key={`ln${i}`}
              x1={gx + i * cell + (cell - 3) / 2}
              y1={inputY + boxH + 12}
              x2={gx + pass.dest[i] * cell + (cell - 3) / 2}
              y2={outY}
              strokeWidth={key === 6 ? 2 : 1}
              style={{ stroke: key === 6 ? color.accent : color.faint }}
            />
          ))}

          <text x={gx} y={binY - 6} fontSize={11} style={{ fill: color.muted }}>
            bins: digit, count, start
          </text>
          {pass.counts.map((count, d) => {
            const span = (n * cell) / 4;
            return (
              <g key={`bin${d}`}>
                <rect
                  x={gx + d * span}
                  y={binY}
                  width={span - 6}
                  height={boxH}
                  rx={4}
                  style={{ fill: color.well, stroke: color.lineStrong }}
                />
                <text x={gx + d * span + (span - 6) / 2} y={binY + 16} fontSize={12} textAnchor="middle" style={{ fill: color.fg }}>
                  digit {d}: {count}
                </text>
                <text x={gx + d * span + (span - 6) / 2} y={binY + 31} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                  start {pass.offsets[d]}, next {cursor[d]}
                </text>
              </g>
            );
          })}

          <text x={gx} y={outY - 6} fontSize={11} style={{ fill: color.muted }}>
            output slots
          </text>
          {slots.map((item, s) => (
            <g key={`out${s}`}>
              <rect
                x={gx + s * cell}
                y={outY}
                width={cell - 3}
                height={boxH}
                rx={4}
                style={{
                  fill: item ? wash(toneOf(item[0]), 28) : "none",
                  stroke: s === next ? color.fg : color.line,
                  strokeDasharray: item ? "none" : "3 3",
                }}
              />
              <text x={gx + s * cell + (cell - 3) / 2} y={outY + 24} fontSize={13} textAnchor="middle" style={{ fill: color.fg }}>
                {item ? `${item[0]}${item[1]}` : ""}
              </text>
              <text x={gx + s * cell + (cell - 3) / 2} y={outY + boxH + 14} fontSize={10} textAnchor="middle" style={{ fill: color.faint }}>
                {s}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
