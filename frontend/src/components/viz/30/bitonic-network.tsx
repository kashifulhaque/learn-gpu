import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// The 8-element bitonic sorting network. Every stage and every array comes from
// scripts/figures/30-sort-top-k-and-sampling.py. A pair (i, j) with i < j puts the
// smaller value at i when the arrow points up, and at j when it points down.

const START = [7, 3, 6, 1, 5, 8, 2, 4];

type Pair = readonly [number, number, boolean];
type Stage = { k: number; j: number; pairs: readonly Pair[]; after: readonly number[] };

const STAGES: readonly Stage[] = [
  { k: 2, j: 1, pairs: [[0, 1, true], [2, 3, false], [4, 5, true], [6, 7, false]], after: [3, 7, 6, 1, 5, 8, 4, 2] },
  { k: 4, j: 2, pairs: [[0, 2, true], [1, 3, true], [4, 6, false], [5, 7, false]], after: [3, 1, 6, 7, 5, 8, 4, 2] },
  { k: 4, j: 1, pairs: [[0, 1, true], [2, 3, true], [4, 5, false], [6, 7, false]], after: [1, 3, 6, 7, 8, 5, 4, 2] },
  { k: 8, j: 4, pairs: [[0, 4, true], [1, 5, true], [2, 6, true], [3, 7, true]], after: [1, 3, 4, 2, 8, 5, 6, 7] },
  { k: 8, j: 2, pairs: [[0, 2, true], [1, 3, true], [4, 6, true], [5, 7, true]], after: [1, 2, 4, 3, 6, 5, 8, 7] },
  { k: 8, j: 1, pairs: [[0, 1, true], [2, 3, true], [4, 5, true], [6, 7, true]], after: [1, 2, 3, 4, 5, 6, 7, 8] },
];

export default function BitonicNetwork() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [stage, setStage] = useState(6);

  const values = stage === 0 ? START : STAGES[stage - 1].after;
  const current = stage === 0 ? null : STAGES[stage - 1];
  const before = stage <= 1 ? START : STAGES[stage - 2].after;

  const cell = Math.max(30, Math.min(56, Math.floor((width - 24) / 8)));
  const gx = 12;
  const rowH = 44;
    const svgW = Math.max(width, gx * 2 + 8 * cell);
  const x = (i: number) => gx + i * cell + (cell - 3) / 2;
  const barY = 96;

  return (
    <Figure
      title="The bitonic network on eight values"
      caption="Each stage compares the pairs i and i XOR j. Arcs that point right keep the smaller value on the left. The eight values need six stages of four compare-exchanges, 24 in all, and the order of the stages never depends on the data."
      controls={<Slider label="Stage" min={0} max={6} value={stage} onChange={setStage} />}
      readout={
        <>
          <Stat label="Block size k" value={current ? String(current.k) : "start"} tone={color.a} />
          <Stat label="Pair distance j" value={current ? String(current.j) : "none"} tone={color.b} />
          <Stat label="Compare-exchanges so far" value={String(stage * 4)} tone={color.accent} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={svgW}
          height={barY + rowH + 36}
          role="img"
          aria-label={`Bitonic network after stage ${stage} of 6. Array ${values.join(", ")}.`}
        >
          {current &&
            current.pairs.map(([i, j, up], p) => {
              const span = Math.abs(x(j) - x(i));
              const h = Math.min(80, 16 + (j - i) * 16);
              const swapped = (before[i] > before[j]) === up;
              return (
                <g key={`pair${p}`}>
                  <path
                    d={`M ${x(i)} ${barY - 4} C ${x(i)} ${barY - 4 - h}, ${x(j)} ${barY - 4 - h}, ${x(j)} ${barY - 4}`}
                    fill="none"
                    strokeWidth={2}
                    style={{ stroke: swapped ? color.accent : color.lineStrong }}
                  />
                  <text x={x(i) + span / 2} y={barY - 4 - h * 0.78} fontSize={11} textAnchor="middle" style={{ fill: swapped ? color.accent : color.muted }}>
                    {up ? "→" : "←"}
                  </text>
                </g>
              );
            })}
          {values.map((v, i) => (
            <g key={`v${i}`}>
              <rect
                x={gx + i * cell}
                y={barY}
                width={cell - 3}
                height={rowH}
                rx={4}
                style={{ fill: wash(color.b, 10 + v * 4), stroke: color.lineStrong }}
              />
              <text x={x(i)} y={barY + rowH / 2 + 5} fontSize={15} textAnchor="middle" style={{ fill: color.fg }}>
                {v}
              </text>
              <text x={x(i)} y={barY + rowH + 14} fontSize={10} textAnchor="middle" style={{ fill: color.faint }}>
                {i}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
