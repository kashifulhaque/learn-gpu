import { useState } from "react";
import { Figure, Slider, StepControls, Stat, color, useStepper, useWidth, wash } from "../kit";

// An 8 x 8 tile stands in for the lab's 32 x 32 (float32) or 64 x 64 (bfloat16)
// tile, so every cell can be read. Cell text "rc" names the input element at
// row r, column c. The shared tile holds the input tile unchanged; the output
// tile holds its transpose.
const N = 8;
const STEPS = [
  "Global read: one input row",
  "Store the row into the shared tile",
  "Barrier: __syncthreads()",
  "Shared read: one column",
  "Global write: one output row",
];

const PANELS = [
  { title: "Input tile (global)", note: "row-major, read by rows" },
  { title: "Shared tile", note: "same shape as the input" },
  { title: "Output tile (global)", note: "row-major, written by rows" },
];

export default function TilePath() {
  const stepper = useStepper(STEPS.length, 1100);
  const [k, setK] = useState(2);
  const [ref, width] = useWidth<HTMLDivElement>();
  const step = stepper.step;

  const stacked = width < 480;
  const gap = 18;
  const cell = stacked
    ? Math.min(26, Math.floor((width - 24) / N))
    : Math.min(28, Math.floor((width - 2 * gap - 16) / (3 * N)));
  const side = cell * N;
  const labelH = 34;
  const panelH = side + labelH;
  const height = stacked ? 3 * panelH + 2 * 10 : panelH + 8;

  const grid = (panel: number, ox: number, oy: number) => {
    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        let active = false;
        let filled = false;
        let label = `${r}${c}`;
        if (panel === 0) {
          active = step === 0 && r === k;
          filled = true;
        } else if (panel === 1) {
          active = (step === 1 && r === k) || (step === 3 && c === k);
          filled = step >= 2 || (step === 1 && r === k);
          if (!filled) label = "";
        } else {
          active = step === 4 && r === k;
          filled = step >= 4 && r <= k;
          label = filled ? `${c}${r}` : "";
        }
        cells.push(
          <g key={`${panel}-${r}-${c}`}>
            <rect
              x={ox + c * cell + 0.5}
              y={oy + r * cell + 0.5}
              width={cell - 1}
              height={cell - 1}
              rx={2}
              style={{
                fill: active ? wash(color.accent, 45) : filled ? wash(color.info, 14) : color.well,
                stroke: active ? color.accent : color.line,
                strokeWidth: active ? 1.5 : 0.5,
              }}
            />
            {cell >= 18 && label && (
              <text
                x={ox + c * cell + cell / 2}
                y={oy + r * cell + cell / 2 + 3.5}
                textAnchor="middle"
                style={{ fill: active ? color.fg : color.muted, fontSize: 10 }}
              >
                {label}
              </text>
            )}
          </g>,
        );
      }
    }
    return cells;
  };

  const positions = PANELS.map((_, i) =>
    stacked ? { x: Math.max(8, (width - side) / 2), y: i * (panelH + 10) } : { x: 8 + i * (side + gap), y: 0 },
  );

  return (
    <Figure
      title="One tile, start to finish"
      caption="Eight lanes handle row k of an 8 by 8 stand-in for the real tile. The global read and the global write each move a row, so both coalesce. The turn from rows to columns happens in the shared tile, where a column read is the access that can conflict."
      controls={
        <>
          <StepControls stepper={stepper} count={STEPS.length} describe={(s) => STEPS[s]} />
          <Slider label="Row k" value={k} min={0} max={N - 1} onChange={setK} />
        </>
      }
      readout={
        <>
          <Stat label="Step" value={`${step + 1} of ${STEPS.length}: ${STEPS[step]}`} />
          <Stat label="Row k" value={k} tone={color.accent} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label={`Step ${step + 1}: ${STEPS[step]}, row ${k}`}>
          {PANELS.map((p, i) => (
            <g key={p.title}>
              <text x={positions[i].x} y={positions[i].y + 12} style={{ fill: color.fg, fontSize: 12, fontWeight: 600 }}>
                {p.title}
              </text>
              <text x={positions[i].x} y={positions[i].y + 26} style={{ fill: color.muted, fontSize: 10 }}>
                {p.note}
              </text>
              {grid(i, positions[i].x, positions[i].y + labelH)}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
