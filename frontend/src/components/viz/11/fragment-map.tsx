// Which lane of the warp holds which element of an mma.sync.m16n8k16 tile.
// The formulas are the PTX ISA's fragment layouts for .f16 and .bf16 inputs,
// with g = lane / 4 (groupID) and t = lane % 4 (threadID_in_group).
import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

type Operand = "A" | "B" | "C";
type Cell = { lane: number; reg: string };

const SHAPE: Record<Operand, { rows: number; cols: number; rowLabel: string; colLabel: string }> = {
  A: { rows: 16, cols: 16, rowLabel: "m", colLabel: "k" },
  B: { rows: 16, cols: 8, rowLabel: "k", colLabel: "n" },
  C: { rows: 16, cols: 8, rowLabel: "m", colLabel: "n" },
};

// Build the owner of every element: owners[row][col] lists the lane and register.
function owners(op: Operand): Cell[][] {
  const { rows, cols } = SHAPE[op];
  const grid: Cell[][] = Array.from({ length: rows }, () => Array.from({ length: cols }, () => ({ lane: -1, reg: "" })));
  for (let lane = 0; lane < 32; lane++) {
    const g = lane >> 2;
    const t = lane & 3;
    if (op === "A") {
      for (let i = 0; i < 8; i++) {
        const r = i < 2 || (i >= 4 && i < 6) ? g : g + 8;
        const c = 2 * t + (i & 1) + (i >= 4 ? 8 : 0);
        grid[r][c] = { lane, reg: `a${i}` };
      }
    } else if (op === "B") {
      for (let i = 0; i < 4; i++) {
        const k = 2 * t + (i & 1) + (i >= 2 ? 8 : 0);
        grid[k][g] = { lane, reg: `b${i}` };
      }
    } else {
      for (let i = 0; i < 4; i++) {
        const r = i < 2 ? g : g + 8;
        const c = 2 * t + (i & 1);
        grid[r][c] = { lane, reg: `c${i}` };
      }
    }
  }
  return grid;
}

export default function FragmentMap() {
  const [op, setOp] = useState<Operand>("A");
  const [lane, setLane] = useState(5);
  const [ref, width] = useWidth<HTMLDivElement>();
  const { rows, cols, rowLabel, colLabel } = SHAPE[op];
  const grid = owners(op);

  const left = 28;
  const top = 18;
  const cell = Math.max(14, Math.min(30, Math.floor((width - left - 8) / cols)));
  const w = left + cell * cols + 8;
  const h = top + cell * rows + 4;
  const g = lane >> 2;
  const t = lane & 3;
  const mine: string[] = [];
  grid.forEach((row, r) =>
    row.forEach((c, k) => {
      if (c.lane === lane) mine.push(`${c.reg} = [${r}, ${k}]`);
    }),
  );

  return (
    <Figure
      title="Who holds what in an m16n8k16 tile"
      caption="Each cell shows the lane that holds it. The highlighted cells are the selected lane's, labelled with the register element that holds each one. Two neighbouring elements of a row share one 32-bit register."
      controls={
        <>
          <Segmented
            label="Operand"
            value={op}
            onChange={setOp}
            options={[
              { value: "A", label: "A (16 x 16)" },
              { value: "B", label: "B (16 x 8)" },
              { value: "C", label: "C, D (16 x 8)" },
            ]}
          />
          <Slider label="Lane" value={lane} min={0} max={31} onChange={setLane} />
        </>
      }
      readout={
        <>
          <Stat label="Lane" value={lane} tone={color.accent} />
          <Stat label="groupID g = lane / 4" value={g} tone={color.a} />
          <Stat label="t = lane % 4" value={t} tone={color.b} />
          <Stat label="Elements [row, col]" value={mine.join("   ")} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={w} height={h} role="img" aria-label={`Fragment layout of operand ${op}`}>
          <text x={left} y={11} style={{ fill: color.muted, fontSize: 10 }}>
            {colLabel} →
          </text>
          <text x={2} y={top + 10} style={{ fill: color.muted, fontSize: 10 }}>
            {rowLabel} ↓
          </text>
          {grid.map((row, r) =>
            row.map((c, k) => {
              const mineCell = c.lane === lane;
              const sameGroup = !mineCell && c.lane >> 2 === g;
              return (
                <g key={`${r}-${k}`}>
                  <rect
                    x={left + k * cell}
                    y={top + r * cell}
                    width={cell - 1}
                    height={cell - 1}
                    rx={2}
                    style={{
                      fill: mineCell ? color.accent : sameGroup ? wash(color.a, 22) : color.well,
                      stroke: mineCell ? color.accent : color.line,
                    }}
                  />
                  <text
                    x={left + k * cell + (cell - 1) / 2}
                    y={top + r * cell + (cell - 1) / 2 + 3}
                    textAnchor="middle"
                    style={{ fill: mineCell ? color.paper : color.fg, fontSize: cell >= 22 ? 10 : 8, fontWeight: mineCell ? 700 : 400 }}
                  >
                    {c.lane}
                  </text>
                </g>
              );
            }),
          )}
        </svg>
      </div>
    </Figure>
  );
}
