import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Accumulator thread-value layout of mma.sync.m16n8k16, from chapter 13
// and the PTX fragment table. Column-major 16 by 8 tile, offset = m + 16 * n.
//   m = floor(lane / 4) + 8 * floor(value / 2)
//   n = 2 * (lane mod 4) + (value mod 2)
// scripts/figures/24-cute-dsl-gemm.py prints the same grid.

const ROWS = 16;
const COLS = 8;

function cellOf(lane: number, value: number): { m: number; n: number } {
  return {
    m: Math.floor(lane / 4) + 8 * Math.floor(value / 2),
    n: 2 * (lane % 4) + (value % 2),
  };
}

const OWNERS: { lane: number; value: number }[][] = Array.from({ length: ROWS }, () =>
  Array.from({ length: COLS }, () => ({ lane: -1, value: -1 })),
);
for (let lane = 0; lane < 32; lane += 1) {
  for (let value = 0; value < 4; value += 1) {
    const { m, n } = cellOf(lane, value);
    OWNERS[m][n] = { lane, value };
  }
}

export default function FragmentOwners() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [lane, setLane] = useState(1);
  const cell = Math.max(22, Math.min(36, Math.floor((width - 56) / COLS)));
  const gridW = COLS * cell;
  const gridH = ROWS * cell;
  const ox = Math.max(36, Math.floor((width - gridW) / 2));
  const oy = 28;
  const owned = [0, 1, 2, 3].map((value) => cellOf(lane, value));

  return (
    <Figure
      title="Who owns the accumulator"
      caption="Each cell is one fp32 accumulator in the 16 by 8 tile of mma.sync.m16n8k16. The number is the lane. Lane 1 owns (0, 2), (0, 3), (8, 2), and (8, 3)."
      controls={<Slider label="Lane" min={0} max={31} value={lane} onChange={setLane} />}
      readout={
        <>
          <Stat label="Lane" value={lane} tone={color.a} />
          <Stat
            label="Cells (m, n)"
            value={owned.map(({ m, n }) => `(${m}, ${n})`).join("  ")}
            tone={color.b}
          />
          <Stat label="Values" value="4" />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={oy + gridH + 16} role="img" aria-label="16 by 8 accumulator tile colored by lane">
          {OWNERS.map((row, m) =>
            row.map((owner, n) => {
              const hot = owner.lane === lane;
              return (
                <g key={`${m}-${n}`}>
                  <rect
                    x={ox + n * cell}
                    y={oy + m * cell}
                    width={cell - 1}
                    height={cell - 1}
                    rx={2}
                    style={{
                      fill: hot ? wash(color.a, 55) : wash(color.b, 18),
                      stroke: hot ? color.a : color.line,
                    }}
                  />
                  <text
                    x={ox + n * cell + (cell - 1) / 2}
                    y={oy + m * cell + (cell - 1) / 2 + 4}
                    textAnchor="middle"
                    fontSize={cell > 28 ? 11 : 9}
                    style={{ fill: hot ? color.a : color.fg }}
                  >
                    {owner.lane}
                  </text>
                </g>
              );
            }),
          )}
          <text x={ox} y={16} fontSize={12} style={{ fill: color.muted }}>
            n →
          </text>
          <text x={8} y={oy + 12} fontSize={12} style={{ fill: color.muted }}>
            m
          </text>
        </svg>
      </div>
    </Figure>
  );
}
