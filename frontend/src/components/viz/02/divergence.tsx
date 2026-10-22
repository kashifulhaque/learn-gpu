import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// One warp is 32 threads (docs/FACTS.md section 4.1). The drawing follows the
// lab kernel: `ways` branches that each cost the same, chosen either by lane
// (threadIdx.x % 32 % ways) or by warp (warp number % ways).
const WARP = 32;
const WAYS = [1, 2, 4, 8, 16, 32] as const;
const TONES = [color.a, color.b, color.c, color.d, color.violet, color.info, color.ok, color.warn];

type Mode = "lane" | "warp";

export default function Divergence() {
  const [ways, setWays] = useState<number>(4);
  const [mode, setMode] = useState<Mode>("lane");
  const [ref, width] = useWidth<HTMLDivElement>(560);

  // Lane mode: one warp runs `ways` passes, and in pass p only lanes with
  // lane % ways == p are active. Warp mode: each of the shown warps takes one
  // branch and runs one pass.
  const passes = mode === "lane" ? ways : 1;
  const activeLanes = mode === "lane" ? Math.ceil(WARP / ways) : WARP;
  const efficiency = mode === "lane" ? 1 / ways : 1;

  const pad = 10;
  const labelW = 70;
  const cell = Math.max(6, Math.min(18, Math.floor((width - labelW - 2 * pad) / WARP) - 2));
  const rowH = cell + 4;
  const shownRows = mode === "lane" ? Math.min(passes, 32) : 4;
  const height = pad * 2 + 22 + shownRows * rowH;

  const rows = [];
  for (let row = 0; row < shownRows; row++) {
    const y = pad + 22 + row * rowH;
    // In lane mode a row is one pass of one warp. In warp mode a row is one
    // warp that takes branch (row % ways) in its single pass.
    const label = mode === "lane" ? `pass ${row + 1}` : `warp ${row}`;
    const branch = mode === "lane" ? row : row % ways;
    rows.push(
      <g key={row}>
        <text x={pad} y={y + cell - 2} fontSize={11} style={{ fill: color.muted }}>
          {label}
        </text>
        {Array.from({ length: WARP }, (_, lane) => {
          const active = mode === "lane" ? lane % ways === row : true;
          return (
            <rect
              key={lane}
              x={pad + labelW + lane * (cell + 2)}
              y={y}
              width={cell}
              height={cell}
              rx={2}
              style={{ fill: active ? wash(TONES[branch % TONES.length], 80) : color.well, stroke: color.line }}
            />
          );
        })}
      </g>,
    );
  }

  return (
    <Figure
      title="One warp, several branches"
      caption={
        mode === "lane"
          ? "Lane mode: the warp issues every branch in turn, and the lanes that did not take it sit masked off (grey). The prediction is a slowdown equal to the number of branches."
          : "Warp mode: all 32 lanes of a warp agree, so each warp issues one branch and the number of branches costs nothing extra. Four warps are drawn."
      }
      controls={
        <>
          <Slider label="Branches (ways)" value={ways} values={WAYS} onChange={setWays} />
          <Segmented
            label="Branch chosen by"
            value={mode}
            options={[
              { value: "lane", label: "Lane within the warp" },
              { value: "warp", label: "Warp number" },
            ]}
            onChange={setMode}
          />
        </>
      }
      readout={
        <>
          <Stat label="Passes per warp" value={passes} tone={passes > 1 ? color.warn : color.ok} />
          <Stat label="Active lanes per pass" value={`${activeLanes} of ${WARP}`} />
          <Stat label="Lane efficiency" value={`${(efficiency * 100).toFixed(1)}%`} />
          <Stat label="Predicted slowdown" value={`${passes}x`} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Rows of 32 lane cells, one row per pass of the warp">
          <text x={pad + labelW} y={pad + 12} fontSize={11} style={{ fill: color.subtle }}>
            lane 0 to 31, left to right
          </text>
          {rows}
        </svg>
      </div>
    </Figure>
  );
}
