import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";
import { DATA } from "./data";

// Which (warp, lane, register) holds each element of a 16 by 16 tile, for four
// Gluon-style blocked layouts. The numbers come from scripts/figures/38-tile-languages.py,
// which calls blocked_owner from the lab's solution. The register numbering is the
// lab's convention. Each cell shows its lane. The tint is its warp.

const WARP_TONES = [color.a, color.b, color.c, color.d];

export default function BlockedLayout() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [preset, setPreset] = useState(0);
  const [row, setRow] = useState(1);
  const [col, setCol] = useState(5);
  const lay = DATA.layouts[preset];
  const [warp, lane, reg] = lay.owner[row][col];

  const cell = Math.max(18, Math.min(30, Math.floor((width - 44) / 16)));
  const gx = 30;
  const gy = 22;

  let mine = 0;
  for (const r of lay.owner) for (const o of r) if (o[0] === warp && o[1] === lane) mine++;

  return (
    <Figure
      title="Which thread holds which element"
      caption={`Each cell shows the lane that holds the element, and its tint is the warp. The outlined cells are the ${mine} elements held by the thread that holds your pick. Layout: size_per_thread ${JSON.stringify(lay.spt)}, threads_per_warp ${JSON.stringify(lay.tpw)}, warps_per_cta ${JSON.stringify(lay.wpc)}, order ${JSON.stringify(lay.order)}.`}
      controls={
        <>
          <Segmented
            label="Layout"
            value={preset}
            options={DATA.layouts.map((_, i) => ({ value: i, label: `L${i + 1}` }))}
            onChange={setPreset}
          />
          <Slider label="Row" min={0} max={15} value={row} onChange={setRow} />
          <Slider label="Column" min={0} max={15} value={col} onChange={setCol} />
        </>
      }
      readout={
        <>
          <Stat label="Warp" value={String(warp)} tone={WARP_TONES[warp % 4]} />
          <Stat label="Lane" value={String(lane)} />
          <Stat label="Register" value={String(reg)} />
          <Stat label="Registers per thread" value={String(lay.regs_per_thread)} />
        </>
      }
    >
      <div ref={ref}>
        <div style={{ fontSize: 12, color: color.muted, marginBottom: 4 }}>{lay.name}</div>
        <svg
          width={Math.max(width, gx + 16 * cell + 8)}
          height={gy + 16 * cell + 6}
          role="img"
          aria-label={`Layout ${preset + 1}, ${lay.name}. Element (${row}, ${col}) is held by warp ${warp}, lane ${lane}, register ${reg}.`}
        >
          {Array.from({ length: 16 }, (_, c) => (
            <text
              key={`c${c}`}
              x={gx + c * cell + cell / 2}
              y={14}
              fontSize={10}
              textAnchor="middle"
              style={{ fill: color.muted }}
            >
              {c}
            </text>
          ))}
          {lay.owner.map((rowData, r) => (
            <g key={r}>
              <text x={gx - 6} y={gy + r * cell + cell / 2 + 3} fontSize={10} textAnchor="end" style={{ fill: color.muted }}>
                {r}
              </text>
              {rowData.map(([w, l], c) => {
                const same = w === warp && l === lane;
                const picked = r === row && c === col;
                return (
                  <g key={c}>
                    <rect
                      x={gx + c * cell}
                      y={gy + r * cell}
                      width={cell - 1}
                      height={cell - 1}
                      rx={2}
                      style={{
                        fill: picked ? WARP_TONES[w % 4] : wash(WARP_TONES[w % 4], 38),
                        stroke: same ? color.fg : "none",
                        strokeWidth: same ? 1.5 : 0,
                      }}
                    />
                    <text
                      x={gx + c * cell + (cell - 1) / 2}
                      y={gy + r * cell + (cell - 1) / 2 + 3}
                      fontSize={10}
                      textAnchor="middle"
                      style={{ fill: picked ? color.paper : color.fg }}
                    >
                      {l}
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
