import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Tensor memory is 128 lanes by 512 columns of 32-bit cells. The figure draws
// 4 lane groups (the lanes one warp of a warpgroup reaches) by 16 blocks of
// 32 columns. The rows below come from scripts/figures/35-blackwell-tcgen05.py,
// which calls the lab's accumulator_footprint and accumulator_stages:
// [N, allocated columns, stages separate, stages carved, idle separate, idle carved]
const ROWS: readonly (readonly [number, number, number, number, number, number])[] = [
  [64, 64, 8, 8, 0, 0],
  [96, 128, 4, 5, 128, 32],
  [128, 128, 4, 4, 0, 0],
  [160, 256, 2, 3, 192, 32],
  [192, 256, 2, 2, 128, 128],
  [256, 256, 2, 2, 0, 0],
];

const GROUPS = 4;
const BLOCKS = 16; // 32 columns each
const TONES = [color.a, color.b, color.c, color.d];

export default function TmemMap() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [n, setN] = useState(160);
  const [mode, setMode] = useState<"separate" | "carved">("separate");

  const row = ROWS.find((r) => r[0] === n) ?? ROWS[3];
  const [, alloc, stagesSep, stagesCarved, idleSep, idleCarved] = row;
  const carved = mode === "carved";
  const stages = carved ? stagesCarved : stagesSep;
  const idle = carved ? idleCarved : idleSep;
  const step = carved ? n : alloc; // columns each stage owns
  const colWidth = Math.max(8, Math.min(34, Math.floor((width - 70) / BLOCKS)));
  const gx = 48;
  const gy = 30;
  const rowH = 26;
  const pxPerColumn = colWidth / 32;

  return (
    <Figure
      title="Tensor memory: 128 lanes by 512 columns"
      caption="Each row is one warp's 32 lanes, and the whole stack is 128 lanes. A column spans all four rows. Every accumulator of width N takes N columns in all of its lanes. With one tcgen05.alloc per accumulator, the allocation rounds up to a power of two, so 160 columns take 256. One 512-column allocation, carved by offset, lets a 160-column accumulator fit three times."
      controls={
        <>
          <Segmented
            label="Accumulator width N"
            value={n}
            options={ROWS.map((r) => ({ value: r[0], label: String(r[0]) }))}
            onChange={setN}
          />
          <Segmented
            label="Allocation"
            value={mode}
            options={[
              { value: "separate", label: "One alloc each" },
              { value: "carved", label: "Carve 512" },
            ]}
            onChange={setMode}
          />
        </>
      }
      readout={
        <>
          <Stat label="Columns allocated per stage" value={String(step)} tone={color.a} />
          <Stat label="Stages that fit" value={String(stages)} tone={color.b} />
          <Stat label="Columns holding no accumulator" value={String(idle)} tone={color.d} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, gx + BLOCKS * colWidth + 8)}
          height={gy + GROUPS * rowH + 40}
          role="img"
          aria-label={`Tensor memory with ${stages} accumulators of ${n} columns, ${step} columns each, leaving ${idle} columns without an accumulator.`}
        >
          <text x={gx} y={14} fontSize={11} style={{ fill: color.muted }}>
            column 0
          </text>
          <text x={gx + BLOCKS * colWidth} y={14} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            column 511
          </text>
          {Array.from({ length: GROUPS }, (_, g) => (
            <g key={g}>
              <text x={gx - 6} y={gy + g * rowH + 17} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {`${g * 32}-${g * 32 + 31}`}
              </text>
              <rect
                x={gx}
                y={gy + g * rowH}
                width={BLOCKS * colWidth}
                height={rowH - 3}
                rx={3}
                style={{ fill: wash(color.faint, 25) }}
              />
              {Array.from({ length: stages }, (_, s) => (
                <rect
                  key={s}
                  x={gx + s * step * pxPerColumn}
                  y={gy + g * rowH}
                  width={n * pxPerColumn - 1}
                  height={rowH - 3}
                  rx={3}
                  style={{ fill: TONES[s % TONES.length] }}
                />
              ))}
              {!carved &&
                Array.from({ length: stages }, (_, s) =>
                  alloc > n ? (
                    <rect
                      key={`pad${s}`}
                      x={gx + (s * step + n) * pxPerColumn}
                      y={gy + g * rowH}
                      width={(alloc - n) * pxPerColumn - 1}
                      height={rowH - 3}
                      rx={3}
                      style={{ fill: "none", stroke: color.bad, strokeDasharray: "3 2" }}
                    />
                  ) : null,
                )}
            </g>
          ))}
          <text x={gx} y={gy + GROUPS * rowH + 16} fontSize={11} style={{ fill: color.muted }}>
            lanes on the left. Dashed: columns an allocation reserves beyond N.
          </text>
        </svg>
      </div>
    </Figure>
  );
}
