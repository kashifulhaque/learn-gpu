import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// 128-byte TMA swizzle, 16-byte atomicity, from the PTX ISA swizzle table
// (section 5.5.7). Eight rows of eight 16-byte chunks. The physical chunk
// is the logical chunk XOR the row. scripts/figures/26-hopper-gemm.py prints
// the same map, including row 1 chunk 1 at byte 128.

const ROWS = 8;
const CHUNKS = 8;

function physicalChunk(row: number, chunk: number): number {
  return chunk ^ (row & 7);
}

function byteOffset(row: number, chunk: number): number {
  return row * 128 + physicalChunk(row, chunk) * 16;
}

export default function SwizzleMap() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [row, setRow] = useState(1);
  const [chunk, setChunk] = useState(1);

  const cell = Math.max(28, Math.min(48, Math.floor((width - 56) / CHUNKS)));
  const gx = 36;
  const gy = 28;
  const selected = byteOffset(row, chunk);
  const plain = row * 128 + chunk * 16;

  return (
    <Figure
      title="128-byte swizzle of an 8 by 8 chunk tile"
      caption="Each cell is one 16-byte chunk. The number is the physical chunk, logical chunk XOR row. Row 1, chunk 1 moves from byte 144 to byte 128. The pattern is the PTX 128-byte mode with 16-byte atomicity, and it repeats every eight rows."
      controls={
        <>
          <Slider label="Row" min={0} max={7} value={row} onChange={setRow} />
          <Slider label="Logical chunk" min={0} max={7} value={chunk} onChange={setChunk} />
        </>
      }
      readout={
        <>
          <Stat label="Physical chunk" value={String(physicalChunk(row, chunk))} />
          <Stat label="Swizzled byte" value={String(selected)} tone={color.a} />
          <Stat label="Unswizzled byte" value={String(plain)} tone={color.b} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, gx + CHUNKS * cell + 8)}
          height={gy + ROWS * cell + 28}
          role="img"
          aria-label={`Eight by eight swizzle map. Row ${row}, logical chunk ${chunk}, lands on physical chunk ${physicalChunk(row, chunk)} at byte ${selected}.`}
        >
          <text x={gx} y={14} fontSize={11} style={{ fill: color.muted }}>
            logical chunk
          </text>
          {Array.from({ length: CHUNKS }, (_, c) => (
            <text
              key={`h${c}`}
              x={gx + c * cell + cell / 2}
              y={24}
              fontSize={11}
              textAnchor="middle"
              style={{ fill: color.muted }}
            >
              {c}
            </text>
          ))}
          {Array.from({ length: ROWS }, (_, r) =>
            Array.from({ length: CHUNKS }, (_, c) => {
              const on = r === row && c === chunk;
              const hotRow = r === row;
              return (
                <g key={`${r}-${c}`}>
                  <rect
                    x={gx + c * cell}
                    y={gy + r * cell}
                    width={cell - 2}
                    height={cell - 2}
                    rx={3}
                    style={{
                      fill: on ? color.a : hotRow ? wash(color.a, 28) : wash(color.faint, 35),
                    }}
                  />
                  <text
                    x={gx + c * cell + (cell - 2) / 2}
                    y={gy + r * cell + (cell - 2) / 2 + 4}
                    fontSize={12}
                    textAnchor="middle"
                    style={{ fill: on ? color.paper : color.fg }}
                  >
                    {physicalChunk(r, c)}
                  </text>
                </g>
              );
            }),
          )}
          {Array.from({ length: ROWS }, (_, r) => (
            <text
              key={`r${r}`}
              x={gx - 8}
              y={gy + r * cell + cell / 2 + 4}
              fontSize={11}
              textAnchor="end"
              style={{ fill: color.muted }}
            >
              {r}
            </text>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
