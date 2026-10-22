import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants. The 32-byte sector and the 128-byte line are from the CUDA C++
// Best Practices Guide and the CUDA 11.8 Programming Guide (docs/FACTS.md,
// section 3.1). A warp has 32 lanes, and each lane loads one float32 (4 B).
const SECTOR = 32;
const LINE = 128;
const WARP = 32;
const FLOAT = 4;
const SECTORS_PER_ROW = 32;
// The lab's measured strided slowdown, from learn-inference chapter 12.
const STRIDES = [1, 2, 4, 8, 16, 32] as const;

type Width = 4 | 16;

export default function Coalescing() {
  const [stride, setStride] = useState<number>(32);
  const [offset, setOffset] = useState<number>(0);
  const [bytes, setBytes] = useState<Width>(4);
  const [ref, width] = useWidth<HTMLDivElement>(560);

  // Lane t loads `bytes` starting at (offset + t * stride) elements of `bytes`.
  const lanes = Array.from({ length: WARP }, (_, t) => (offset * FLOAT + t * stride * bytes));
  const sectorSet = new Map<number, number>();
  const lineSet = new Set<number>();
  for (const start of lanes) {
    for (let b = start; b < start + bytes; b += 1) {
      const s = Math.floor(b / SECTOR);
      sectorSet.set(s, (sectorSet.get(s) ?? 0) + (b === start ? 1 : 0));
      lineSet.add(Math.floor(b / LINE));
    }
  }
  const useful = WARP * bytes;
  const fetched = sectorSet.size * SECTOR;
  const efficiency = useful / fetched;
  const totalSectors = Math.max(SECTORS_PER_ROW, Math.max(...sectorSet.keys()) + 1);
  const rows = Math.ceil(totalSectors / SECTORS_PER_ROW);

  // A float4 access must start on a 16-byte boundary, which is every 4 floats.
  const misaligned = bytes === 16 && offset % 4 !== 0;

  const pad = 8;
  const cell = Math.max(8, Math.floor((width - 2 * pad) / SECTORS_PER_ROW));
  const rowH = cell + 6;
  const height = pad * 2 + rows * rowH;

  return (
    <Figure
      title="One warp load, sector by sector"
      caption="Each square is one 32-byte sector, and a thick bar marks every 128-byte cache line. Filled squares are the sectors a warp's 32 loads touch; the number on a square is how many lanes read it."
      controls={
        <>
          <Slider
            label="Stride between lanes (elements)"
            value={stride}
            values={STRIDES}
            onChange={setStride}
          />
          <Slider label="Start offset (elements)" value={offset} min={0} max={7} onChange={setOffset} />
          <Segmented
            label="Bytes per lane"
            value={bytes}
            options={[
              { value: 4, label: "4 (float)" },
              { value: 16, label: "16 (float4)" },
            ]}
            onChange={setBytes}
          />
        </>
      }
      readout={
        <>
          <Stat label="Sectors fetched" value={sectorSet.size} tone={color.accent} />
          <Stat label="Bytes fetched" value={fetched} />
          <Stat label="Useful bytes" value={useful} tone={color.ok} />
          <Stat
            label="Efficiency"
            value={`${(efficiency * 100).toFixed(1)}%`}
            tone={efficiency > 0.99 ? color.ok : efficiency < 0.5 ? color.bad : color.warn}
          />
          <Stat label="Cache lines" value={lineSet.size} />
          {misaligned && <Stat label="float4 at this offset" value="misaligned: a fault" tone={color.bad} />}
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Grid of 32-byte sectors with the touched ones filled">
          {Array.from({ length: totalSectors }, (_, s) => {
            const row = Math.floor(s / SECTORS_PER_ROW);
            const col = s % SECTORS_PER_ROW;
            const x = pad + col * cell;
            const y = pad + row * rowH;
            const lanesHere = sectorSet.get(s);
            const touched = lanesHere !== undefined;
            return (
              <g key={s}>
                <rect
                  x={x + 1}
                  y={y}
                  width={cell - 2}
                  height={cell}
                  rx={2}
                  style={{
                    fill: touched ? wash(color.accent, 70) : color.well,
                    stroke: touched ? color.accent : color.line,
                  }}
                />
                {touched && cell >= 14 && lanesHere > 0 && (
                  <text
                    x={x + cell / 2}
                    y={y + cell / 2 + 4}
                    textAnchor="middle"
                    fontSize={10}
                    style={{ fill: color.fg }}
                  >
                    {lanesHere}
                  </text>
                )}
                {col % 4 === 0 && (
                  <rect x={x} y={y - 2} width={2} height={cell + 4} style={{ fill: color.lineStrong }} />
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "sector touched", tone: color.accent },
          { label: "sector not fetched", tone: color.line },
        ]}
      />
    </Figure>
  );
}
