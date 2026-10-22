// Bank conflicts for one ldmatrix 8x8 matrix: eight rows of 16 bytes each.
// Shared memory has 32 banks of 4 bytes, so 128 bytes per wavefront
// (docs/FACTS.md, section 2). The eight rows are served in one wavefront only
// if no bank is touched twice.
import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

const BANKS = 32;
const BANK_BYTES = 4;
const ROW_BYTES = 16;
const ROWS = 8;
const BYTES_PER_ELEMENT = 2; // bf16

const STRIDES = [
  { value: 32, label: "32 (64 B)" },
  { value: 40, label: "40 (80 B)" },
  { value: 64, label: "64 (128 B)" },
  { value: 72, label: "72 (144 B)" },
  { value: 136, label: "136 (272 B)" },
] as const;

export default function LdmatrixBanks() {
  const [stride, setStride] = useState<number>(64);
  const [ref, width] = useWidth<HTMLDivElement>();

  const strideBytes = stride * BYTES_PER_ELEMENT;
  // hits[bank] = how many of the eight rows touch that bank.
  const hits = new Array<number>(BANKS).fill(0);
  const rowBanks: number[][] = [];
  for (let r = 0; r < ROWS; r++) {
    const start = (r * strideBytes) % (BANKS * BANK_BYTES);
    const banks: number[] = [];
    for (let b = 0; b < ROW_BYTES / BANK_BYTES; b++) {
      const bank = (Math.floor(start / BANK_BYTES) + b) % BANKS;
      banks.push(bank);
      hits[bank] += 1;
    }
    rowBanks.push(banks);
  }
  const ways = Math.max(...hits);

  const left = 34;
  const cell = Math.max(8, Math.floor((width - left - 6) / BANKS));
  const h = 18 + ROWS * 16 + 22;
  const tones = [color.a, color.b, color.c, color.d];

  return (
    <Figure
      title="Eight ldmatrix rows against 32 banks"
      caption="Each row of the 8 x 8 matrix is 16 bytes, which covers 4 banks. A stride of 64 elements (128 bytes) puts every row on the same 4 banks: an 8-way conflict. A stride of 40 or 136 spreads the rows over all 32 banks."
      controls={<Segmented label="Row stride, elements (bytes)" value={stride} onChange={setStride} options={STRIDES} />}
      readout={
        <>
          <Stat label="Row stride" value={`${stride} elements = ${strideBytes} B`} />
          <Stat label="Worst bank is hit by" value={`${ways} row${ways > 1 ? "s" : ""}`} tone={ways === 1 ? color.ok : color.bad} />
          <Stat label="Wavefronts for the matrix" value={ways} tone={ways === 1 ? color.ok : color.bad} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={left + cell * BANKS + 6} height={h} role="img" aria-label="Bank usage of eight ldmatrix rows">
          {Array.from({ length: BANKS }, (_, b) => (
            <text key={b} x={left + b * cell + cell / 2} y={10} textAnchor="middle" style={{ fill: color.muted, fontSize: cell >= 14 ? 8 : 0 }}>
              {b}
            </text>
          ))}
          {rowBanks.map((banks, r) => (
            <g key={r}>
              <text x={2} y={18 + r * 16 + 11} style={{ fill: color.muted, fontSize: 10 }}>
                row {r}
              </text>
              {Array.from({ length: BANKS }, (_, b) => (
                <rect
                  key={b}
                  x={left + b * cell}
                  y={18 + r * 16}
                  width={cell - 1}
                  height={14}
                  rx={2}
                  style={{ fill: banks.includes(b) ? wash(tones[r % 4], hits[b] > 1 ? 85 : 55) : color.well, stroke: banks.includes(b) && hits[b] > 1 ? color.bad : color.line }}
                />
              ))}
            </g>
          ))}
          <text x={left} y={18 + ROWS * 16 + 14} style={{ fill: color.muted, fontSize: 10 }}>
            bank 0 to 31, left to right. Red outline: a bank two or more rows touch.
          </text>
        </svg>
      </div>
    </Figure>
  );
}
