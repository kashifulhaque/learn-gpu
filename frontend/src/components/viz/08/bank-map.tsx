import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// The model is the chapter's: 32 banks, each 4 bytes wide. The bank of a byte
// address is (address / 4) mod 32 (docs/FACTS.md, section 3.2). A tile is 32
// elements wide for 4-byte elements and 64 for 2-byte elements, so one row is
// 128 bytes. The three layouts match the lab's smem_index.
type Layout = "plain" | "padded" | "swizzled";

const LANES = 32;
const BANKS = 32;

function tileDim(elemBytes: number) {
  return elemBytes === 4 ? 32 : 64;
}

function index(layout: Layout, r: number, c: number, elemBytes: number) {
  const t = tileDim(elemBytes);
  const w = 4 / elemBytes;
  if (layout === "plain") return r * t + c;
  if (layout === "padded") return r * (t + w) + c;
  return r * t + (c ^ ((r & 31) * w));
}

function tileBytes(layout: Layout, elemBytes: number) {
  const t = tileDim(elemBytes);
  const w = 4 / elemBytes;
  return t * (layout === "padded" ? t + w : t) * elemBytes;
}

type Entry = { lanes: number[] };

/** For each bank, the distinct 4-byte words that the warp's 32 lanes touch. */
function wordsPerBank(layout: Layout, elemBytes: number, column: number) {
  const banks: Map<number, Entry>[] = Array.from({ length: BANKS }, () => new Map());
  for (let lane = 0; lane < LANES; lane++) {
    const byte = index(layout, lane, column, elemBytes) * elemBytes;
    const word = Math.floor(byte / 4);
    const bank = word % BANKS;
    const entry = banks[bank].get(word) ?? { lanes: [] };
    entry.lanes.push(lane);
    banks[bank].set(word, entry);
  }
  return banks;
}

function degreeOf(layout: Layout, elemBytes: number, column: number) {
  return Math.max(...wordsPerBank(layout, elemBytes, column).map((m) => m.size));
}

export default function BankMap() {
  const [layout, setLayout] = useState<Layout>("plain");
  const [elemBytes, setElemBytes] = useState<number>(4);
  const [column, setColumn] = useState(0);
  const [ref, width] = useWidth<HTMLDivElement>();

  const t = tileDim(elemBytes);
  const col = Math.min(column, t - 1);
  const banks = wordsPerBank(layout, elemBytes, col);
  const degree = Math.max(...banks.map((m) => m.size));
  let worst = 1;
  for (let c = 0; c < t; c++) worst = Math.max(worst, degreeOf(layout, elemBytes, c));

  const left = 34;
  const right = 8;
  const cell = (width - left - right) / BANKS;
  const cellH = 11;
  const stackRows = Math.max(4, degree);
  const top = 22;
  const height = top + stackRows * (cellH + 1) + 26;
  const showEvery = cell < 16 ? 4 : 1;

  return (
    <Figure
      title="Which banks a column read hits"
      caption="Lane l of a warp reads tile[l][c], one column of the tile. Each square is one 32-bit word, stacked in its bank. A stack taller than one is a bank conflict: the bank serves its words one clock at a time."
      controls={
        <>
          <Segmented
            label="Layout"
            value={layout}
            onChange={setLayout}
            options={[
              { value: "plain", label: "Plain" },
              { value: "padded", label: "Padded" },
              { value: "swizzled", label: "Swizzled" },
            ]}
          />
          <Segmented
            label="Element"
            value={elemBytes}
            onChange={setElemBytes}
            options={[
              { value: 4, label: "float32" },
              { value: 2, label: "bfloat16" },
            ]}
          />
          <Slider label="Column c" value={col} min={0} max={t - 1} onChange={setColumn} />
        </>
      }
      readout={
        <>
          <Stat label="Layout" value={`${layout}, ${elemBytes === 4 ? "float32" : "bfloat16"}, column ${col}`} />
          <Stat label="Conflict degree" value={`${degree}-way`} tone={degree > 1 ? color.bad : color.ok} />
          <Stat label="Worst column" value={`${worst}-way`} tone={worst > 1 ? color.bad : color.ok} />
          <Stat label="Bandwidth left" value={`${Math.round(100 / degree)}%`} />
          <Stat label="Tile in smem" value={`${tileBytes(layout, elemBytes)} B`} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Column ${col} read from a ${layout} ${elemBytes}-byte tile: ${degree}-way conflict`}
        >
          <text x={left - 6} y={top - 8} textAnchor="end" style={{ fill: color.muted, fontSize: 10 }}>
            bank
          </text>
          {banks.map((words, b) => {
            const x = left + b * cell;
            const entries = Array.from(words.values());
            const tone = entries.length > 1 ? color.bad : color.ok;
            return (
              <g key={b}>
                {b % showEvery === 0 && (
                  <text x={x + cell / 2} y={top - 8} textAnchor="middle" style={{ fill: color.muted, fontSize: 10 }}>
                    {b}
                  </text>
                )}
                <rect
                  x={x + 0.5}
                  y={top}
                  width={Math.max(cell - 1, 1)}
                  height={stackRows * (cellH + 1)}
                  style={{ fill: color.well, stroke: color.line, strokeWidth: 0.5 }}
                />
                {entries.map((entry, k) => (
                  <rect
                    key={k}
                    x={x + 1.5}
                    y={top + k * (cellH + 1) + 1}
                    width={Math.max(cell - 3, 1)}
                    height={cellH}
                    rx={2}
                    style={{ fill: wash(tone, 55), stroke: tone, strokeWidth: 1 }}
                  >
                    <title>{`bank ${b}: lanes ${entry.lanes.join(", ")}`}</title>
                  </rect>
                ))}
              </g>
            );
          })}
          <text x={left} y={height - 8} style={{ fill: color.muted, fontSize: 11 }}>
            {degree === 1
              ? "32 lanes, 32 banks: one transaction"
              : `${degree} words queue in one bank: ${degree} transactions`}
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "bank serves one word", tone: color.ok },
          { label: "bank serves several words", tone: color.bad },
        ]}
      />
    </Figure>
  );
}
