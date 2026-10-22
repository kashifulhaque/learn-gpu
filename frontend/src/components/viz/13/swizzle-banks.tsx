import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// A 32 x 32 tile of 4-byte words in shared memory, row-major: the word at
// (row, col) has offset 32 * row + col. Bank = offset mod 32 (docs/FACTS.md
// section 3.2: 32 banks of 4 bytes, so bank = word address mod 32). A warp
// access of 32 words takes as many transactions as the largest number of
// distinct words that share one bank. Swizzle<B,M,S> is the CuTe functor: the
// B-bit field at bit M + max(0, S) is shifted by S and XORed into the B-bit field
// at bit M - min(0, S). The lab's solution.py implements the same function.

const N = 32;
const WORDS = N * N;

function swizzle(offset: number, b: number, m: number, s: number): number {
  const mask = (1 << b) - 1;
  const yyy = mask << (m + Math.max(0, s));
  const picked = offset & yyy;
  return offset ^ (s >= 0 ? picked >> s : picked << -s);
}

type Access = "column" | "row";

/** Physical offsets a warp touches, and the bank-conflict degree. */
function warp(access: Access, line: number, b: number, m: number, s: number) {
  const offsets: number[] = [];
  for (let t = 0; t < 32; t++) {
    const logical = access === "column" ? t * N + line : line * N + t;
    offsets.push(swizzle(logical, b, m, s));
  }
  const perBank: Set<number>[] = Array.from({ length: 32 }, () => new Set<number>());
  offsets.forEach((o) => perBank[o % 32].add(o));
  const counts = perBank.map((set) => set.size);
  return { offsets, counts, degree: Math.max(...counts) };
}

export default function SwizzleBanks() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [access, setAccess] = useState<Access>("column");
  const [line, setLine] = useState(5);
  const [b, setB] = useState(5);
  const [m, setM] = useState(0);
  const [s, setS] = useState(5);

  const plain = warp(access, line, 0, 0, 0);
  const sw = warp(access, line, b, m, s);

  // Is the swizzle a permutation of the tile? (It must be, or two words collide.)
  const image = new Set<number>();
  for (let o = 0; o < WORDS; o++) image.add(swizzle(o, b, m, s));
  const bijective = image.size === WORDS;
  const valid = Math.abs(s) >= b;

  const gridW = Math.min(width, 420);
  const cell = Math.floor((gridW - 28) / N);
  const gx = 28;
  const hot = new Set(sw.offsets);

  const barH = 70;
  const barW = Math.max(6, Math.floor((width - 28) / 32) - 2);
  const maxCount = Math.max(4, ...sw.counts);

  return (
    <Figure
      title="Swizzle and shared-memory banks"
      caption={`A warp reads one ${access} of a 32 x 32 tile of 4-byte words. Highlighted cells are where the words sit after the swizzle; the bars count distinct words per bank. A bar of height k means a k-way conflict. Swizzle<5,0,5> XORs the row into the column, so a column read lands in 32 different banks.`}
      controls={
        <>
          <Segmented
            label="Warp reads a"
            value={access}
            onChange={setAccess}
            options={[
              { value: "column", label: "column" },
              { value: "row", label: "row" },
            ]}
          />
          <Slider label={access === "column" ? "Column" : "Row"} min={0} max={31} value={line} onChange={setLine} />
          <Slider label="B (mask bits)" min={0} max={5} value={b} onChange={setB} />
          <Slider label="M (kept low bits)" min={0} max={3} value={m} onChange={setM} />
          <Slider label="S (shift)" min={1} max={7} value={s} onChange={setS} />
        </>
      }
      readout={
        <>
          <Stat label="Without swizzle" value={`${plain.degree}-way`} tone={plain.degree > 1 ? color.bad : color.ok} />
          <Stat label={`Swizzle<${b},${m},${s}>`} value={`${sw.degree}-way`} tone={sw.degree > 1 ? color.bad : color.ok} />
          <Stat label="Permutes the tile" value={bijective ? "yes" : "no"} tone={bijective ? color.ok : color.bad} />
          <Stat label="Valid (|S| ≥ B)" value={valid ? "yes" : "no"} tone={valid ? color.ok : color.warn} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, 200)}
          height={28 + N * cell + 30 + barH + 22}
          role="img"
          aria-label={`Shared-memory tile with the words one warp touches, and a bar chart of words per bank. Swizzle ${b},${m},${s} gives a ${sw.degree}-way conflict; no swizzle gives ${plain.degree}-way.`}
        >
          <text x={gx} y={12} fontSize={11} style={{ fill: color.muted }}>
            physical column (word within row) →
          </text>
          {Array.from({ length: N }, (_, r) =>
            Array.from({ length: N }, (_, c) => {
              const o = r * N + c;
              const on = hot.has(o);
              return (
                <rect
                  key={o}
                  x={gx + c * cell}
                  y={20 + r * cell}
                  width={cell - 1}
                  height={cell - 1}
                  rx={1.5}
                  style={{ fill: on ? color.accent : wash(color.faint, 25) }}
                />
              );
            }),
          )}
          <text x={4} y={20 + (N * cell) / 2} fontSize={11} style={{ fill: color.muted }} transform={`rotate(-90 10 ${20 + (N * cell) / 2})`} textAnchor="middle">
            row
          </text>
          <g transform={`translate(0, ${20 + N * cell + 22})`}>
            <text x={gx} y={-6} fontSize={11} style={{ fill: color.muted }}>
              distinct words per bank, banks 0 to 31
            </text>
            {sw.counts.map((count, bank) => {
              const bh = (count / maxCount) * barH;
              return (
                <rect
                  key={bank}
                  x={gx + bank * (barW + 2)}
                  y={barH - bh}
                  width={barW}
                  height={Math.max(bh, count ? 1 : 0)}
                  rx={1.5}
                  style={{ fill: count > 1 ? color.bad : color.ok }}
                />
              );
            })}
            <line x1={gx} x2={gx + 32 * (barW + 2)} y1={barH} y2={barH} style={{ stroke: color.lineStrong }} />
          </g>
        </svg>
      </div>
    </Figure>
  );
}
