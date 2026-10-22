import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";
import { expertLoads } from "./loads";

// Constants. The preset row counts are illustrative splits of 200 rows over 5
// experts (the lab's own example [0, 5, 64, 1, 130] is the "ragged" preset).
// The launch bound is ceil(rows / BLOCK_M) + experts, the formula the lab's
// max_m_tiles uses.
const PRESETS = {
  ragged: [0, 5, 64, 1, 130],
  balanced: [40, 40, 40, 40, 40],
  skewed: expertLoads(200, 5, 1.5),
} as const;
type Preset = keyof typeof PRESETS;

type Tile = { expert: number; firstRow: number; rows: number };

function schedule(counts: readonly number[], blockM: number): Tile[] {
  const tiles: Tile[] = [];
  let start = 0;
  counts.forEach((count, expert) => {
    for (let first = 0; first < count; first += blockM) {
      tiles.push({ expert, firstRow: start + first, rows: Math.min(blockM, count - first) });
    }
    start += count;
  });
  return tiles;
}

const TONES = [color.a, color.b, color.c, color.d, color.violet, color.ok];

export default function TileScheduler() {
  const [preset, setPreset] = useState<Preset>("ragged");
  const [blockM, setBlockM] = useState(64);
  const [pid, setPid] = useState(0);
  const [ref, width] = useWidth<HTMLDivElement>();

  const counts = PRESETS[preset];
  const rows = counts.reduce((a, b) => a + b, 0);
  const tiles = schedule(counts, blockM);
  const launched = Math.ceil(rows / blockM) + counts.length;
  const padded = tiles.length * blockM;
  const pidClamped = Math.min(pid, launched - 1);
  const mine = pidClamped < tiles.length ? tiles[pidClamped] : null;

  const margin = 8;
  const innerW = Math.max(width, 260) - 2 * margin;
  const unit = innerW / Math.max(rows, 1);
  const cell = Math.min(34, Math.max(10, (innerW - 4) / launched - 3));

  return (
    <Figure
      title="From a program id to an expert and a tile of rows"
      caption={
        <>
          The top strip is the sorted rows, coloured by expert. Below it, each square is one program along the grid&apos;s
          row axis. The launch has ceil(rows / BLOCK_M) + experts programs because the grid is sized before the counts are
          known. A program past the last real tile exits with no work. Choose a program id to see which expert and rows it
          owns.
        </>
      }
      controls={
        <>
          <Segmented
            label="Routing"
            value={preset}
            options={[
              { value: "ragged", label: "Ragged" },
              { value: "balanced", label: "Balanced" },
              { value: "skewed", label: "Skewed" },
            ]}
            onChange={(v) => {
              setPreset(v);
              setPid(0);
            }}
          />
          <Slider label="BLOCK_M" value={blockM} values={[16, 32, 64, 128]} onChange={setBlockM} />
          <Slider label="Program id" value={pidClamped} min={0} max={launched - 1} step={1} onChange={setPid} />
        </>
      }
      readout={
        <>
          <Stat label="Rows" value={rows} />
          <Stat label="Real tiles" value={tiles.length} tone={color.ok} />
          <Stat label="Programs launched" value={launched} />
          <Stat label="Exit immediately" value={launched - tiles.length} tone={color.muted} />
          <Stat label="Padding" value={`${padded === 0 ? 0 : (((padded - rows) / padded) * 100).toFixed(1)}%`} tone={color.accent} />
          <Stat
            label={`Program ${pidClamped}`}
            value={mine ? `expert ${mine.expert}, rows ${mine.firstRow} to ${mine.firstRow + mine.rows - 1}` : "no work"}
            tone={color.info}
          />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg
          width={Math.max(width, 260)}
          height={cell * 2 + 70}
          role="img"
          aria-label="Strip of sorted rows coloured by expert above a row of program squares"
        >
          {(() => {
            let x = margin;
            return counts.map((c, e) => {
              const w = c * unit;
              const node = (
                <rect key={e} x={x} y={6} width={Math.max(w - 1, c > 0 ? 1 : 0)} height={22} style={{ fill: TONES[e % TONES.length] }} />
              );
              x += w;
              return node;
            });
          })()}
          {Array.from({ length: launched }, (_, p) => {
            const t = p < tiles.length ? tiles[p] : null;
            const x = margin + p * (cell + 3);
            return (
              <g key={p}>
                <rect
                  x={x}
                  y={52}
                  width={cell}
                  height={cell}
                  rx={3}
                  style={{
                    fill: t ? TONES[t.expert % TONES.length] : wash(color.muted, 14),
                    stroke: p === pidClamped ? color.fg : "none",
                    strokeWidth: 2,
                    opacity: t ? 1 : 0.7,
                  }}
                />
                {t && t.rows < blockM ? (
                  <rect x={x} y={52 + cell - 4} width={cell} height={4} style={{ fill: wash(color.paper, 70) }} />
                ) : null}
              </g>
            );
          })}
          <text x={margin} y={46} style={{ fill: color.muted, fontSize: 11 }}>
            programs along the row axis
          </text>
        </svg>
      </div>
      <Legend
        items={[
          ...counts.map((c, e) => ({ label: `expert ${e}: ${c}`, tone: TONES[e % TONES.length] })),
          { label: "exits at once", tone: wash(color.muted, 40) },
        ]}
      />
    </Figure>
  );
}
