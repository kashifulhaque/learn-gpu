import { useState } from "react";
import { Figure, Segmented, Slider, StepControls, Stat, color, useStepper, useWidth, wash } from "../kit";

// The forward pass of FlashAttention-2 splits the N x N score matrix into
// T x T tiles. One program owns one row of tiles (one query tile) and walks
// along it. The counts are pure arithmetic on T: T^2 tiles in a full pass,
// T(T+1)/2 in a causal pass (the diagonal tiles are kept, and masked inside).
// The 108 is the A100's SM count (docs/FACTS.md section 2.1).
const SM_COUNT = 108;
const T_STOPS = [4, 6, 8, 12, 16] as const;

type Mode = "full" | "causal";

export default function TileGrid() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tiles, setTiles] = useState<number>(8);
  const [mode, setMode] = useState<Mode>("causal");
  const causal = mode === "causal";

  // Each program walks its row left to right. Row i has (causal ? i + 1 : T) tiles.
  const rowLen = (i: number) => (causal ? i + 1 : tiles);
  const longest = rowLen(tiles - 1);
  const stepper = useStepper(longest + 1, 600);
  const step = Math.min(stepper.step, longest);

  const total = Array.from({ length: tiles }, (_, i) => rowLen(i)).reduce((a, b) => a + b, 0);
  const full = tiles * tiles;
  const average = total / tiles;

  const size = Math.max(180, Math.min(width - 90, 360));
  const cell = size / tiles;
  const ox = 70;
  const oy = 28;
  const h = oy + size + 26;

  return (
    <Figure
      title="Which tiles one pass computes"
      caption="Each row is one program. A causal row stops at the diagonal, so the last program does T tiles of work while the average does (T + 1) / 2."
      controls={
        <>
          <Slider label="Tiles per side, T" values={T_STOPS} value={tiles} onChange={(v) => { setTiles(v); stepper.reset(); }} />
          <Segmented
            label="Mask"
            value={mode}
            onChange={(v) => { setMode(v); stepper.reset(); }}
            options={[
              { value: "full", label: "None" },
              { value: "causal", label: "Causal" },
            ]}
          />
          <StepControls stepper={stepper} count={longest + 1} describe={(s) => (s === 0 ? "nothing computed yet" : `tile ${s} of each row`)} />
        </>
      }
      readout={
        <>
          <Stat label="Tiles computed" value={`${total} of ${full}`} tone={color.a} />
          <Stat label="Share of a full pass" value={`${((total / full) * 100).toFixed(1)}%`} />
          <Stat label="Longest row vs average" value={`${longest} vs ${average.toFixed(1)}`} tone={color.c} />
          <Stat label="Programs per head" value={`${tiles} (A100 has ${SM_COUNT} SMs)`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label={`A ${tiles} by ${tiles} grid of score tiles. Rows are query tiles, columns are key tiles. ${causal ? "Tiles above the diagonal are skipped." : "Every tile is computed."}`}>
          <text x={ox} y={14} fontSize={11} style={{ fill: color.muted }}>
            key tiles (j) →
          </text>
          <text x={8} y={oy + size / 2} fontSize={11} style={{ fill: color.muted }} transform={`rotate(-90 14 ${oy + size / 2})`} textAnchor="middle">
            query tiles (i)
          </text>
          {Array.from({ length: tiles }).flatMap((_, i) =>
            Array.from({ length: tiles }).map((__, j) => {
              const kept = !causal || j <= i;
              const diagonal = causal && j === i;
              const done = kept && j < step;
              let fill: string = wash(color.faint, 12);
              let stroke: string = color.line;
              if (kept) {
                fill = done ? wash(color.a, 60) : wash(color.a, 22);
                stroke = color.a;
              }
              if (diagonal) {
                fill = done ? wash(color.c, 60) : wash(color.c, 25);
                stroke = color.c;
              }
              return (
                <rect key={`${i}-${j}`} x={ox + j * cell + 1} y={oy + i * cell + 1} width={Math.max(2, cell - 2)} height={Math.max(2, cell - 2)} rx={2} style={{ fill, stroke }} strokeWidth={1} />
              );
            }),
          )}
          <text x={ox} y={h - 6} fontSize={11} style={{ fill: color.muted }}>
            green: computed unmasked{causal ? " · amber: diagonal, masked inside the tile · grey: skipped, never launched" : ""}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
