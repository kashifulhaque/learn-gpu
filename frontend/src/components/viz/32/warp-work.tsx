import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth } from "../kit";

// Work per warp, in lockstep iterations, for three SpMV schemes on the seeded
// 4096 by 4096 power-law matrix of lab 32 (scripts/figures/32-sparse-and-irregular-work.py,
// seed 32): 32,838 nonzeros, a mean row of 8.017 entries, a longest row of 1,153.
// There are 128 warps of 32 rows. thread-per-row: a warp runs as long as its longest
// row. warp-per-row: a row takes ceil(length / 32) iterations, at least 1, and warp w
// takes rows w, w + 128, ... merge-path: equal shares of rows + nonzeros, ceil(share / 32).
// These are model counts from the lab's reference solution, not GPU measurements.

type Scheme = "thread" | "warp" | "merge";

const WORK: Record<Scheme, number[]> = {
  thread: [75, 34, 64, 173, 26, 35, 15, 47, 169, 15, 32, 8, 202, 236, 31, 66, 47, 34, 22, 20, 15, 22, 18, 45, 80, 97, 17, 24, 224, 328, 66, 111, 32, 26, 46, 35, 23, 26, 58, 38, 32, 42, 9, 13, 26, 67, 45, 18, 118, 395, 565, 40, 138, 24, 77, 243, 66, 240, 41, 37, 56, 19, 36, 1056, 82, 47, 62, 17, 50, 309, 10, 25, 17, 80, 14, 23, 93, 28, 34, 46, 36, 142, 16, 36, 210, 1153, 40, 84, 17, 23, 58, 31, 20, 52, 966, 34, 34, 68, 259, 28, 40, 70, 44, 36, 212, 196, 363, 29, 51, 23, 112, 13, 35, 34, 24, 92, 139, 81, 147, 25, 707, 194, 46, 203, 16, 51, 31, 61],
  warp: [32, 34, 33, 32, 32, 32, 35, 33, 36, 34, 33, 44, 39, 33, 32, 36, 32, 33, 33, 32, 32, 32, 33, 33, 34, 32, 38, 55, 32, 41, 34, 38, 32, 39, 32, 51, 44, 35, 32, 36, 32, 35, 35, 40, 45, 32, 38, 33, 33, 35, 32, 32, 32, 33, 34, 32, 32, 32, 41, 32, 34, 69, 33, 33, 32, 33, 32, 38, 33, 33, 33, 33, 32, 32, 33, 32, 37, 33, 34, 33, 33, 33, 32, 36, 62, 32, 41, 35, 33, 49, 32, 34, 32, 32, 43, 33, 42, 32, 32, 33, 32, 33, 33, 38, 32, 35, 34, 32, 32, 32, 33, 32, 33, 33, 32, 32, 39, 32, 64, 33, 34, 33, 34, 32, 33, 34, 33, 32],
  merge: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8],
};

const NAMES: Record<Scheme, string> = {
  thread: "Thread per row",
  warp: "Warp per row",
  merge: "Merge path",
};

type Scale = "own" | "shared";

export default function WarpWork() {
  const [scheme, setScheme] = useState<Scheme>("thread");
  const [scale, setScale] = useState<Scale>("own");
  const [ref, width] = useWidth<HTMLDivElement>(560);

  const work = WORK[scheme];
  const max = Math.max(...work);
  const mean = work.reduce((a, b) => a + b, 0) / work.length;
  const sharedMax = Math.max(...WORK.thread);
  const top = scale === "own" ? max : sharedMax;

  const padL = 44;
  const padR = 12;
  const height = 200;
  const plotH = height - 34;
  const plotW = Math.max(200, width - padL - padR);
  const barW = plotW / work.length;
  const y = (v: number) => 8 + plotH - (v / top) * plotH;
  const tone = scheme === "merge" ? color.ok : scheme === "warp" ? color.b : color.accent;

  return (
    <Figure
      title="Work per warp on a power-law matrix"
      caption="Each bar is one of 128 warps. A thread-per-row warp runs as long as its longest row, so one warp runs 1,153 iterations while the mean is 104.7. A merge-path warp takes an equal share of rows plus nonzeros, and every warp runs 10 iterations except the last, which runs 8."
      controls={
        <>
          <Segmented
            label="Scheme"
            value={scheme}
            options={[
              { value: "thread", label: "Thread per row" },
              { value: "warp", label: "Warp per row" },
              { value: "merge", label: "Merge path" },
            ]}
            onChange={setScheme}
          />
          <Segmented
            label="Vertical scale"
            value={scale}
            options={[
              { value: "own", label: "Own maximum" },
              { value: "shared", label: "Shared with thread per row" },
            ]}
            onChange={setScale}
          />
        </>
      }
      readout={
        <>
          <Stat label="Scheme" value={NAMES[scheme]} tone={tone} />
          <Stat label="Max / mean" value={(max / mean).toFixed(2)} tone={tone} />
          <Stat label="Max iterations" value={String(max)} />
          <Stat label="Mean iterations" value={mean.toFixed(1)} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={padL + plotW + padR}
          height={height}
          role="img"
          aria-label={`${NAMES[scheme]}: 128 warps, maximum ${max} iterations, mean ${mean.toFixed(1)}, max over mean ${(max / mean).toFixed(2)}.`}
        >
          <line x1={padL} x2={padL + plotW} y1={8 + plotH} y2={8 + plotH} style={{ stroke: color.line }} />
          <line x1={padL} x2={padL} y1={8} y2={8 + plotH} style={{ stroke: color.line }} />
          <text x={padL - 6} y={12} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            {Math.round(top)}
          </text>
          <text x={padL - 6} y={8 + plotH} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            0
          </text>
          {work.map((value, index) => (
            <rect
              key={index}
              x={padL + index * barW + 0.5}
              y={y(value)}
              width={Math.max(1, barW - 1)}
              height={8 + plotH - y(value)}
              style={{ fill: tone }}
            />
          ))}
          <line
            x1={padL}
            x2={padL + plotW}
            y1={y(mean)}
            y2={y(mean)}
            strokeDasharray="4 3"
            style={{ stroke: color.fg }}
          />
          <text x={padL + plotW} y={height - 6} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            warp index, 0 to 127
          </text>
        </svg>
        <Legend items={[{ label: "Mean work per warp", tone: color.fg, dashed: true }]} />
      </div>
    </Figure>
  );
}
