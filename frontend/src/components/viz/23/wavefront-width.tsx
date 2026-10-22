import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// A warp-level sum written with shfl_down, lane 0 collecting. With the loop
// starting at offset s (a power of two), lane 0 adds lanes 0 .. 2s-1 and no
// others, so it sums min(2s, width) lanes. NVIDIA warps are 32 lanes wide; the
// AMD Instinct parts in docs/FACTS.md section 8.1 have wavefronts of 64.
type Start = 16 | 32;

export default function WavefrontWidth() {
  const [lanes, setLanes] = useState<32 | 64>(64);
  const [start, setStart] = useState<Start>(16);
  const [ref, width] = useWidth<HTMLDivElement>();

  const covered = Math.min(2 * start, lanes);
  const correct = covered === lanes;
  const steps = Math.log2(start) + 1;
  const perRow = 32;
  const rows = lanes / perRow;
  const cell = Math.max(8, Math.min(22, Math.floor((width - 4) / perRow)));
  const height = rows * (cell + 4) + 4;

  return (
    <Figure
      title="One hard-coded constant, two wavefront widths"
      caption={
        <>
          Each square is a lane. A shuffle-down reduction that starts at offset 16 gives lane 0 the sum of lanes 0 to 31. That
          is the whole warp on a 32-wide NVIDIA GPU and half the wavefront on a 64-wide AMD Instinct GPU. Starting at 32 fixes
          it, at the cost of one more step.
        </>
      }
      controls={
        <>
          <Segmented
            label="Lanes per warp or wavefront"
            value={lanes}
            options={[
              { value: 32, label: "32 (NVIDIA)" },
              { value: 64, label: "64 (CDNA)" },
            ]}
            onChange={setLanes}
          />
          <Segmented
            label="First shuffle offset"
            value={start}
            options={[
              { value: 16, label: "16 (written for 32)" },
              { value: 32, label: "32 (written for 64)" },
            ]}
            onChange={setStart}
          />
        </>
      }
      readout={
        <>
          <Stat label="Lanes in lane 0's sum" value={`${covered} of ${lanes}`} tone={correct ? color.ok : color.bad} />
          <Stat label="Shuffle steps" value={steps} />
          <Stat label="Result" value={correct ? "complete sum" : "silently partial"} tone={correct ? color.ok : color.bad} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`${covered} of ${lanes} lanes summed into lane 0`}>
          {Array.from({ length: lanes }, (_, i) => {
            const r = Math.floor(i / perRow);
            const c = i % perRow;
            const inSum = i < covered;
            return (
              <rect
                key={i}
                x={2 + c * cell}
                y={2 + r * (cell + 4)}
                width={cell - 2}
                height={cell}
                rx={2}
                style={{ fill: inSum ? wash(color.ok, 80) : wash(color.bad, 55) }}
              />
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "lane contributes to lane 0", tone: color.ok },
          { label: "lane is never added", tone: color.bad },
        ]}
      />
    </Figure>
  );
}
