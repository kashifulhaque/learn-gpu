import { useState } from "react";
import { Figure, Segmented, StepControls, Stat, color, useStepper, useWidth, wash } from "../kit";

// A warp is 32 lanes (docs/FACTS.md section 4.1). The toy warp has 8 lanes so
// the numbers stay legible; the real one needs five steps, offsets 16 to 1.
// __shfl_down_sync gives a lane whose source is out of range its own value
// back (CUDA Programming Guide, warp shuffle functions), and this figure
// models that rule exactly.

function simulate(lanes: number): number[][] {
  const rows: number[][] = [Array.from({ length: lanes }, (_, i) => i + 1)];
  for (let off = lanes >> 1; off > 0; off >>= 1) {
    const prev = rows[rows.length - 1];
    rows.push(prev.map((v, i) => v + (i + off < lanes ? prev[i + off] : v)));
  }
  return rows;
}

export default function WarpShuffle() {
  const [lanes, setLanes] = useState<number>(8);
  const [ref, width] = useWidth<HTMLDivElement>();
  const rows = simulate(lanes);
  const steps = rows.length - 1;
  const stepper = useStepper(steps, 900);
  const k = Math.min(stepper.step, steps - 1); // the shuffle round shown in accent
  const offsetOf = (round: number) => lanes >> (round + 1);

  const labelW = 74;
  const cellW = Math.max(8, (width - labelW - 8) / lanes);
  const showText = cellW >= 26;
  const rowH = 26;
  const rowGap = 26;
  const height = rows.length * (rowH + rowGap) - rowGap + 8;
  const rowY = (r: number) => r * (rowH + rowGap) + 4;
  const cx = (i: number) => labelW + i * cellW + cellW / 2;

  // A lane holds a true partial sum while every source it has added came from
  // inside the warp. Track that rule round by round.
  const okRows: boolean[][] = [Array.from({ length: lanes }, () => true)];
  for (let r = 0; r < steps; r++) {
    const off = offsetOf(r);
    const prev = okRows[r];
    okRows.push(prev.map((good, i) => good && i + off < lanes && prev[i + off]));
  }
  const valid = (r: number, i: number) => okRows[r][i];

  return (
    <Figure
      title="A warp sum with __shfl_down_sync"
      caption={
        <>
          Lane i starts holding the value i + 1. Each round, every lane adds the value of the lane <em>offset</em> places
          above it, so lane 0 ends holding the sum of all {lanes}: {(lanes * (lanes + 1)) / 2}. Faded numbers are lanes
          that ran past the end of the warp; nothing reads them.
        </>
      }
      controls={
        <>
          <Segmented
            label="Lanes drawn"
            value={lanes}
            options={[
              { value: 8, label: "8 (toy)" },
              { value: 32, label: "32 (real warp)" },
            ]}
            onChange={(v) => {
              setLanes(v);
              stepper.reset();
            }}
          />
          <StepControls stepper={stepper} count={steps} describe={(s) => `offset ${offsetOf(s)}`} />
        </>
      }
      readout={
        <>
          <Stat label="Shuffle rounds" value={steps} tone={color.a} />
          <Stat label="Shown: offset" value={offsetOf(k)} />
          <Stat label="Lane 0 after the last round" value={rows[steps][0]} tone={color.ok} />
          <Stat label="Shared memory used" value="0 bytes" />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Lane values after each shuffle round">
          {rows.map((vals, r) => (
            <g key={r}>
              <text x={0} y={rowY(r) + rowH / 2 + 4} style={{ fill: color.muted, fontSize: 11 }}>
                {r === 0 ? "start" : `offset ${offsetOf(r - 1)}`}
              </text>
              {vals.map((v, i) => {
                const ok = valid(r, i);
                return (
                  <g key={i}>
                    <rect
                      x={labelW + i * cellW + 0.5}
                      y={rowY(r)}
                      width={cellW - 1}
                      height={rowH}
                      rx={2}
                      style={{
                        fill: i === 0 ? wash(color.ok, 30) : ok ? wash(color.accent, r === 0 ? 10 : 22) : wash(color.muted, 8),
                        stroke: i === 0 ? color.ok : color.line,
                        strokeWidth: 1,
                      }}
                    />
                    {showText && (
                      <text
                        x={cx(i)}
                        y={rowY(r) + rowH / 2 + 4}
                        textAnchor="middle"
                        style={{ fill: ok ? color.fg : color.faint, fontSize: 11 }}
                      >
                        {v}
                      </text>
                    )}
                  </g>
                );
              })}
              {r < steps &&
                vals.map((_, i) => {
                  const off = offsetOf(r);
                  if (i + off >= lanes) return null;
                  const hot = r === k;
                  return (
                    <line
                      key={`a${i}`}
                      x1={cx(i + off)}
                      y1={rowY(r) + rowH}
                      x2={cx(i)}
                      y2={rowY(r + 1)}
                      style={{
                        stroke: hot ? color.accent : color.lineStrong,
                        strokeWidth: hot ? 1.6 : 0.6,
                        opacity: hot ? 1 : 0.55,
                      }}
                    />
                  );
                })}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
