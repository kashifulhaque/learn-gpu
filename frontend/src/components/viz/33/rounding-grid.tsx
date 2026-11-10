import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// The positive E2M1 values: 1 sign bit, 2 exponent bits, 1 mantissa bit
// (OCP MX spec, NVIDIA NVFP4 post). scripts/figures/33-fp8-and-fp4.py prints
// the same grid and the round-to-nearest-even results used below.
const GRID = [0, 0.5, 1, 1.5, 2, 3, 4, 6];

// Round to nearest, ties to the even code (the code index is the grid index).
function rne(x: number): number {
  const a = Math.min(Math.abs(x), 6);
  let best = 0;
  for (let i = 1; i < GRID.length; i++) {
    const dLo = a - GRID[i - 1];
    const dHi = GRID[i] - a;
    if (a >= GRID[i - 1] && a <= GRID[i]) {
      if (dLo < dHi) best = i - 1;
      else if (dHi < dLo) best = i;
      else best = (i - 1) % 2 === 0 ? i - 1 : i;
    }
  }
  return GRID[best];
}

function neighbours(a: number): [number, number] {
  for (let i = 1; i < GRID.length; i++) {
    if (a <= GRID[i]) return [GRID[i - 1], GRID[i]];
  }
  return [6, 6];
}

export default function RoundingGrid() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [steps, setSteps] = useState(30);
  const x = steps / 20; // 0 to 6 in steps of 0.05
  const [lo, hi] = neighbours(x);
  const pUp = hi === lo ? 0 : (x - lo) / (hi - lo);
  const nearest = rne(x);

  const w = Math.max(width, 320);
  const pad = 24;
  const sx = (v: number) => pad + (v / 6) * (w - 2 * pad);
  const y0 = 70;

  return (
    <Figure
      title="Rounding a value to the E2M1 grid"
      caption="The eight positive E2M1 values are unevenly spaced. Round to nearest goes to one neighbour every time, so 0.3 becomes 0.5 and its error never averages out. Stochastic rounding goes up with probability equal to the fraction of the gap, so its mean is x. Exact ties, such as 0.75, go to the neighbour with the even code."
      controls={<Slider label="Value x" min={0} max={120} value={steps} onChange={setSteps} format={(v) => (v / 20).toFixed(2)} />}
      readout={
        <>
          <Stat label="Neighbours" value={`${lo} and ${hi}`} />
          <Stat label="Round to nearest" value={String(nearest)} tone={color.a} />
          <Stat label="P(round up)" value={pUp.toFixed(2)} tone={color.b} />
          <Stat label="Mean of stochastic" value={x.toFixed(2)} tone={color.c} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={120}
          role="img"
          aria-label={`The E2M1 grid with x equal to ${x.toFixed(2)}. Round to nearest gives ${nearest}. The stochastic probability of rounding up from ${lo} to ${hi} is ${pUp.toFixed(2)}.`}
        >
          <line x1={sx(0)} x2={sx(6)} y1={y0} y2={y0} style={{ stroke: color.lineStrong }} strokeWidth={2} />
          <rect x={sx(lo)} y={y0 - 6} width={Math.max(0, sx(hi) - sx(lo))} height={12} rx={3} style={{ fill: wash(color.b, 30) }} />
          {GRID.map((g) => (
            <g key={g}>
              <circle cx={sx(g)} cy={y0} r={5} style={{ fill: g === nearest ? color.a : color.fg }} />
              <text x={sx(g)} y={y0 + 26} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                {g}
              </text>
            </g>
          ))}
          <line x1={sx(x)} x2={sx(x)} y1={y0 - 34} y2={y0 - 8} style={{ stroke: color.c }} strokeWidth={2} />
          <text x={sx(x)} y={y0 - 40} fontSize={11} textAnchor="middle" style={{ fill: color.c }}>
            x = {x.toFixed(2)}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
