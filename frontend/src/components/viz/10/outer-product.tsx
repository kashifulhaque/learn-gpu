import { useState } from "react";
import { Figure, Legend, Segmented, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

// Constants (docs/FACTS.md):
//  - An A100 SM has 32 shared-memory banks x 4 B = 128 B per clock [CPG-BANKS] and 64 FP32 cores,
//    so it can retire 64 FMA per clock [GA100-WP].
//  - The budget is therefore 128 / 64 = 2 bytes of shared-memory reads per FMA.
//  - A thread with a TM x TN register tile reads (TM + TN) values per k step and does TM * TN FMAs,
//    so it needs 4 (TM + TN) / (TM * TN) bytes per FMA. This ignores broadcasts and bank conflicts.
const SMEM_BYTES_PER_CLOCK = 128;
const FMA_PER_CLOCK = 64;
const BUDGET = SMEM_BYTES_PER_CLOCK / FMA_PER_CLOCK; // 2 B per FMA
const BK = 8;

const TILES = [
  { value: "1x1", label: "1 x 1", tm: 1, tn: 1 },
  { value: "8x1", label: "8 x 1", tm: 8, tn: 1 },
  { value: "4x4", label: "4 x 4", tm: 4, tn: 4 },
  { value: "8x8", label: "8 x 8", tm: 8, tn: 8 },
] as const;
type TileKey = (typeof TILES)[number]["value"];

export default function OuterProduct() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [key, setKey] = useState<TileKey>("8x8");
  const stepper = useStepper(BK, 600);
  const tile = TILES.find((t) => t.value === key)!;
  const { tm, tn } = tile;

  const steps = stepper.step + 1; // k steps done, 1 to BK
  const loads = steps * (tm + tn);
  const fmas = steps * tm * tn;
  const bytesPerFma = (4 * (tm + tn)) / (tm * tn);
  const ceilingPct = Math.min(100, (BUDGET / bytesPerFma) * 100);

  const cell = Math.max(14, Math.min(30, Math.floor((width - 140) / (tn + 1))));
  const gridW = tn * cell;
  const gridH = tm * cell;
  const ox = Math.max(cell + 64, (width - gridW) / 2);
  const oy = cell + 34;
  const h = oy + gridH + 28;

  return (
    <Figure
      title="One thread, one outer product"
      caption="Each k step reads TM values of A and TN values of B into registers, then every pairing updates one accumulator. A bigger thread tile does more FMAs per shared-memory read."
      controls={
        <>
          <Segmented label="Thread tile" value={key} onChange={setKey} options={TILES.map((t) => ({ value: t.value, label: t.label }))} />
          <StepControls stepper={stepper} count={BK} describe={(s) => `k = ${s}`} />
        </>
      }
      readout={
        <>
          <Stat label="Reads per k step" value={`${tm + tn} values`} tone={color.a} />
          <Stat label="FMAs per k step" value={tm * tn} tone={color.b} />
          <Stat label="Smem bytes per FMA" value={bytesPerFma.toFixed(2)} tone={bytesPerFma > BUDGET ? color.warn : color.ok} />
          <Stat label="Budget" value={`${BUDGET.toFixed(1)} B per FMA`} />
          <Stat label="Smem allows" value={`${ceilingPct.toFixed(0)}% of FMA peak`} tone={ceilingPct < 100 ? color.warn : color.ok} />
          <Stat label={`After ${steps} of ${BK} steps`} value={`${loads} reads, ${fmas} FMAs`} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={h}
          role="img"
          aria-label={`A ${tm} by ${tn} thread tile. Each k step reads ${tm} values of A and ${tn} values of B and does ${tm * tn} FMAs, which is ${bytesPerFma.toFixed(2)} shared-memory bytes per FMA against a budget of ${BUDGET}.`}
        >
          <text x={ox + gridW / 2} y={14} textAnchor="middle" fontSize={11} style={{ fill: color.b }}>
            regB: {tn} value{tn > 1 ? "s" : ""} of B at this k
          </text>
          {Array.from({ length: tn }, (_, j) => (
            <rect key={`b${j}`} x={ox + j * cell + 1} y={22} width={cell - 2} height={cell - 2} rx={3} style={{ fill: wash(color.b, 45), stroke: color.b }} />
          ))}
          <text x={ox - cell - 8} y={oy + gridH / 2 - 8} textAnchor="end" fontSize={11} style={{ fill: color.a }}>
            regA
          </text>
          <text x={ox - cell - 8} y={oy + gridH / 2 + 6} textAnchor="end" fontSize={11} style={{ fill: color.a }}>
            {tm} of A
          </text>
          {Array.from({ length: tm }, (_, i) => (
            <rect key={`a${i}`} x={ox - cell - 2} y={oy + i * cell + 1} width={cell - 2} height={cell - 2} rx={3} style={{ fill: wash(color.a, 45), stroke: color.a }} />
          ))}
          {Array.from({ length: tm }, (_, i) =>
            Array.from({ length: tn }, (_, j) => (
              <rect
                key={`c${i}-${j}`}
                x={ox + j * cell + 1}
                y={oy + i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={3}
                style={{ fill: wash(color.c, 18 + (steps / BK) * 50), stroke: color.c }}
              />
            )),
          )}
          <text x={ox + gridW / 2} y={oy + gridH + 18} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            acc: {tm} x {tn} = {tm * tn} registers, each updated once per step (darker means more steps done)
          </text>
        </svg>
        <Legend
          items={[
            { label: "regA", tone: color.a },
            { label: "regB", tone: color.b },
            { label: "accumulators", tone: color.c },
          ]}
        />
      </div>
    </Figure>
  );
}
