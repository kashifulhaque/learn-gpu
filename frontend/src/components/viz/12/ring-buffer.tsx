import { useState } from "react";
import { Figure, Legend, Segmented, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

// The kernel's main loop for a short K, drawn as a ring of shared-memory slots.
// The schedule is the one in the chapter and in the lab's `schedule` function:
// prologue loads tiles 0 to S - 2; iteration kt waits until at most S - 2 groups
// are pending, synchronizes, loads tile kt + S - 1 into the slot tile kt - 1 used,
// commits a group, and computes tile kt.
const K_TILES = 8;

type Slot = { tile: number | null; state: "computing" | "inflight" | "empty" };

export default function RingBuffer() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [stages, setStages] = useState(3);
  const stepper = useStepper(K_TILES, 900);
  const kt = stepper.step;

  // State after the load for tile kt + S - 1 is issued, during compute of tile kt.
  const slots: Slot[] = [];
  for (let s = 0; s < stages; s++) slots.push({ tile: null, state: "empty" });
  for (let t = kt; t <= kt + stages - 1; t++) {
    if (t >= K_TILES) continue;
    slots[t % stages] = { tile: t, state: t === kt ? "computing" : "inflight" };
  }
  const committed = stages - 1 + kt + 1;
  const wait = stages - 2;
  const pendingBeforeWait = stages - 1; // groups for tiles kt .. kt + S - 2 at the top of the iteration

  const slotW = Math.min(110, (width - 24) / stages - 8);
  const slotH = 54;
  const height = slotH + 30 + 46;
  const stripW = Math.min(width - 24, 420);
  const cell = stripW / K_TILES;

  const fillFor = (state: Slot["state"]) =>
    state === "computing" ? wash(color.b, 40) : state === "inflight" ? wash(color.warn, 45) : color.well;
  const strokeFor = (state: Slot["state"]) =>
    state === "computing" ? color.b : state === "inflight" ? color.warn : color.line;

  return (
    <Figure
      title="The ring of stages, one loop iteration at a time"
      controls={
        <>
          <Segmented
            label="Stages"
            value={stages}
            options={[
              { value: 2, label: "2" },
              { value: 3, label: "3" },
              { value: 4, label: "4" },
            ]}
            onChange={(v) => {
              setStages(v);
              stepper.reset();
            }}
          />
          <StepControls stepper={stepper} count={K_TILES} describe={(s) => `computing k-tile ${s}`} />
        </>
      }
      readout={
        <>
          <Stat label="Wait before computing" value={`wait_group ${wait}`} tone={color.a} />
          <Stat label="Groups pending at the wait" value={pendingBeforeWait} />
          <Stat label="Groups committed so far" value={committed} />
          <Stat label="Copies in flight during compute" value={Math.min(stages - 1, Math.max(0, K_TILES - 1 - kt))} tone={color.warn} />
        </>
      }
      caption={
        <>
          Eight k-tiles through {stages} slots. During the compute of tile {kt}, tile {kt} sits in slot {kt % stages} and
          the next {stages - 1} tile{stages === 2 ? " is" : "s are"} in flight in the other slot{stages === 2 ? "" : "s"}.
          The wait counts groups, not tiles, so its argument is always stages - 2.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Shared memory ring with ${stages} slots at iteration ${kt}`}>
          {slots.map((slot, i) => {
            const x0 = 12 + i * (slotW + 8);
            return (
              <g key={i}>
                <rect x={x0} y={8} width={slotW} height={slotH} rx={4} style={{ fill: fillFor(slot.state), stroke: strokeFor(slot.state) }} />
                <text x={x0 + slotW / 2} y={8 + 22} textAnchor="middle" style={{ fill: color.muted, fontSize: 10 }}>
                  slot {i}
                </text>
                <text x={x0 + slotW / 2} y={8 + 42} textAnchor="middle" style={{ fill: color.fg, fontSize: 12 }}>
                  {slot.tile === null ? "empty" : `tile ${slot.tile}`}
                </text>
              </g>
            );
          })}
          {Array.from({ length: K_TILES }, (_, t) => {
            const state = t < kt ? "done" : t === kt ? "computing" : t <= kt + stages - 1 ? "inflight" : "later";
            const fill = state === "done" ? wash(color.ok, 35) : state === "computing" ? wash(color.b, 40) : state === "inflight" ? wash(color.warn, 45) : color.well;
            const stroke = state === "done" ? color.ok : state === "computing" ? color.b : state === "inflight" ? color.warn : color.line;
            return (
              <g key={t}>
                <rect x={12 + t * cell} y={slotH + 26} width={cell - 3} height={20} rx={3} style={{ fill, stroke }} />
                <text x={12 + t * cell + (cell - 3) / 2} y={slotH + 40} textAnchor="middle" style={{ fill: color.fg, fontSize: 10 }}>
                  {t}
                </text>
              </g>
            );
          })}
          <text x={12} y={slotH + 20} style={{ fill: color.subtle, fontSize: 10 }}>
            k-tiles along K
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Computing", tone: color.b },
          { label: "Copy in flight", tone: color.warn },
          { label: "Done", tone: color.ok },
        ]}
      />
    </Figure>
  );
}
