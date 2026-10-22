import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants: 108 SMs on the A100 [A100-DS, GA100-WP], from docs/FACTS.md. A wave is one full
// set of resident blocks, so its capacity is SMs x resident blocks per SM (FACTS section 4.4).
const SMS = 108;
const COLS = 12;
const ROWS = 9; // 12 x 9 = 108
const MAX_BLOCKS = 540;

function waveCount(blocks: number, perSm: number): number {
  return Math.ceil(blocks / (perSm * SMS));
}

function efficiency(blocks: number, perSm: number): number {
  const waves = waveCount(blocks, perSm);
  return waves === 0 ? 1 : blocks / (waves * perSm * SMS);
}

export default function Waves() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [blocks, setBlocks] = useState<number>(109);
  const [perSm, setPerSm] = useState<number>(1);

  const capacity = perSm * SMS;
  const waves = waveCount(blocks, perSm);
  const eff = efficiency(blocks, perSm);
  const last = blocks - (waves - 1) * capacity; // blocks in the final wave
  const idleSlots = capacity - last;

  // The grid of SMs shows the final wave, filled slot by slot, row-major.
  const cell = Math.max(10, Math.min(28, Math.floor((width - 24) / COLS) - 3));
  const gridW = COLS * (cell + 3);
  const gridH = ROWS * (cell + 3);
  const slotsOn = (sm: number) => Math.max(0, Math.min(perSm, last - sm * perSm));

  // The sawtooth: efficiency against grid size.
  const ch = 150;
  const m = { l: 40, r: 12, t: 10, b: 28 };
  const pw = Math.max(120, width - m.l - m.r);
  const ph = ch - m.t - m.b;
  const sx = (b: number) => m.l + ((b - 1) / (MAX_BLOCKS - 1)) * pw;
  const sy = (e: number) => m.t + ph - e * ph;
  const points: string[] = [];
  for (let b = 1; b <= MAX_BLOCKS; b++) {
    points.push(`${b === 1 ? "M" : "L"}${sx(b).toFixed(1)},${sy(efficiency(b, perSm)).toFixed(1)}`);
  }

  return (
    <Figure
      title="Waves and the tail"
      caption="Each block runs on one SM slot for the same time, so a grid takes as many block-times as it has waves. The grid shows the last wave: dark slots are busy and pale ones are idle."
      controls={
        <>
          <Slider label="Blocks in the grid" min={1} max={MAX_BLOCKS} step={1} value={blocks} onChange={setBlocks} />
          <Segmented
            label="Resident blocks per SM"
            value={perSm}
            onChange={setPerSm}
            options={[
              { value: 1, label: "1" },
              { value: 2, label: "2" },
              { value: 4, label: "4" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Waves" value={waves} tone={color.b} />
          <Stat label="Blocks in last wave" value={`${last} of ${capacity}`} />
          <Stat label="Idle slots, last wave" value={idleSlots} tone={color.warn} />
          <Stat label="Efficiency" value={`${(eff * 100).toFixed(1)}%`} tone={color.a} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={gridH + 8}
          role="img"
          aria-label={`The ${SMS} SMs during the last of ${waves} waves: ${last} of ${capacity} block slots are busy.`}
        >
          <g transform={`translate(${Math.max(0, (width - gridW) / 2)}, 4)`}>
            {Array.from({ length: SMS }, (_, sm) => {
              const col = sm % COLS;
              const row = Math.floor(sm / COLS);
              const on = slotsOn(sm);
              const slotH = cell / perSm;
              return (
                <g key={sm} transform={`translate(${col * (cell + 3)}, ${row * (cell + 3)})`}>
                  <rect width={cell} height={cell} rx={2} style={{ fill: wash(color.faint, 25), stroke: color.line }} />
                  {Array.from({ length: on }, (_, s) => (
                    <rect key={s} y={cell - (s + 1) * slotH} width={cell} height={slotH - 1} rx={1} style={{ fill: color.b }} />
                  ))}
                </g>
              );
            })}
          </g>
        </svg>
        <svg
          width={width}
          height={ch}
          role="img"
          aria-label={`Efficiency against grid size for ${perSm} resident block per SM. It drops to just over half whenever the grid has one block more than a whole number of waves.`}
        >
          {[0, 0.5, 1].map((e) => (
            <g key={e}>
              <line x1={m.l} x2={m.l + pw} y1={sy(e)} y2={sy(e)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={sy(e) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {Math.round(e * 100)}%
              </text>
            </g>
          ))}
          <path d={points.join(" ")} fill="none" strokeWidth={1.5} style={{ stroke: color.a }} />
          <line x1={sx(blocks)} x2={sx(blocks)} y1={m.t} y2={m.t + ph} style={{ stroke: color.accent, strokeDasharray: "3 3" }} />
          <circle cx={sx(blocks)} cy={sy(eff)} r={5} strokeWidth={2} style={{ fill: wash(color.accent, 35), stroke: color.accent }} />
          <text x={m.l + pw / 2} y={ch - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            blocks in the grid
          </text>
        </svg>
        <Legend
          items={[
            { label: "Busy slot", tone: color.b },
            { label: "Efficiency by grid size", tone: color.a },
            { label: "Your grid", tone: color.accent },
          ]}
        />
      </div>
    </Figure>
  );
}
