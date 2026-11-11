import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";
import { DATA } from "./data";

// The software-pipeline schedule of an S-stage tile loop over 8 tiles. Each
// column is one step. Step s issues the load of tile s and runs the MMA of tile
// s - (S - 1). The numbers come from scripts/figures/38-tile-languages.py, which
// calls pipeline_schedule and max_resident from the lab's solution. One step is
// one unit of time, which assumes a load and an MMA take equally long.

export default function PipelineSchedule() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [stages, setStages] = useState(3);
  const [step, setStep] = useState(2);
  const row = DATA.pipeline.find((r) => r.stages === stages) ?? DATA.pipeline[0];
  const steps = row.schedule.length;
  const at = Math.min(step, steps - 1);

  // A tile is resident from its load step through its MMA step, inclusive.
  const resident: number[] = [];
  for (let t = 0; t < DATA.tiles; t++) {
    const loadStep = row.schedule.findIndex(([l]) => l === t);
    const mmaStep = row.schedule.findIndex(([, m]) => m === t);
    if (loadStep <= at && at <= mmaStep) resident.push(t);
  }

  const labelW = 44;
  const cell = Math.max(26, Math.min(44, Math.floor((width - labelW - 8) / steps)));
  const rowH = 34;

  return (
    <Figure
      title="An S-stage tile pipeline"
      caption="Row L issues the copy of a tile. Row M multiplies a tile. From step S - 1 the two rows are busy in the same column, so the copy of tile s + S - 1 hides behind the multiply of tile s. The outlined column is the step you picked, and the tiles listed under the drawing are the ones that occupy shared memory then."
      controls={
        <>
          <Slider label="Stages S" values={[2, 3, 4, 5]} value={stages} onChange={setStages} />
          <Slider label="Step" min={0} max={steps - 1} value={at} onChange={setStep} />
        </>
      }
      readout={
        <>
          <Stat label="Steps" value={`${row.steps} (serial ${row.serial_steps})`} tone={color.a} />
          <Stat label="Steps with overlap" value={String(row.overlap_steps)} tone={color.b} />
          <Stat label="Most tiles resident" value={String(row.resident)} tone={color.c} />
          <Stat label={`Resident at step ${at}`} value={resident.length ? resident.join(", ") : "none"} />
        </>
      }
    >
      <div ref={ref}>
        <Legend
          items={[
            { label: "L: copy tile", tone: color.a },
            { label: "M: multiply tile", tone: color.b },
          ]}
        />
        <svg
          width={Math.max(width, labelW + steps * cell + 8)}
          height={rowH * 2 + 40}
          role="img"
          aria-label={`Pipeline with ${stages} stages over ${DATA.tiles} tiles: ${row.steps} steps against ${row.serial_steps} serial, ${row.overlap_steps} steps with a copy and a multiply together.`}
        >
          {Array.from({ length: steps }, (_, s) => (
            <text
              key={`h${s}`}
              x={labelW + s * cell + cell / 2}
              y={12}
              fontSize={11}
              textAnchor="middle"
              style={{ fill: color.muted }}
            >
              {s}
            </text>
          ))}
          {(["L", "M"] as const).map((name, r) => (
            <g key={name}>
              <text x={labelW - 10} y={26 + r * rowH + 14} fontSize={12} textAnchor="end" style={{ fill: color.fg }}>
                {name}
              </text>
              {row.schedule.map((pair, s) => {
                const tile = pair[r];
                if (tile < 0) return null;
                const tone = r === 0 ? color.a : color.b;
                return (
                  <g key={`${name}${s}`}>
                    <rect
                      x={labelW + s * cell}
                      y={20 + r * rowH}
                      width={cell - 3}
                      height={rowH - 5}
                      rx={4}
                      style={{ fill: wash(tone, 55) }}
                    />
                    <text
                      x={labelW + s * cell + (cell - 3) / 2}
                      y={20 + r * rowH + (rowH - 5) / 2 + 4}
                      fontSize={12}
                      textAnchor="middle"
                      style={{ fill: color.fg }}
                    >
                      {tile}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
          <rect
            x={labelW + at * cell - 2}
            y={17}
            width={cell + 1}
            height={rowH * 2 - 1}
            rx={5}
            style={{ fill: "none", stroke: color.accent, strokeWidth: 2 }}
          />
        </svg>
      </div>
    </Figure>
  );
}
