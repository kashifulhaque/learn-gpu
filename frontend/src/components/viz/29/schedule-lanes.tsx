import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// Per-SM work for two GEMM shapes on the A100's 108 SMs, one CTA per SM. The numbers are
// printed by scripts/figures/29-gemm-scheduling.py, which deals each schedule's CTAs to SMs
// in wave order and applies the cost model of lab 29 (312 TFLOP/s, 1275 GB/s, 5 us launch).
// A unit of work is one K-iteration of one output tile.
//   big    M=2304, N=1544, K=4096, tile 256x128x64: the 117-tile grid of chapter 4.
//   decode M=64, N=4096, K=4096, tile 64x128x64: a skinny GEMM with 32 tiles.
type Row = {
  schedule: string;
  splits: number;
  ctas: number;
  critical_iters: number;
  utilisation: number;
  reduced_tiles: number;
  cost_us: number;
  load: number[];
};
type Shape = { tiles: number; iters_per_tile: number; total_iters: number; rows: Row[] };
const DATA = {"big":{"tiles":117,"iters_per_tile":64,"total_iters":7488,"rows":[{"schedule":"data_parallel","splits":1,"ctas":117,"critical_iters":128,"utilisation":0.5417,"reduced_tiles":0,"cost_us":185.84,"compute_us":185.84,"memory_us":30.3,"extra_us":0.0,"load":[128,128,128,128,128,128,128,128,128,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]},{"schedule":"split_k","splits":2,"ctas":234,"critical_iters":96,"utilisation":0.7222,"reduced_tiles":117,"cost_us":189.02,"compute_us":139.38,"memory_us":30.3,"extra_us":49.64,"load":[96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]},{"schedule":"split_k","splits":4,"ctas":468,"critical_iters":80,"utilisation":0.8667,"reduced_tiles":117,"cost_us":210.43,"compute_us":116.15,"memory_us":30.3,"extra_us":94.28,"load":[80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]},{"schedule":"stream_k","splits":1,"ctas":108,"critical_iters":70,"utilisation":0.9905,"reduced_tiles":105,"cost_us":123.22,"compute_us":101.63,"memory_us":30.3,"extra_us":21.59,"load":[70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69]}]},"decode":{"tiles":32,"iters_per_tile":64,"total_iters":2048,"rows":[{"schedule":"data_parallel","splits":1,"ctas":32,"critical_iters":64,"utilisation":0.2963,"reduced_tiles":0,"cost_us":91.6,"compute_us":23.23,"memory_us":91.6,"extra_us":0.0,"load":[64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"schedule":"split_k","splits":3,"ctas":96,"critical_iters":22,"utilisation":0.862,"reduced_tiles":32,"cost_us":40.47,"compute_us":7.99,"memory_us":30.53,"extra_us":9.93,"load":[22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,22,21,21,0,0,0,0,0,0,0,0,0,0,0,0]},{"schedule":"split_k","splits":4,"ctas":128,"critical_iters":32,"utilisation":0.5926,"reduced_tiles":32,"cost_us":38.72,"compute_us":11.61,"memory_us":27.14,"extra_us":11.58,"load":[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]},{"schedule":"stream_k","splits":1,"ctas":108,"critical_iters":19,"utilisation":0.9981,"reduced_tiles":32,"cost_us":28.78,"compute_us":6.9,"memory_us":27.14,"extra_us":1.64,"load":[19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,18,18,18,18]}]}};

const SHAPE_LABEL: Record<string, string> = {
  big: "2304 x 1544 x 4096",
  decode: "64 x 4096 x 4096",
};

function label(r: Row): string {
  if (r.schedule === "data_parallel") return "Data-parallel";
  if (r.schedule === "split_k") return `Split-K ${r.splits}`;
  return "Stream-K";
}

export default function ScheduleLanes() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [shapeKey, setShapeKey] = useState<"big" | "decode">("big");
  const [pick, setPick] = useState(0);

  const shape = (DATA as Record<string, Shape>)[shapeKey];
  const row = shape.rows[Math.min(pick, shape.rows.length - 1)];
  const maxIters = Math.max(...shape.rows.map((r) => r.critical_iters));
  const ideal = shape.total_iters / 108;

  const m = { l: 40, r: 10, t: 12, b: 26 };
  const h = 170;
  const pw = Math.max(160, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const barW = pw / 108;
  const sy = (v: number) => m.t + ph - (v / maxIters) * ph;
  const best = Math.min(...shape.rows.map((r) => r.cost_us));

  return (
    <Figure
      title="Work per SM under four schedules"
      caption="Each bar is one of the 108 SMs, and its height is the K-iterations that SM runs. The kernel ends when the tallest bar ends. The dashed line is the perfectly even share. Times come from the lab 29 cost model, not from a measurement."
      controls={
        <>
          <Segmented
            label="Problem (M x N x K)"
            value={shapeKey}
            onChange={(v) => {
              setShapeKey(v);
              setPick(0);
            }}
            options={[
              { value: "big", label: SHAPE_LABEL.big },
              { value: "decode", label: SHAPE_LABEL.decode },
            ]}
          />
          <Segmented
            label="Schedule"
            value={pick}
            onChange={setPick}
            options={shape.rows.map((r, i) => ({ value: i, label: label(r) }))}
          />
        </>
      }
      readout={
        <>
          <Stat label="Critical path" value={`${row.critical_iters} iterations`} tone={color.a} />
          <Stat label="SM utilisation" value={`${(row.utilisation * 100).toFixed(1)}%`} />
          <Stat
            label={row.schedule === "split_k" ? "Tiles reduced" : "Tiles with fix-up"}
            value={row.schedule === "data_parallel" ? "0" : `${row.reduced_tiles} of ${shape.tiles}`}
            tone={color.warn}
          />
          <Stat label="Model time" value={`${row.cost_us.toFixed(1)} us`} tone={row.cost_us === best ? color.a : color.b} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, 200)}
          height={h}
          role="img"
          aria-label={`${label(row)} on ${SHAPE_LABEL[shapeKey]}: the busiest SM runs ${row.critical_iters} K-iterations and utilisation is ${(row.utilisation * 100).toFixed(1)} percent.`}
        >
          {row.load.map((v, i) => (
            <rect
              key={i}
              x={m.l + i * barW + 0.5}
              y={sy(v)}
              width={Math.max(1, barW - 1)}
              height={m.t + ph - sy(v)}
              style={{ fill: v === row.critical_iters ? color.a : wash(color.b, 55) }}
            />
          ))}
          <line x1={m.l} x2={m.l + pw} y1={sy(ideal)} y2={sy(ideal)} strokeDasharray="4 3" style={{ stroke: color.warn }} />
          <line x1={m.l} x2={m.l + pw} y1={m.t + ph} y2={m.t + ph} style={{ stroke: color.faint }} />
          <text x={m.l - 6} y={sy(maxIters) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            {maxIters}
          </text>
          <text x={m.l - 6} y={m.t + ph + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            0
          </text>
          <text x={m.l} y={h - 8} fontSize={11} style={{ fill: color.muted }}>
            SM 0
          </text>
          <text x={m.l + pw} y={h - 8} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            SM 107
          </text>
        </svg>
        <Legend
          items={[
            { label: "Busiest SMs", tone: color.a },
            { label: "Other SMs", tone: wash(color.b, 55) },
            { label: `Even share, ${ideal.toFixed(1)}`, tone: color.warn, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
