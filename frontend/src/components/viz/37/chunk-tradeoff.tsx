import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// The cost of the chunked forward pass against the chunk size C, for dk = dv = 128.
// Every row below is the output of scripts/figures/37-chunked-scans.py. The model
// is the one in the lab: FLOPs count every full C by C block, bytes count q, k, v,
// and o in bf16 plus each chunk state (fp32) written once and read once, and the
// time is the larger of the FLOP time and the byte time. The rates are the A100
// 80GB PCIe's 312 TFLOP/s dense bf16 (docs/FACTS.md) and the 1275 GB/s copy
// ceiling measured in learn-inference chapter 10. This is a model, not a measurement.

type Row = { c: number; gflop: number; mbytes: number; flop_us: number; byte_us: number };

const ROWS: Record<number, readonly Row[]> = {"4096": [{"c": 16, "gflop": 0.302, "mbytes": 37.749, "flop_us": 0.968, "byte_us": 29.607}, {"c": 32, "gflop": 0.336, "mbytes": 20.972, "flop_us": 1.075, "byte_us": 16.448}, {"c": 64, "gflop": 0.403, "mbytes": 12.583, "flop_us": 1.291, "byte_us": 9.869}, {"c": 128, "gflop": 0.537, "mbytes": 8.389, "flop_us": 1.721, "byte_us": 6.579}, {"c": 256, "gflop": 0.805, "mbytes": 6.291, "flop_us": 2.581, "byte_us": 4.934}, {"c": 512, "gflop": 1.342, "mbytes": 5.243, "flop_us": 4.302, "byte_us": 4.112}, {"c": 1024, "gflop": 2.416, "mbytes": 4.719, "flop_us": 7.743, "byte_us": 3.701}], "16384": [{"c": 16, "gflop": 1.208, "mbytes": 150.995, "flop_us": 3.872, "byte_us": 118.427}, {"c": 32, "gflop": 1.342, "mbytes": 83.886, "flop_us": 4.302, "byte_us": 65.793}, {"c": 64, "gflop": 1.611, "mbytes": 50.332, "flop_us": 5.162, "byte_us": 39.476}, {"c": 128, "gflop": 2.147, "mbytes": 33.554, "flop_us": 6.883, "byte_us": 26.317}, {"c": 256, "gflop": 3.221, "mbytes": 25.166, "flop_us": 10.324, "byte_us": 19.738}, {"c": 512, "gflop": 5.369, "mbytes": 20.972, "flop_us": 17.207, "byte_us": 16.448}, {"c": 1024, "gflop": 9.664, "mbytes": 18.874, "flop_us": 30.973, "byte_us": 14.803}], "65536": [{"c": 16, "gflop": 4.832, "mbytes": 603.98, "flop_us": 15.487, "byte_us": 473.71}, {"c": 32, "gflop": 5.369, "mbytes": 335.544, "flop_us": 17.207, "byte_us": 263.172}, {"c": 64, "gflop": 6.442, "mbytes": 201.327, "flop_us": 20.649, "byte_us": 157.903}, {"c": 128, "gflop": 8.59, "mbytes": 134.218, "flop_us": 27.532, "byte_us": 105.269}, {"c": 256, "gflop": 12.885, "mbytes": 100.663, "flop_us": 41.298, "byte_us": 78.952}, {"c": 512, "gflop": 21.475, "mbytes": 83.886, "flop_us": 68.83, "byte_us": 65.793}, {"c": 1024, "gflop": 38.655, "mbytes": 75.497, "flop_us": 123.893, "byte_us": 59.214}]};

const SOFTMAX_GFLOP: Record<number, number> = {"4096": 4.295, "16384": 68.719, "65536": 1099.512};

const LENGTHS = [4096, 16384, 65536] as const;
const CHUNKS = [16, 32, 64, 128, 256, 512, 1024] as const;

export default function ChunkTradeoff() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [t, setT] = useState<number>(16384);
  const [c, setC] = useState<number>(64);

  const rows = ROWS[t];
  const w = Math.max(width, 320);
  const h = 230;
  const pad = { l: 54, r: 14, t: 14, b: 40 };
  const maxV = Math.max(...rows.map((r) => Math.max(r.flop_us, r.byte_us)));
  const x = (i: number) => pad.l + (i / (CHUNKS.length - 1)) * (w - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / maxV) * (h - pad.t - pad.b);
  const line = (pick: (r: Row) => number) =>
    rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(pick(r)).toFixed(1)}`).join(" ");
  const idx = Math.max(0, CHUNKS.indexOf(c as (typeof CHUNKS)[number]));
  const at = rows[idx];
  const bound = at.flop_us >= at.byte_us ? "FLOP bound" : "memory bound";
  const bestRow = rows.reduce((p, r) => (Math.max(r.flop_us, r.byte_us) < Math.max(p.flop_us, p.byte_us) ? r : p), rows[0]);

  return (
    <Figure
      title="Chunk size trades FLOPs against state traffic"
      caption="Larger chunks do more intra-chunk work, which grows with C, and write fewer chunk states, which shrinks with 1/C. The modelled time is the higher of the two curves, so it bottoms out where they cross. The rates are the A100 80GB PCIe's tensor-core and copy-ceiling figures. This is a model, not a measurement: the lab sweeps the real kernel."
      controls={
        <>
          <Segmented
            label="Sequence length T"
            value={t}
            options={LENGTHS.map((v) => ({ value: v, label: v.toLocaleString("en-US") }))}
            onChange={setT}
          />
          <Slider label="Chunk size C" values={CHUNKS} value={c} onChange={setC} />
        </>
      }
      readout={
        <>
          <Stat label="Chunked FLOPs" value={`${at.gflop.toFixed(2)} GFLOP`} tone={color.a} />
          <Stat label="Causal softmax FLOPs" value={`${SOFTMAX_GFLOP[t].toLocaleString("en-US")} GFLOP`} tone={color.faint} />
          <Stat label="Bytes moved" value={`${at.mbytes.toFixed(1)} MB`} tone={color.b} />
          <Stat label="Modelled time" value={`${Math.max(at.flop_us, at.byte_us).toFixed(1)} us, ${bound}`} tone={color.accent} />
          <Stat label="Best C in the table" value={String(bestRow.c)} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Modelled time against chunk size at T = ${t}. At C = ${c}, FLOP time ${at.flop_us} microseconds and byte time ${at.byte_us} microseconds. The minimum is at C = ${bestRow.c}.`}
        >
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={pad.l} x2={w - pad.r} y1={y(maxV * f)} y2={y(maxV * f)} style={{ stroke: color.line }} />
              <text x={pad.l - 6} y={y(maxV * f) + 4} fontSize={10} textAnchor="end" style={{ fill: color.muted }}>
                {(maxV * f).toFixed(0)}
              </text>
            </g>
          ))}
          <text x={12} y={pad.t + 4} fontSize={10} style={{ fill: color.muted }}>
            us
          </text>
          <path d={line((r) => r.flop_us)} fill="none" strokeWidth={2} style={{ stroke: color.a }} />
          <path d={line((r) => r.byte_us)} fill="none" strokeWidth={2} style={{ stroke: color.b }} />
          {rows.map((r, i) => (
            <text
              key={r.c}
              x={x(i)}
              y={h - pad.b + 16}
              fontSize={10}
              textAnchor="middle"
              style={{ fill: r.c === c ? color.fg : color.muted, fontWeight: r.c === c ? 700 : 400 }}
            >
              {r.c}
            </text>
          ))}
          <text x={(pad.l + w - pad.r) / 2} y={h - 6} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            chunk size C
          </text>
          <line x1={x(idx)} x2={x(idx)} y1={pad.t} y2={h - pad.b} strokeDasharray="3 3" style={{ stroke: color.accent }} />
          <circle cx={x(idx)} cy={y(at.flop_us)} r={4} style={{ fill: color.a }} />
          <circle cx={x(idx)} cy={y(at.byte_us)} r={4} style={{ fill: color.b }} />
        </svg>
        <Legend
          items={[
            { label: "FLOP time at 312 TFLOP/s", tone: color.a },
            { label: "byte time at 1275 GB/s", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
