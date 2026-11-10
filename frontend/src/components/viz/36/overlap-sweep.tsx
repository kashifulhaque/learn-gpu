import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Times in microseconds from scripts/figures/36-symmetric-memory.py, pasted
// from its printed JSON. The model is the alpha-beta cost of one all-gather
// plus one GEMM on 8 GPUs: NVLink 3 at 300 GB/s per direction and 312 TFLOP/s
// dense bf16 (docs/FACTS.md). The latencies are assumptions, not measurements:
// alpha 5 us per ring message, 5 us per extra kernel launch, 2 us per signal,
// and 10% of the compute rate lost to the SMs that communicate.
//   A: 8192 x 8192 bf16 gathered (128 MiB), GEMM against 8192 x 1024.
//   B: 256 x 8192 bf16 gathered (4 MiB), GEMM against 8192 x 1024.

const CHUNKS = [1, 2, 4, 8, 16, 32, 64] as const;

const DATA = {
  A: {
    serial: 866.978,
    overlapped: [916.029, 725.295, 657.608, 737.038, 986.753, 1531.611, 2644.04],
    fused: [878.029, 682.295, 584.428, 535.494, 511.027, 498.794, 527.04],
  },
  B: {
    serial: 60.999,
    overlapped: [67.376, 94.805, 161.019, 299.126, 578.18, 1137.707, 2257.47],
    fused: [29.376, 23.805, 24.019, 30.126, 45.18, 76.707, 140.47],
  },
} as const;

type Case = keyof typeof DATA;

function winner(serial: number, overlapped: number, fused: number): string {
  const best = Math.min(serial, overlapped, fused);
  if (best === serial) return "serial";
  if (best === overlapped) return "overlapped";
  return "fused";
}

export default function OverlapSweep() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [which, setWhich] = useState<Case>("A");
  const [chunks, setChunks] = useState<number>(8);

  const d = DATA[which];
  const idx = CHUNKS.indexOf(chunks as (typeof CHUNKS)[number]);
  const ov = d.overlapped[idx];
  const fu = d.fused[idx];

  const w = Math.max(width, 320);
  const h = 230;
  const pad = { l: 52, r: 12, t: 12, b: 34 };
  const top = Math.max(d.serial, ...d.overlapped.slice(0, 6), ...d.fused) * 1.05;
  const x = (i: number) => pad.l + (i / (CHUNKS.length - 1)) * (w - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - Math.min(v, top) / top) * (h - pad.t - pad.b);
  const path = (vals: readonly number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <Figure
      title="Serial, overlapped, and fused time against chunk count"
      caption="Separate collectives pay the ring latency and a kernel launch on every chunk, so their curve turns up early. A fused kernel pays one signal per chunk and keeps falling longer. Case B is small enough that the overlapped version never beats the serial line. The lines above the plot's top edge are clipped."
      controls={
        <>
          <Segmented
            label="Case"
            value={which}
            onChange={setWhich}
            options={[
              { value: "A", label: "A: 128 MiB gathered" },
              { value: "B", label: "B: 4 MiB gathered" },
            ]}
          />
          <Slider label="Chunks" values={CHUNKS} value={chunks} onChange={setChunks} />
        </>
      }
      readout={
        <>
          <Stat label="Serial" value={`${d.serial.toFixed(1)} us`} tone={color.faint} />
          <Stat label="Overlapped" value={`${ov.toFixed(1)} us`} tone={color.b} />
          <Stat label="Fused" value={`${fu.toFixed(1)} us`} tone={color.a} />
          <Stat label="Winner" value={winner(d.serial, ov, fu)} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Time in microseconds against chunk count for case ${which}. At ${chunks} chunks, serial is ${d.serial.toFixed(1)}, overlapped is ${ov.toFixed(1)}, and fused is ${fu.toFixed(1)}.`}
        >
          <line x1={pad.l} x2={w - pad.r} y1={h - pad.b} y2={h - pad.b} style={{ stroke: color.line }} />
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={pad.l} x2={w - pad.r} y1={y(top * f)} y2={y(top * f)} style={{ stroke: color.line }} strokeDasharray="2 4" />
              <text x={pad.l - 6} y={y(top * f) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {Math.round(top * f)}
              </text>
            </g>
          ))}
          {CHUNKS.map((c, i) => (
            <text key={c} x={x(i)} y={h - pad.b + 16} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
              {c}
            </text>
          ))}
          <text x={(pad.l + w - pad.r) / 2} y={h - 4} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            chunks (microseconds up)
          </text>
          <line x1={pad.l} x2={w - pad.r} y1={y(d.serial)} y2={y(d.serial)} strokeWidth={2} strokeDasharray="6 4" style={{ stroke: color.faint }} />
          <path d={path(d.overlapped)} fill="none" strokeWidth={2.5} style={{ stroke: color.b }} />
          <path d={path(d.fused)} fill="none" strokeWidth={2.5} style={{ stroke: color.a }} />
          <line x1={x(idx)} x2={x(idx)} y1={pad.t} y2={h - pad.b} style={{ stroke: color.lineStrong }} />
          <circle cx={x(idx)} cy={y(ov)} r={5} style={{ fill: color.b }} />
          <circle cx={x(idx)} cy={y(fu)} r={5} style={{ fill: color.a }} />
        </svg>
        <Legend
          items={[
            { label: "Serial", tone: color.faint, dashed: true },
            { label: "Overlapped (separate collectives)", tone: color.b },
            { label: "Fused (signals in one kernel)", tone: color.a },
          ]}
        />
      </div>
    </Figure>
  );
}
