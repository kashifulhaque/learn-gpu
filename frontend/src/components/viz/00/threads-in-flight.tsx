import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants, all from docs/FACTS.md:
//  - 108 SMs, 2048 resident threads per SM (compute capability 8.0 limits, [CPG-CC]).
//  - 1935 GB/s HBM2e rating of the A100 80GB PCIe [A100-DS].
//  - 1275 GB/s measured device-to-device copy (learn-inference chapter 10).
//  - HBM latency 466.3 cycles at 1410 MHz = 330.7 ns, measured on an A100 PCIe 40GB [LUO-HOPPER].
const SMS = 108;
const RATED_GBS = 1935;
const COPY_GBS = 1275;
const LATENCY_NS = 466.3 / 1.41;
const THREAD_STOPS = [32, 64, 128, 256, 512, 1024, 1536, 2048] as const;

/** Little's law: bytes in flight divided by latency, capped by the rating. */
function predictedGbs(threadsPerSm: number, bytesPerThread: number): number {
  const inFlight = threadsPerSm * SMS * bytesPerThread; // bytes outstanding across the GPU
  return Math.min(RATED_GBS, inFlight / LATENCY_NS); // bytes per ns is GB/s
}

export default function ThreadsInFlight() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [threads, setThreads] = useState<number>(256);
  const [bytes, setBytes] = useState<number>(4);

  const h = 230;
  const m = { l: 52, r: 16, t: 14, b: 40 };
  const pw = Math.max(120, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const x = (t: number) => m.l + (Math.log2(t / 32) / Math.log2(2048 / 32)) * pw;
  const y = (g: number) => m.t + ph - (g / 2200) * ph;

  const curve = [] as string[];
  for (let i = 0; i <= 60; i++) {
    const t = 32 * 2 ** ((i / 60) * Math.log2(64));
    curve.push(`${i === 0 ? "M" : "L"}${x(t).toFixed(1)},${y(predictedGbs(t, bytes)).toFixed(1)}`);
  }
  const now = predictedGbs(threads, bytes);
  const inFlightKb = (threads * SMS * bytes) / 1000;

  return (
    <Figure
      title="Threads in flight and memory bandwidth"
      caption="Little's law with one outstanding load per thread: bandwidth equals bytes in flight divided by latency, up to the 1935 GB/s rating. A model, not a measurement."
      controls={
        <>
          <Slider label="Resident threads per SM" values={THREAD_STOPS} value={threads} onChange={setThreads} />
          <Segmented
            label="Bytes per load"
            value={bytes}
            onChange={setBytes}
            options={[
              { value: 4, label: "4 (float)" },
              { value: 8, label: "8" },
              { value: 16, label: "16 (float4)" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Bytes in flight, whole GPU" value={`${inFlightKb.toFixed(0)} KB`} tone={color.b} />
          <Stat label="Predicted bandwidth" value={`${now.toFixed(0)} GB/s`} tone={color.a} />
          <Stat label="Needed to saturate" value="640 KB" />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="Predicted bandwidth rises with resident threads per SM until it reaches the rated 1935 GB/s">
          {[0, 500, 1000, 1500, 2000].map((g) => (
            <g key={g}>
              <line x1={m.l} x2={m.l + pw} y1={y(g)} y2={y(g)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={y(g) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {g}
              </text>
            </g>
          ))}
          <line x1={m.l} x2={m.l + pw} y1={y(RATED_GBS)} y2={y(RATED_GBS)} strokeDasharray="5 4" style={{ stroke: color.faint }} />
          <text x={m.l + pw} y={y(RATED_GBS) - 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            rated 1935
          </text>
          <line x1={m.l} x2={m.l + pw} y1={y(COPY_GBS)} y2={y(COPY_GBS)} strokeDasharray="2 3" style={{ stroke: color.warn }} />
          <text x={m.l + pw} y={y(COPY_GBS) + 13} textAnchor="end" fontSize={11} style={{ fill: color.warn }}>
            measured copy 1275
          </text>
          <path d={curve.join(" ")} fill="none" strokeWidth={2.5} style={{ stroke: color.a }} />
          {THREAD_STOPS.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.t + ph} y2={m.t + ph + 4} style={{ stroke: color.lineStrong }} />
              <text x={x(t)} y={m.t + ph + 16} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
                {t}
              </text>
            </g>
          ))}
          <text x={m.l + pw / 2} y={h - 6} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            resident threads per SM (log scale)
          </text>
          <text x={14} y={m.t + ph / 2} fontSize={11} textAnchor="middle" transform={`rotate(-90 14 ${m.t + ph / 2})`} style={{ fill: color.muted }}>
            GB/s
          </text>
          <line x1={x(threads)} x2={x(threads)} y1={y(now)} y2={m.t + ph} style={{ stroke: color.b, strokeDasharray: "3 3" }} />
          <circle cx={x(threads)} cy={y(now)} r={6} style={{ fill: wash(color.b, 35), stroke: color.b }} strokeWidth={2} />
        </svg>
        <Legend
          items={[
            { label: "Predicted bandwidth", tone: color.a },
            { label: "Selected point", tone: color.b },
            { label: "Measured copy ceiling", tone: color.warn, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
