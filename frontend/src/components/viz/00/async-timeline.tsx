import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// The host queues a launch in about 5 microseconds (measured on the lab A100 by
// learn-inference chapter 12). The default kernel time, 154 microseconds, is the
// elementwise kernel on 2^24 floats: 3 x 64 MiB at 1304 GB/s (docs/FACTS.md 4.5).
// This figure is a model of the three clocks, not a measurement.
const QUEUE_US = 5;
const KERNEL_STOPS = [20, 50, 154, 500, 2000] as const;

type Clock = "none" | "sync" | "events";

export default function AsyncTimeline() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [kernel, setKernel] = useState<number>(154);
  const [clock, setClock] = useState<Clock>("none");

  const total = QUEUE_US + kernel;
  const h = 190;
  const lx = 64;
  const pw = Math.max(140, width - lx - 16);
  const sx = (us: number) => lx + (us / total) * pw;
  const laneH = 34;
  const cpuY = 30;
  const gpuY = 84;

  const reading = clock === "none" ? QUEUE_US : clock === "sync" ? QUEUE_US + kernel : kernel;
  const label = clock === "none" ? "Host timer, no sync" : clock === "sync" ? "Host timer after sync" : "CUDA events";
  const error = ((reading - kernel) / kernel) * 100;

  // The timer's bracket on its own row.
  const t0 = clock === "events" ? QUEUE_US : 0;
  const t1 = clock === "none" ? QUEUE_US : total;

  return (
    <Figure
      title="Launches are asynchronous"
      caption="The host queues the kernel and moves on. Only a synchronize, or an event recorded on the GPU, observes when the kernel ends."
      controls={
        <>
          <Slider label="Kernel time" values={KERNEL_STOPS} value={kernel} onChange={setKernel} format={(v) => `${v} µs`} />
          <Segmented
            label="Clock"
            value={clock}
            onChange={setClock}
            options={[
              { value: "none", label: "Host, no sync" },
              { value: "sync", label: "Host + sync" },
              { value: "events", label: "CUDA events" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label={label} value={`${reading.toFixed(0)} µs`} tone={color.c} />
          <Stat label="Kernel runs for" value={`${kernel} µs`} tone={color.a} />
          <Stat label="Reading versus kernel" value={`${error > 0 ? "+" : ""}${error.toFixed(0)}%`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="A timeline with a host lane, where the launch takes about 5 microseconds, and a GPU lane, where the kernel runs afterward">
          <text x={lx - 8} y={cpuY + laneH / 2 + 4} textAnchor="end" fontSize={12} style={{ fill: color.muted }}>
            Host
          </text>
          <text x={lx - 8} y={gpuY + laneH / 2 + 4} textAnchor="end" fontSize={12} style={{ fill: color.muted }}>
            GPU
          </text>
          <rect x={lx} y={cpuY} width={pw} height={laneH} rx={4} style={{ fill: color.well }} />
          <rect x={lx} y={gpuY} width={pw} height={laneH} rx={4} style={{ fill: color.well }} />
          <rect x={sx(0)} y={cpuY} width={Math.max(3, sx(QUEUE_US) - sx(0))} height={laneH} rx={3} style={{ fill: wash(color.b, 45), stroke: color.b }} />
          <text x={sx(QUEUE_US) + 6} y={cpuY + laneH / 2 + 4} fontSize={11} style={{ fill: color.muted }}>
            queue launch, about 5 µs
          </text>
          <rect x={sx(QUEUE_US)} y={gpuY} width={sx(total) - sx(QUEUE_US)} height={laneH} rx={3} style={{ fill: wash(color.a, 45), stroke: color.a }} />
          <text x={(sx(QUEUE_US) + sx(total)) / 2} y={gpuY + laneH / 2 + 4} textAnchor="middle" fontSize={11} style={{ fill: color.fg }}>
            kernel runs {kernel} µs
          </text>
          {clock !== "events" && clock !== "none" && (
            <text x={sx(QUEUE_US) + 6} y={cpuY + laneH + 14} fontSize={11} style={{ fill: color.muted }}>
              host waits in synchronize
            </text>
          )}
          <line x1={sx(t0)} x2={sx(t1)} y1={150} y2={150} strokeWidth={4} style={{ stroke: color.c }} />
          <line x1={sx(t0)} x2={sx(t0)} y1={142} y2={158} style={{ stroke: color.c }} strokeWidth={2} />
          <line x1={sx(t1)} x2={sx(t1)} y1={142} y2={158} style={{ stroke: color.c }} strokeWidth={2} />
          <text x={lx} y={178} fontSize={11} style={{ fill: color.muted }}>
            {label}: {reading.toFixed(0)} µs
          </text>
          <text x={lx + pw} y={14} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            time axis spans {total} µs
          </text>
        </svg>
      </div>
    </Figure>
  );
}
