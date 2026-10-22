import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants.
const LAUNCH_US = 5; // host cost to queue one kernel: learn-inference chapter 12 (rule of thumb 5 to 10 us)
// NVIDIA's V100 measurement: a 2.9 us kernel took 3.4 us per kernel inside a graph of 20 kernels, so the
// gap between kernels is about 0.5 us. docs/FACTS.md section 4.5, [CUDA-GRAPHS-BLOG].
const GRAPH_GAP_US = 0.5;

type Span = { start: number; end: number };

/** Eager: the host queues kernel i by (i + 1) * LAUNCH; the GPU runs it when it is queued and the GPU is free. */
function eagerTimeline(count: number, duration: number) {
  const host: Span[] = [];
  const gpu: Span[] = [];
  let free = 0;
  for (let i = 0; i < count; i++) {
    host.push({ start: i * LAUNCH_US, end: (i + 1) * LAUNCH_US });
    const start = Math.max((i + 1) * LAUNCH_US, free);
    gpu.push({ start, end: start + duration });
    free = start + duration;
  }
  return { host, gpu, total: free };
}

/** Graph: one launch, then the GPU runs the kernels with a small gap between them. */
function graphTimeline(count: number, duration: number) {
  const host: Span[] = [{ start: 0, end: LAUNCH_US }];
  const gpu: Span[] = [];
  let t = LAUNCH_US;
  for (let i = 0; i < count; i++) {
    gpu.push({ start: t, end: t + duration });
    t += duration + (i < count - 1 ? GRAPH_GAP_US : 0);
  }
  return { host, gpu, total: t };
}

export default function GraphTimeline() {
  const [count, setCount] = useState(8);
  const [duration, setDuration] = useState(3);
  const [ref, width] = useWidth();

  const eager = eagerTimeline(count, duration);
  const graph = graphTimeline(count, duration);
  const horizon = Math.max(eager.total, graph.total);

  const pad = 12;
  const labelW = 70;
  const x0 = pad + labelW;
  const plotW = Math.max(120, width - x0 - pad);
  const sx = (t: number) => x0 + (t / horizon) * plotW;
  const laneH = 20;
  const lanes = [
    { label: "Eager host", spans: eager.host, tone: color.warn },
    { label: "Eager GPU", spans: eager.gpu, tone: color.accent },
    { label: "Graph host", spans: graph.host, tone: color.warn },
    { label: "Graph GPU", spans: graph.gpu, tone: color.ok },
  ];
  const height = lanes.length * (laneH + 10) + 28;
  const busy = count * duration;
  const idle = (total: number) => Math.round((1 - busy / total) * 100);

  return (
    <Figure
      title="Eager launches against one graph launch"
      controls={
        <>
          <Slider label="Kernels in the step" value={count} min={2} max={24} onChange={setCount} />
          <Slider
            label="GPU time per kernel (us)"
            value={duration}
            min={1}
            max={20}
            onChange={setDuration}
            format={(v) => `${v} us`}
          />
        </>
      }
      readout={
        <>
          <Stat label="Eager total" value={`${eager.total.toFixed(1)} us`} tone={color.accent} />
          <Stat label="Graph total" value={`${graph.total.toFixed(1)} us`} tone={color.ok} />
          <Stat label="Predicted speedup" value={`${(eager.total / graph.total).toFixed(2)}x`} />
          <Stat label="GPU idle, eager" value={`${idle(eager.total)}%`} tone={color.accent} />
          <Stat label="GPU idle, graph" value={`${idle(graph.total)}%`} tone={color.ok} />
        </>
      }
      caption={
        <>
          The host needs {LAUNCH_US} us to queue each kernel, so a kernel shorter than that leaves the GPU idle until
          the next arrives. A graph queues them all in one launch and runs them about {GRAPH_GAP_US} us apart. Make the
          kernels longer than {LAUNCH_US} us and the eager lane catches up: graphs stop mattering.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Host and GPU timelines, eager against graph replay">
          {lanes.map((lane, i) => {
            const y = 6 + i * (laneH + 10);
            return (
              <g key={lane.label}>
                <text x={pad} y={y + laneH / 2 + 4} style={{ fill: color.muted, fontSize: 11 }}>
                  {lane.label}
                </text>
                <rect x={x0} y={y} width={plotW} height={laneH} style={{ fill: color.well }} />
                {lane.spans.map((span, j) => (
                  <rect
                    key={j}
                    x={sx(span.start)}
                    y={y}
                    width={Math.max(1, sx(span.end) - sx(span.start) - 0.5)}
                    height={laneH}
                    style={{ fill: wash(lane.tone, 45), stroke: lane.tone }}
                  />
                ))}
              </g>
            );
          })}
          <text x={x0} y={height - 4} style={{ fill: color.subtle, fontSize: 10 }}>
            0
          </text>
          <text x={x0 + plotW} y={height - 4} textAnchor="end" style={{ fill: color.subtle, fontSize: 10 }}>
            {horizon.toFixed(0)} us
          </text>
        </svg>
        <Legend
          items={[
            { label: "Host queues a launch", tone: color.warn },
            { label: "Eager kernel", tone: color.accent },
            { label: "Kernel in a graph", tone: color.ok },
          ]}
        />
      </div>
    </Figure>
  );
}
