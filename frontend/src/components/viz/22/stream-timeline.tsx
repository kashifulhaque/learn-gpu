import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// A timing model of a copy-compute-copy pipeline, not a measurement.
//
// Constants, each from the chapter:
// - The tensor is 2^26 float32 values: 268,435,456 bytes (256 MiB).
// - PCIe Gen4 x16 per direction: 16 GT/s x 128/130 x 16 lanes / 8 = 31.5 GB/s
//   (docs/FACTS.md section 7). 268.4e6 / 31.5e9 = 8.52 ms for each copy.
// - The kernel time is a slider; its default equals the copy time, the balance
//   the lab calibrates to.
// - Time of n chunks: T = s1 + s2 + s3 + (n - 1) * max(s1, s2, s3), where the
//   stage times s are the stage totals divided by n.
const BYTES = 268_435_456;
const PCIE_GBS = 31.5;
const COPY_MS = (BYTES / (PCIE_GBS * 1e9)) * 1e3; // 8.52 ms

type Mode = "one" | "many";

function schedule(n: number, h: number, k: number, d: number, mode: Mode) {
  const s1 = h / n;
  const s2 = k / n;
  const s3 = d / n;
  const bars: { lane: 0 | 1 | 2; chunk: number; start: number; end: number }[] = [];
  const hEnd: number[] = [];
  const kEnd: number[] = [];
  const dEnd: number[] = [];
  for (let i = 0; i < n; i++) {
    // One stream: every operation waits for the one before it. Many streams:
    // each engine takes its next chunk when it is free and the chunk's input is ready.
    const prev = i === 0 ? 0 : mode === "one" ? dEnd[i - 1] : hEnd[i - 1];
    const hs = prev;
    hEnd[i] = hs + s1;
    const ks = Math.max(hEnd[i], mode === "one" ? hEnd[i] : i === 0 ? 0 : kEnd[i - 1]);
    kEnd[i] = ks + s2;
    const ds = Math.max(kEnd[i], mode === "one" ? kEnd[i] : i === 0 ? 0 : dEnd[i - 1]);
    dEnd[i] = ds + s3;
    bars.push({ lane: 0, chunk: i, start: hs, end: hEnd[i] });
    bars.push({ lane: 1, chunk: i, start: ks, end: kEnd[i] });
    bars.push({ lane: 2, chunk: i, start: ds, end: dEnd[i] });
  }
  return { bars, total: dEnd[n - 1] };
}

export default function StreamTimeline() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [chunks, setChunks] = useState(8);
  const [kernel, setKernel] = useState(Math.round(COPY_MS * 10) / 10);
  const [mode, setMode] = useState<Mode>("many");

  const sequential = COPY_MS + kernel + COPY_MS;
  const bound = Math.max(COPY_MS, kernel);
  const { bars, total } = schedule(chunks, COPY_MS, kernel, COPY_MS, mode);

  const left = 92;
  const right = 12;
  const plotW = Math.max(120, width - left - right);
  const laneH = 26;
  const top = 8;
  const x = (t: number) => left + (t / sequential) * plotW;
  const height = top + 3 * (laneH + 6) + 30;
  const labels = ["H2D copy engine", "SMs (kernel)", "D2H copy engine"];
  const tones = [color.warn, color.b, color.ok];

  return (
    <Figure
      title="Chunked copy, kernel, copy: sequential against overlapped"
      controls={
        <>
          <Slider label="Chunks" value={chunks} values={[1, 2, 4, 8, 16, 32]} onChange={setChunks} />
          <Slider
            label="Kernel time (ms)"
            value={kernel}
            min={2}
            max={30}
            step={0.1}
            format={(v) => v.toFixed(1)}
            onChange={setKernel}
          />
          <Segmented
            label="Streams"
            value={mode}
            onChange={setMode}
            options={[
              { value: "one", label: "One stream" },
              { value: "many", label: "Stream per chunk" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Sequential (sum of stages)" value={`${sequential.toFixed(1)} ms`} />
          <Stat label="Bound (longest stage)" value={`${bound.toFixed(1)} ms`} tone={color.ok} />
          <Stat label={`Pipeline, ${chunks} chunk${chunks === 1 ? "" : "s"}`} value={`${total.toFixed(1)} ms`} tone={color.b} />
          <Stat label="Efficiency (bound / pipeline)" value={`${((bound / total) * 100).toFixed(0)}%`} />
        </>
      }
      caption={
        <>
          A model with {chunks} chunk{chunks === 1 ? "" : "s"} and {mode === "one" ? "one stream" : "a stream per chunk"}.
          Each copy is {COPY_MS.toFixed(2)} ms: 256 MiB at the PCIe Gen4 x16 rate of 31.5 GB/s per direction. The
          kernel time is yours to set. In one stream the chunks run in order and nothing overlaps; with several streams
          the three engines work on different chunks, and the time approaches the longest stage. The model ignores
          launch overhead and memory contention.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`Timeline of ${chunks} chunks on three engines`}>
          {[0, 1, 2].map((lane) => {
            const y = top + lane * (laneH + 6);
            return (
              <g key={lane}>
                <text x={left - 8} y={y + laneH / 2 + 4} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
                  {labels[lane]}
                </text>
                <rect x={left} y={y} width={plotW} height={laneH} rx={3} style={{ fill: color.well }} />
              </g>
            );
          })}
          {bars.map((bar) => {
            const y = top + bar.lane * (laneH + 6);
            const x0 = x(bar.start);
            const w = Math.max(0.5, x(bar.end) - x0);
            return (
              <g key={`${bar.lane}-${bar.chunk}`}>
                <rect
                  x={x0}
                  y={y + 2}
                  width={w}
                  height={laneH - 4}
                  rx={2}
                  style={{ fill: wash(tones[bar.lane], 40), stroke: tones[bar.lane], strokeWidth: w > 3 ? 1 : 0.5 }}
                />
                {w > 16 && chunks <= 16 && (
                  <text x={x0 + w / 2} y={y + laneH / 2 + 4} textAnchor="middle" style={{ fill: color.fg, fontSize: 10 }}>
                    {bar.chunk + 1}
                  </text>
                )}
              </g>
            );
          })}
          <line
            x1={x(bound)}
            x2={x(bound)}
            y1={top}
            y2={top + 3 * (laneH + 6) - 6}
            style={{ stroke: color.ok, strokeDasharray: "4 3" }}
          />
          <text x={left} y={height - 8} style={{ fill: color.subtle, fontSize: 10 }}>
            0
          </text>
          <text x={left + plotW} y={height - 8} textAnchor="end" style={{ fill: color.subtle, fontSize: 10 }}>
            {sequential.toFixed(1)} ms (sequential); dashed line: the bound
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Copy in", tone: color.warn },
          { label: "Kernel", tone: color.b },
          { label: "Copy out", tone: color.ok },
        ]}
      />
    </Figure>
  );
}
