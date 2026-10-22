import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Constants, each from docs/FACTS.md.
const COPY_GBS = 1275; // measured device-to-device copy ceiling, learn-inference chapter 10
const LAUNCH_US = 5; // host cost to queue one kernel, learn-inference chapter 12 (rule of thumb 5 to 10)
const BYTES_PER_ELEMENT = 4; // float32

/** Time for a pointwise kernel that reads and writes one tensor: bytes over bandwidth, floored by the launch. */
function opTime(elements: number) {
  const bytes = 2 * elements * BYTES_PER_ELEMENT;
  const memoryUs = (bytes / (COPY_GBS * 1e9)) * 1e6;
  return { memoryUs, us: Math.max(memoryUs, LAUNCH_US), launchBound: memoryUs < LAUNCH_US };
}

export default function FusionBytes() {
  const [chain, setChain] = useState(6);
  const [log2n, setLog2n] = useState(24);
  const [ref, width] = useWidth();

  const elements = 2 ** log2n;
  const tensorBytes = elements * BYTES_PER_ELEMENT;
  const one = opTime(elements);
  const eagerUs = chain * one.us;
  const fusedUs = one.us; // one read and one write, whatever the chain length
  const eagerBytes = chain * 2 * tensorBytes;
  const fusedBytes = 2 * tensorBytes;
  const speedup = eagerUs / fusedUs;

  const pad = 12;
  const labelW = 62;
  const barX = pad + labelW;
  const barW = Math.max(120, width - barX - pad);
  const scale = barW / eagerUs;
  const rowH = 34;
  const height = 2 * rowH + 28;
  const tone = one.launchBound ? color.warn : color.accent;

  const fmtUs = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(3)} ms` : `${us.toFixed(1)} us`);

  return (
    <Figure
      title="What fusing a pointwise chain saves"
      controls={
        <>
          <Slider label="Ops in the chain" value={chain} min={1} max={12} onChange={setChain} />
          <Slider
            label="Elements (float32)"
            value={log2n}
            min={12}
            max={26}
            onChange={setLog2n}
            format={(v) => `2^${v} = ${formatBytes(2 ** v * BYTES_PER_ELEMENT, 0)}`}
          />
        </>
      }
      readout={
        <>
          <Stat label="Bytes, eager" value={formatBytes(eagerBytes)} tone={tone} />
          <Stat label="Bytes, fused" value={formatBytes(fusedBytes)} tone={color.ok} />
          <Stat label="Predicted, eager" value={fmtUs(eagerUs)} tone={tone} />
          <Stat label="Predicted, fused" value={fmtUs(fusedUs)} tone={color.ok} />
          <Stat label="Predicted speedup" value={`${speedup.toFixed(1)}x`} />
          <Stat label="Each eager op is" value={one.launchBound ? "launch bound" : "bandwidth bound"} tone={tone} />
        </>
      }
      caption={
        <>
          Each eager op reads and writes a whole tensor, so it moves two tensors of bytes and takes the longer of that
          over {COPY_GBS} GB/s and a {LAUNCH_US} us launch. The fused kernel moves two tensors in total. The model
          ignores the 40 MiB L2 and the GPU&apos;s own minimum kernel time, so it is a prediction, and your lab
          measures the real figure.
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Eager chain as a row of kernels against one fused kernel">
          {[
            { label: "Eager", y: 6, segments: chain },
            { label: "Fused", y: 6 + rowH + 14, segments: 1 },
          ].map((row) => {
            const segW = one.us * scale;
            return (
              <g key={row.label}>
                <text x={pad} y={row.y + rowH / 2 + 4} style={{ fill: color.muted, fontSize: 12 }}>
                  {row.label}
                </text>
                {Array.from({ length: row.segments }).map((_, i) => (
                  <g key={i}>
                    <rect
                      x={barX + i * segW}
                      y={row.y}
                      width={Math.max(1, segW - 1)}
                      height={rowH}
                      rx={2}
                      style={{
                        fill: wash(row.label === "Fused" ? color.ok : tone, 35),
                        stroke: row.label === "Fused" ? color.ok : tone,
                      }}
                    />
                    {segW > 34 && (
                      <text
                        x={barX + i * segW + segW / 2}
                        y={row.y + rowH / 2 + 4}
                        textAnchor="middle"
                        style={{ fill: color.fg, fontSize: 11 }}
                      >
                        {row.label === "Fused" ? `1 kernel` : `op ${i + 1}`}
                      </text>
                    )}
                  </g>
                ))}
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Bandwidth-bound kernel", tone: color.accent },
            { label: "Launch-bound kernel", tone: color.warn },
            { label: "Fused kernel", tone: color.ok },
          ]}
        />
      </div>
    </Figure>
  );
}
