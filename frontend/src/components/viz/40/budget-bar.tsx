import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, useWidth } from "../kit";

// The chapter's worked example: a 24-layer, 2048-wide model with 16 heads,
// 4096 tokens per sequence, fused attention, and mixed-precision Adam.
// Formulas: state = 16 bytes per parameter with 12*h^2 parameters per layer;
// a layer saves 34*s*b*h bytes and keeps 2*s*b*h at its boundary; the peak is
// the lab's peak_activation_bytes with layers 0..k-1 checkpointed.
// scripts/figures/40-memory-management-and-oom.py prints the same numbers:
// state 18.00 GiB; at batch 12 a layer is 3264 MiB, 76.50 GiB for 24 layers,
// and 7 checkpointed layers bring the peak to 55.50 GiB.

const H = 2048;
const S = 4096;
const LAYERS = 24;
const STATE = 16 * 12 * H * H * LAYERS;
const GIB = 1024 ** 3;
const CARD = 80 * GIB; // the chapter treats the 80 GB card as 80 GiB
const HEADROOM = 4 * GIB; // an assumption of the chapter, not a measurement

function peakActivations(batch: number, k: number): number {
  const full = 34 * S * batch * H;
  const boundary = 2 * S * batch * H;
  const stored = (LAYERS - k) * full + k * boundary;
  const rebuild = k > 0 ? k * boundary + full : 0; // layers 0..k-1 stored, then one rebuilt
  return Math.max(stored, rebuild);
}

export default function BudgetBar() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [batch, setBatch] = useState(12);
  const [k, setK] = useState(7);

  const act = peakActivations(batch, k);
  const total = STATE + act + HEADROOM;
  const fits = total <= CARD;
  const w = Math.max(280, width - 24);
  const scale = w / Math.max(CARD, total);
  const pieces = [
    { label: "Weights, gradients, Adam", bytes: STATE, tone: color.b },
    { label: "Activations at the peak", bytes: act, tone: color.a },
    { label: "Headroom", bytes: HEADROOM, tone: color.faint },
  ];
  let x = 12;

  return (
    <Figure
      title="Does the step fit on an 80 GiB card?"
      caption="State is fixed by the model. Activations scale with the batch and fall as you checkpoint layers 0 to k - 1. At batch 12, 7 checkpointed layers is the fewest that fit under 80 GiB."
      controls={
        <>
          <Slider label="Batch size" min={2} max={16} value={batch} onChange={setBatch} />
          <Slider label="Checkpointed layers" min={0} max={LAYERS} value={k} onChange={setK} />
        </>
      }
      readout={
        <>
          <Stat label="State" value={formatBytes(STATE, 2)} tone={color.b} />
          <Stat label="Activation peak" value={formatBytes(act, 2)} tone={color.a} />
          <Stat label="Total" value={formatBytes(total, 2)} tone={fits ? color.ok : color.bad} />
          <Stat label="Verdict" value={fits ? "fits" : "out of memory"} tone={fits ? color.ok : color.bad} />
        </>
      }
    >
      <div ref={ref}>
        <Legend items={pieces.map((p) => ({ label: p.label, tone: p.tone }))} />
        <svg
          width={Math.max(width, 300)}
          height={64}
          role="img"
          aria-label={`Batch ${batch}, ${k} layers checkpointed: ${formatBytes(total, 2)} against an 80 GiB card.`}
        >
          {pieces.map((p) => {
            const piece = p.bytes * scale;
            const left = x;
            x += piece;
            return <rect key={p.label} x={left} y={12} width={Math.max(0, piece - 1)} height={28} style={{ fill: p.tone }} />;
          })}
          <line x1={12 + CARD * scale} x2={12 + CARD * scale} y1={4} y2={48} strokeWidth={2} style={{ stroke: color.bad }} />
          <text x={12 + CARD * scale} y={60} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
            80 GiB
          </text>
        </svg>
      </div>
    </Figure>
  );
}
