import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Constants.
//  - The card is the A100 80GB: 80 GB of HBM2e (docs/FACTS.md section 2.1). Decimal gigabytes: 80e9 bytes.
//  - Vocabulary sizes are published model configurations: Llama 3 128,256; Qwen2.5 151,936; Gemma 2 256,000.
//  - Everything else is arithmetic: logits bytes = tokens x vocab x bytes per element.
const CARD_BYTES = 80e9;
const TOKEN_STOPS = [4096, 8192, 16384, 32768, 65536] as const;
const CHUNK_STOPS = [256, 512, 1024, 2048, 4096] as const;
const VOCABS = [
  { value: 128256, label: "128,256" },
  { value: 151936, label: "151,936" },
  { value: 256000, label: "256,000" },
] as const;

export default function LogitsMemory() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tokens, setTokens] = useState<number>(8192);
  const [vocab, setVocab] = useState<number>(151936);
  const [bytes, setBytes] = useState<number>(4);
  const [chunk, setChunk] = useState<number>(1024);

  const full = tokens * vocab * bytes;
  const live = Math.min(chunk, tokens) * vocab * bytes;
  // The reference keeps the logits and a softmax of the same size for the backward pass.
  const reference = 2 * full;

  const h = 150;
  const m = { l: 120, r: 16, t: 12, b: 28 };
  const pw = Math.max(120, width - m.l - m.r);
  const scaleMax = Math.max(CARD_BYTES, reference);
  const x = (b: number) => m.l + (b / scaleMax) * pw;
  const rows = [
    { name: "Unfused: logits + softmax", value: reference, tone: color.bad },
    { name: "Fused: one chunk", value: live, tone: color.a },
  ];

  return (
    <Figure
      title="How big the logits get"
      caption="Logits are tokens times vocabulary times bytes per element. The unfused loss keeps them, and a softmax the same size, for the backward pass. The fused loss keeps one chunk. The card line is the A100's 80 GB, before any weights or activations."
      controls={
        <>
          <Slider label="Tokens per step" values={TOKEN_STOPS} value={tokens} onChange={setTokens} />
          <Slider label="Chunk (rows)" values={CHUNK_STOPS} value={chunk} onChange={setChunk} />
          <Segmented label="Vocabulary" value={vocab} onChange={setVocab} options={VOCABS} />
          <Segmented
            label="Logits dtype"
            value={bytes}
            onChange={setBytes}
            options={[
              { value: 2, label: "bf16 (2 B)" },
              { value: 4, label: "fp32 (4 B)" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Full logits" value={formatBytes(full)} tone={color.b} />
          <Stat label="Unfused, with softmax" value={formatBytes(reference)} tone={color.bad} />
          <Stat label="Fused, one chunk" value={formatBytes(live)} tone={color.a} />
          <Stat label="Ratio" value={`${(reference / live).toFixed(1)}x`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label="Bars comparing the unfused logits memory with one fused chunk, against the 80 GB card">
          {rows.map((r, i) => {
            const y = m.t + i * 44;
            return (
              <g key={r.name}>
                <text x={m.l - 8} y={y + 20} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                  {r.name}
                </text>
                <rect x={m.l} y={y} width={Math.max(2, x(r.value) - m.l)} height={28} rx={3} style={{ fill: wash(r.tone, 35), stroke: r.tone }} strokeWidth={1.5} />
                <text x={Math.min(m.l + pw - 4, x(r.value) + 6)} y={y + 19} fontSize={11} textAnchor={x(r.value) + 70 > m.l + pw ? "end" : "start"} style={{ fill: color.fg }}>
                  {formatBytes(r.value)}
                </text>
              </g>
            );
          })}
          <line x1={x(CARD_BYTES)} x2={x(CARD_BYTES)} y1={m.t - 4} y2={m.t + 92} strokeDasharray="5 4" style={{ stroke: color.warn }} />
          <text x={x(CARD_BYTES) - 4} y={h - 10} textAnchor="end" fontSize={11} style={{ fill: color.warn }}>
            80 GB card
          </text>
          <line x1={m.l} x2={m.l} y1={m.t - 4} y2={m.t + 92} style={{ stroke: color.lineStrong }} />
        </svg>
        <Legend
          items={[
            { label: "Unfused (logits + softmax)", tone: color.bad },
            { label: "Fused (one chunk)", tone: color.a },
            { label: "Card memory", tone: color.warn, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
