import { useState } from "react";
import { Figure, Legend, Segmented, Stat, StepControls, color, formatBytes, useStepper, useWidth, wash } from "../kit";

// Constants. All arithmetic: N tokens, vocabulary V, float32 logits.
//  - V = 151,936 (the Qwen2.5 vocabulary), N = 8192 tokens, 4 bytes per logit.
//  - The chunk size sets how many chunks there are: N / chunk.
const N = 8192;
const V = 151936;
const BYTES = 4;

export default function ChunkedXent() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [chunk, setChunk] = useState<number>(2048);
  const chunks = N / chunk;
  const stepper = useStepper(chunks + 1, 900);
  // step 0: nothing computed. step k: chunks 0..k-1 are finished; chunk k-1 is the one just done.
  const k = Math.min(stepper.step, chunks);
  const current = Math.max(0, k - 1);

  const full = N * V * BYTES;
  const live = chunk * V * BYTES;

  const h = 250;
  const colW = Math.max(60, Math.min(120, (width - 40) / 5));
  const gap = (width - 4 * colW) / 5;
  const px = (i: number) => gap + i * (colW + gap);
  const top = 34;
  const colH = 160;
  const rowH = colH / chunks;

  return (
    <Figure
      title="One chunk of logits at a time"
      caption="Step through the loop. Each step builds one chunk of logits, overwrites it with its gradient, uses it to write its rows of the hidden-state gradient, and adds to the weight gradient. Only the highlighted chunk of logits exists."
      controls={
        <>
          <Segmented
            label="Chunk"
            value={chunk}
            onChange={(c) => {
              setChunk(c);
              stepper.reset();
            }}
            options={[
              { value: 1024, label: "1024 rows" },
              { value: 2048, label: "2048 rows" },
              { value: 4096, label: "4096 rows" },
            ]}
          />
          <StepControls
            stepper={stepper}
            count={chunks + 1}
            describe={(s) => (s === 0 ? "nothing computed yet" : `chunk ${s} of ${chunks} done`)}
          />
        </>
      }
      readout={
        <>
          <Stat label="Chunks" value={`${k} of ${chunks}`} />
          <Stat label="Logits alive" value={formatBytes(live)} tone={color.a} />
          <Stat label="Full logits" value={formatBytes(full)} tone={color.bad} />
          <Stat label="Ratio" value={`${chunks}x`} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label={`Chunk ${k} of ${chunks} processed: hidden states on the left, one live chunk of logits in the middle, the hidden-state gradient filling in on the right`}>
          {[
            { title: "h  [N, d]", i: 0 },
            { title: "logits  [N, V]", i: 1 },
            { title: "grad_h  [N, d]", i: 2 },
            { title: "grad_w  [V, d]", i: 3 },
          ].map((c) => (
            <text key={c.title} x={px(c.i) + colW / 2} y={20} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              {c.title}
            </text>
          ))}
          {Array.from({ length: chunks }, (_, r) => {
            const done = r < k;
            const live_ = k > 0 && r === current;
            const y = top + r * rowH;
            return (
              <g key={r}>
                <rect x={px(0)} y={y + 1} width={colW} height={rowH - 2} rx={3} style={{ fill: live_ ? wash(color.b, 40) : wash(color.b, 14), stroke: color.b }} strokeWidth={live_ ? 2 : 1} />
                <rect
                  x={px(1)}
                  y={y + 1}
                  width={colW}
                  height={rowH - 2}
                  rx={3}
                  strokeDasharray={live_ ? undefined : "3 3"}
                  strokeWidth={live_ ? 2 : 1}
                  style={{ fill: live_ ? wash(color.a, 45) : "transparent", stroke: live_ ? color.a : color.faint }}
                />
                <rect x={px(2)} y={y + 1} width={colW} height={rowH - 2} rx={3} style={{ fill: done ? wash(color.c, 40) : "transparent", stroke: done ? color.c : color.faint }} strokeWidth={1} strokeDasharray={done ? undefined : "3 3"} />
              </g>
            );
          })}
          <rect x={px(3)} y={top} width={colW} height={colH} rx={3} style={{ fill: wash(color.d, k === 0 ? 4 : 12 + (60 * k) / chunks), stroke: color.d }} strokeWidth={1.5} />
          <text x={px(3) + colW / 2} y={top + colH / 2 + 4} textAnchor="middle" fontSize={11} style={{ fill: color.fg }}>
            {k} of {chunks} terms
          </text>
          <text x={px(1) + colW / 2} y={top + colH + 18} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            dashed: never allocated
          </text>
          <text x={px(2) + colW / 2} y={top + colH + 18} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            rows written
          </text>
          <text x={px(3) + colW / 2} y={top + colH + 18} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
            summed over chunks
          </text>
        </svg>
        <Legend
          items={[
            { label: "Hidden states", tone: color.b },
            { label: "Live logits chunk", tone: color.a },
            { label: "grad_h rows", tone: color.c },
            { label: "grad_w accumulator", tone: color.d },
          ]}
        />
      </div>
    </Figure>
  );
}
