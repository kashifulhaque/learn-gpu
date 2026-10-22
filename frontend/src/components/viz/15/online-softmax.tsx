import { useEffect, useMemo, useState } from "react";
import { Figure, Legend, Segmented, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

// One row of twelve logits, chosen by hand so that the second block raises the
// running maximum and forces a rescale. Nothing here is a measurement.
const LOGITS = [1.2, 0.4, 2.0, -0.5, 3.1, 0.9, 1.7, 4.3, -1.0, 2.6, 0.2, 3.8];
const HEIGHT = 200;
const PAD = { left: 40, right: 12, top: 14, bottom: 26 };
const Y_MIN = -1.5;
const Y_MAX = 5;

type State = { m: number; d: number; factor: number | null; seen: number };

/** The (m, d) pair after each block, following the Milakov and Gimelshein recurrence. */
function trace(block: number): State[] {
  const states: State[] = [{ m: -Infinity, d: 0, factor: null, seen: 0 }];
  let m = -Infinity;
  let d = 0;
  for (let start = 0; start < LOGITS.length; start += block) {
    const chunk = LOGITS.slice(start, start + block);
    const mNew = Math.max(m, ...chunk);
    const factor = Number.isFinite(m) ? Math.exp(m - mNew) : 0;
    d = d * factor + chunk.reduce((sum, x) => sum + Math.exp(x - mNew), 0);
    m = mNew;
    states.push({ m, d, factor: Number.isFinite(factor) && start > 0 ? factor : null, seen: start + chunk.length });
  }
  return states;
}

export default function OnlineSoftmax() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [block, setBlock] = useState<number>(4);
  const states = useMemo(() => trace(block), [block]);
  const stepper = useStepper(states.length, 900);
  const { setStep } = stepper;

  // Start where the first rescale happens, so the printed figure shows it.
  useEffect(() => {
    setStep(2);
  }, [block, setStep]);

  const state = states[Math.min(stepper.step, states.length - 1)];
  const final = states[states.length - 1];
  const exact = LOGITS.reduce((sum, x) => sum + Math.exp(x - final.m), 0);

  const plotW = Math.max(width - PAD.left - PAD.right, 120);
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const slot = plotW / LOGITS.length;
  const y = (v: number) => PAD.top + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * plotH;
  const prevSeen = stepper.step > 0 ? states[stepper.step - 1].seen : 0;

  return (
    <Figure
      title="Online softmax over blocks"
      controls={
        <>
          <Segmented
            label="Block size"
            value={block}
            options={[
              { value: 3, label: "3" },
              { value: 4, label: "4" },
              { value: 6, label: "6" },
            ]}
            onChange={setBlock}
          />
          <StepControls
            stepper={stepper}
            count={states.length}
            describe={(s) => (s === 0 ? "nothing read yet" : `columns 1 to ${states[s].seen}`)}
          />
        </>
      }
      readout={
        <>
          <Stat label="Running max m" value={Number.isFinite(state.m) ? state.m.toFixed(1) : "none"} tone={color.c} />
          <Stat label="Denominator d" value={state.d.toFixed(3)} tone={color.a} />
          <Stat
            label="Rescale exp(m_old - m_new)"
            value={state.factor === null ? "1 (no change)" : state.factor.toFixed(3)}
            tone={color.b}
          />
          <Stat label="Final d, from the row" value={exact.toFixed(3)} />
        </>
      }
      caption="Each block multiplies the old denominator by exp(m_old - m_new) before it adds its own terms. When the last block is read, d equals the sum of exp(x - m) over the whole row."
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={HEIGHT} role="img" aria-label="Twelve logits as bars, with the running maximum drawn as a dashed line">
          <line x1={PAD.left} x2={width - PAD.right} y1={y(0)} y2={y(0)} style={{ stroke: color.line }} />
          {[-1, 0, 2, 4].map((t) => (
            <text key={t} x={PAD.left - 6} y={y(t) + 4} textAnchor="end" fontSize={10} style={{ fill: color.subtle }}>
              {t}
            </text>
          ))}
          {LOGITS.map((v, i) => {
            const seen = i < state.seen;
            const current = seen && i >= prevSeen && stepper.step > 0;
            const fill = current ? color.b : seen ? color.a : color.faint;
            const x0 = PAD.left + i * slot + slot * 0.15;
            const top = Math.min(y(v), y(0));
            const h = Math.abs(y(v) - y(0));
            return (
              <g key={i}>
                <rect x={x0} y={top} width={slot * 0.7} height={Math.max(h, 1)} rx={2} style={{ fill: wash(fill, seen ? 80 : 45) }} />
                <text x={x0 + slot * 0.35} y={HEIGHT - 8} textAnchor="middle" fontSize={10} style={{ fill: color.subtle }}>
                  {i + 1}
                </text>
                <text x={x0 + slot * 0.35} y={v >= 0 ? top - 3 : top + h + 11} textAnchor="middle" fontSize={9} style={{ fill: color.muted }}>
                  {v}
                </text>
              </g>
            );
          })}
          {Number.isFinite(state.m) && (
            <g>
              <line x1={PAD.left} x2={width - PAD.right} y1={y(state.m)} y2={y(state.m)} strokeDasharray="5 4" strokeWidth={2} style={{ stroke: color.c }} />
              <text x={width - PAD.right - 2} y={y(state.m) - 5} textAnchor="end" fontSize={11} style={{ fill: color.c }}>
                m = {state.m.toFixed(1)}
              </text>
            </g>
          )}
          {/* Block boundaries */}
          {LOGITS.map((_, i) =>
            i > 0 && i % block === 0 ? (
              <line key={i} x1={PAD.left + i * slot} x2={PAD.left + i * slot} y1={PAD.top} y2={HEIGHT - PAD.bottom} strokeDasharray="2 3" style={{ stroke: color.line }} />
            ) : null,
          )}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Block just read", tone: color.b },
          { label: "Read earlier", tone: color.a },
          { label: "Not read yet", tone: color.faint },
          { label: "Running maximum", tone: color.c, dashed: true },
        ]}
      />
    </Figure>
  );
}
