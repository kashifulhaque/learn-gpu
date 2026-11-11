import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Allowed final values of (r0, r1) for three litmus tests, from the lab's
// enumerator in content/labs/39-memory-model/solution.py.
// scripts/figures/39-memory-consistency-and-atomics.py prints these sets.
// "sc" is sequential consistency. The other three run the weak model, in which
// accesses to different addresses may reorder unless an acquire load, a
// release store, or a fence stops them. A model, not a measurement.

type Test = "MP" | "SB" | "LB";
type Variant = "sc" | "relaxed" | "ra" | "fence";
type Pair = readonly [number, number];

const ALL: Pair[] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const ALLOWED: Record<Test, Record<Variant, Pair[]>> = {
  MP: {
    sc: [[0, 0], [0, 1], [1, 1]],
    relaxed: ALL,
    ra: [[0, 0], [0, 1], [1, 1]],
    fence: [[0, 0], [0, 1], [1, 1]],
  },
  SB: {
    sc: [[0, 1], [1, 0], [1, 1]],
    relaxed: ALL,
    ra: ALL,
    fence: [[0, 1], [1, 0], [1, 1]],
  },
  LB: {
    sc: [[0, 0], [0, 1], [1, 0]],
    relaxed: ALL,
    ra: [[0, 0], [0, 1], [1, 0]],
    fence: [[0, 0], [0, 1], [1, 0]],
  },
};

// The outcome sequential consistency forbids, the one the test hunts for.
const WEAK: Record<Test, Pair> = { MP: [1, 0], SB: [0, 0], LB: [1, 1] };

const PROGRAMS: Record<Test, { t0: string[]; t1: string[] }> = {
  MP: {
    t0: ["x = 1", "y = 1   (flag)"],
    t1: ["r0 = y   (flag)", "r1 = x"],
  },
  SB: {
    t0: ["x = 1", "r0 = y"],
    t1: ["y = 1", "r1 = x"],
  },
  LB: {
    t0: ["r0 = x", "y = 1"],
    t1: ["r1 = y", "x = 1"],
  },
};

const VARIANT_NOTE: Record<Variant, string> = {
  sc: "Every access stays in program order.",
  relaxed: "Every access is relaxed. Different addresses may reorder.",
  ra: "Stores are release, loads are acquire.",
  fence: "A fence between the two accesses of each thread.",
};

const same = (a: Pair, b: Pair) => a[0] === b[0] && a[1] === b[1];

export default function Litmus() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [test, setTest] = useState<Test>("MP");
  const [variant, setVariant] = useState<Variant>("relaxed");

  const allowed = ALLOWED[test][variant];
  const weak = WEAK[test];
  const weakAllowed = allowed.some((o) => same(o, weak));
  const cell = Math.max(64, Math.min(96, Math.floor((width - 120) / 2)));
  const gx = 70;
  const gy = 40;
  const prog = PROGRAMS[test];

  return (
    <Figure
      title="Litmus tests and the outcomes a model allows"
      caption="Each square is a pair of final register values. The amber square is the outcome that sequential consistency forbids. Release and acquire fix message passing and load buffering, but only a fence fixes store buffering."
      controls={
        <>
          <Segmented
            label="Test"
            value={test}
            onChange={setTest}
            options={[
              { value: "MP", label: "MP" },
              { value: "SB", label: "SB" },
              { value: "LB", label: "LB" },
            ]}
          />
          <Segmented
            label="Model"
            value={variant}
            onChange={setVariant}
            options={[
              { value: "sc", label: "sc" },
              { value: "relaxed", label: "relaxed" },
              { value: "ra", label: "rel/acq" },
              { value: "fence", label: "fence" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Allowed outcomes" value={`${allowed.length} of 4`} />
          <Stat
            label={`r0 = ${weak[0]}, r1 = ${weak[1]}`}
            value={weakAllowed ? "allowed" : "forbidden"}
            tone={weakAllowed ? color.warn : color.ok}
          />
        </>
      }
    >
      <div ref={ref}>
        <p style={{ margin: "0 0 8px", color: "var(--fg-muted)", fontSize: 13 }}>{VARIANT_NOTE[variant]}</p>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", fontFamily: "var(--font-mono, monospace)", fontSize: 12 }}>
          {[prog.t0, prog.t1].map((lines, i) => (
            <div key={i}>
              <div style={{ color: "var(--fg-muted)" }}>Thread {i}</div>
              {lines.map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>
          ))}
        </div>
        <svg
          width={Math.max(width, gx + 2 * cell + 8)}
          height={gy + 2 * cell + 24}
          role="img"
          aria-label={`${test} under ${variant}: ${allowed.length} of 4 outcomes allowed. The outcome r0 ${weak[0]}, r1 ${weak[1]} is ${weakAllowed ? "allowed" : "forbidden"}.`}
        >
          <text x={gx + cell} y={14} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            r1
          </text>
          <text x={gx - 40} y={gy + cell} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            r0
          </text>
          {[0, 1].map((v) => (
            <g key={v}>
              <text x={gx + v * cell + cell / 2} y={32} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                {v}
              </text>
              <text x={gx - 8} y={gy + v * cell + cell / 2 + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {v}
              </text>
            </g>
          ))}
          {ALL.map((o) => {
            const ok = allowed.some((a) => same(a, o));
            const isWeak = same(o, weak);
            const tone = isWeak ? color.warn : color.ok;
            return (
              <g key={`${o[0]}${o[1]}`}>
                <rect
                  x={gx + o[1] * cell}
                  y={gy + o[0] * cell}
                  width={cell - 4}
                  height={cell - 4}
                  rx={6}
                  strokeWidth={isWeak ? 2.5 : 1}
                  style={{
                    fill: ok ? wash(tone, 30) : wash(color.faint, 12),
                    stroke: isWeak ? color.warn : color.lineStrong,
                  }}
                />
                <text
                  x={gx + o[1] * cell + (cell - 4) / 2}
                  y={gy + o[0] * cell + (cell - 4) / 2 + 4}
                  fontSize={13}
                  textAnchor="middle"
                  style={{ fill: ok ? color.fg : color.faint }}
                >
                  {ok ? `(${o[0]}, ${o[1]})` : `(${o[0]}, ${o[1]}) no`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
