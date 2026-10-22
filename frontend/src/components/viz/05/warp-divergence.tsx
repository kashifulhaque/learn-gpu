import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// A warp is 32 lanes (docs/FACTS.md 4.1). When lanes of one warp take different
// branch paths, the warp runs each path in turn with the other lanes masked off.
// The path length is the number of instructions each path executes; the lab's
// kernels use 256. This figure is a model of issue slots, not a measurement.
const LANES = 32;
const LENGTHS = [16, 64, 256] as const;

type Pattern = "uniform" | "alternate" | "halves" | "one";

const PATTERNS: Record<Pattern, { label: string; path: (lane: number) => number }> = {
  uniform: { label: "Warp-uniform", path: () => 0 },
  alternate: { label: "Alternating lanes", path: (l) => l % 2 },
  halves: { label: "Lanes 0-15 vs 16-31", path: (l) => (l < 16 ? 0 : 1) },
  one: { label: "One lane differs", path: (l) => (l === 0 ? 1 : 0) },
};

const TONES = [color.a, color.b];
const NAMES = ["path A", "path B"];

export default function WarpDivergence() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [pattern, setPattern] = useState<Pattern>("alternate");
  const [len, setLen] = useState<number>(256);

  const paths = Array.from({ length: LANES }, (_, l) => PATTERNS[pattern].path(l));
  const used = [0, 1].filter((p) => paths.includes(p));
  const passes = used.length;
  const issued = passes * len;
  const utilization = 1 / passes;

  const lx = 70;
  const pw = Math.max(200, width - lx - 16);
  const cell = pw / LANES;
  const rowH = 26;
  const rowGap = 8;
  const rows = 1 + passes;
  const h = 24 + rows * (rowH + rowGap) + 18;

  return (
    <Figure
      title="Divergence inside a warp"
      caption="The top row colors each lane by the path it takes. Below it, one row per pass: a filled cell is an active lane, an empty one is masked off. A warp pays for every path any of its lanes takes."
      controls={
        <>
          <Segmented
            label="Branch pattern"
            value={pattern}
            onChange={setPattern}
            options={(Object.keys(PATTERNS) as Pattern[]).map((k) => ({ value: k, label: PATTERNS[k].label }))}
          />
          <Slider label="Path length" values={LENGTHS} value={len} onChange={setLen} format={(v) => `${v} instr`} />
        </>
      }
      readout={
        <>
          <Stat label="Passes" value={passes} tone={color.c} />
          <Stat label="Instructions issued" value={issued} />
          <Stat label="Lanes active per pass" value={`${(utilization * 100).toFixed(0)}%`} tone={color.a} />
        </>
      }
    >
      <div ref={ref}>
        <svg width={width} height={h} role="img" aria-label={`A warp of 32 lanes with the ${PATTERNS[pattern].label} branch pattern runs ${passes} pass${passes > 1 ? "es" : ""}`}>
          <text x={lx - 8} y={24 + rowH / 2 + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            lane path
          </text>
          {paths.map((p, l) => (
            <rect key={l} x={lx + l * cell + 1} y={24} width={cell - 2} height={rowH} rx={2} style={{ fill: wash(TONES[p], 55), stroke: TONES[p] }} />
          ))}
          {used.map((p, r) => {
            const y = 24 + (r + 1) * (rowH + rowGap);
            return (
              <g key={p}>
                <text x={lx - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                  pass {r + 1}: {NAMES[p]}
                </text>
                {paths.map((q, l) => (
                  <rect
                    key={l}
                    x={lx + l * cell + 1}
                    y={y}
                    width={cell - 2}
                    height={rowH}
                    rx={2}
                    style={q === p ? { fill: wash(TONES[p], 55), stroke: TONES[p] } : { fill: color.well, stroke: color.line }}
                  />
                ))}
              </g>
            );
          })}
          <text x={lx} y={h - 4} fontSize={11} style={{ fill: color.muted }}>
            32 lanes, {passes} pass{passes > 1 ? "es" : ""} of {len} instructions
          </text>
        </svg>
      </div>
    </Figure>
  );
}
