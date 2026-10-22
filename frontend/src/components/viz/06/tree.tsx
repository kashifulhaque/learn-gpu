import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants. The warp is 32 threads and shared memory has 32 banks of 4 bytes
// (docs/FACTS.md sections 3.2 and 4.1). The three schemes are the first three
// kernels of Harris, "Optimizing Parallel Reduction in CUDA" (2007).
const WARP = 32;
const BANKS = 32;

type Scheme = "divergent" | "strided" | "sequential";

const SCHEMES: readonly { value: Scheme; label: string }[] = [
  { value: "divergent", label: "Interleaved, tid % 2s" },
  { value: "strided", label: "Interleaved, strided index" },
  { value: "sequential", label: "Sequential" },
];

/** The stride s of round `round` (0-based) for a block of n values. */
function strideOf(scheme: Scheme, n: number, round: number): number {
  return scheme === "sequential" ? n >> (round + 1) : 1 << round;
}

/** Threads that do an add in a round with stride s. */
function activeThreads(scheme: Scheme, n: number, s: number): number[] {
  const out: number[] = [];
  if (scheme === "divergent") {
    for (let t = 0; t < n; t++) if (t % (2 * s) === 0) out.push(t);
  } else if (scheme === "strided") {
    for (let t = 0; t < n / (2 * s); t++) out.push(t);
  } else {
    for (let t = 0; t < s; t++) out.push(t);
  }
  return out;
}

/** The shared-memory word a thread reads first. The second read is at +s. */
function wordOf(scheme: Scheme, t: number, s: number): number {
  return scheme === "strided" ? 2 * s * t : t;
}

/** Worst bank conflict over the warps, for the read at offset `extra` words. */
function conflictDegree(scheme: Scheme, n: number, s: number, active: number[]): number {
  let worst = 1;
  for (let warp = 0; warp < n / WARP; warp++) {
    // The two reads are separate instructions, so count each on its own.
    for (const extra of [0, s]) {
      const per = new Map<number, Set<number>>();
      for (const t of active) {
        if (Math.floor(t / WARP) !== warp) continue;
        const word = wordOf(scheme, t, s) + extra;
        const bank = word % BANKS;
        if (!per.has(bank)) per.set(bank, new Set());
        per.get(bank)!.add(word);
      }
      per.forEach((words) => {
        worst = Math.max(worst, words.size);
      });
    }
  }
  return worst;
}

function warpsWithWork(active: number[]): number {
  return new Set(active.map((t) => Math.floor(t / WARP))).size;
}

export default function TreeRounds() {
  const [scheme, setScheme] = useState<Scheme>("divergent");
  const [n, setN] = useState<number>(256);
  const [round, setRound] = useState(3);
  const [ref, width] = useWidth<HTMLDivElement>();

  const rounds = Math.log2(n);
  const sel = Math.min(round, rounds);
  const warps = n / WARP;

  const rows = Array.from({ length: rounds }, (_, r) => {
    const s = strideOf(scheme, n, r);
    const active = activeThreads(scheme, n, s);
    return { s, active, warps: warpsWithWork(active), conflict: conflictDegree(scheme, n, s, active) };
  });
  const totalIssues = rows.reduce((sum, r) => sum + r.warps, 0);
  const worstConflict = rows.reduce((m, r) => Math.max(m, r.conflict), 1);
  const cur = rows[sel - 1];

  const labelW = 46;
  const rightW = 64;
  const stripW = Math.max(120, width - labelW - rightW);
  const cell = stripW / n;
  const rowH = 16;
  const gap = 4;
  const height = rounds * (rowH + gap) + 6;

  return (
    <Figure
      title="Which threads work in each round"
      caption={
        <>
          Each row is one round of a block-wide tree sum over {n} values, one thread per value. Filled cells do an add.
          Thin lines mark warp boundaries. The interleaved layout keeps every warp busy for most rounds; the sequential
          one retires whole warps.
        </>
      }
      controls={
        <>
          <Segmented label="Addressing" value={scheme} options={SCHEMES} onChange={setScheme} />
          <Segmented
            label="Block size"
            value={n}
            options={[
              { value: 64, label: "64" },
              { value: 256, label: "256" },
            ]}
            onChange={setN}
          />
          <Slider label="Round" value={sel} min={1} max={rounds} onChange={setRound} />
        </>
      }
      readout={
        <>
          <Stat label={`Round ${sel}: stride s`} value={cur.s} tone={color.a} />
          <Stat label="Active threads" value={`${cur.active.length} of ${n}`} />
          <Stat label="Warps with work" value={`${cur.warps} of ${warps}`} />
          <Stat label="Bank conflict" value={`${cur.conflict}-way`} tone={cur.conflict > 1 ? color.bad : color.ok} />
          <Stat label="Warp issues, all rounds" value={totalIssues} tone={color.c} />
          <Stat label="Worst conflict, all rounds" value={`${worstConflict}-way`} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label="Active threads per round of a tree reduction">
          {rows.map((r, ri) => {
            const y = ri * (rowH + gap) + 3;
            const active = new Set(r.active);
            return (
              <g key={ri}>
                <text x={0} y={y + rowH - 4} style={{ fill: ri + 1 === sel ? color.fg : color.muted, fontSize: 11 }}>
                  s = {r.s}
                </text>
                {ri + 1 === sel && (
                  <rect
                    x={labelW - 3}
                    y={y - 2}
                    width={stripW + 6}
                    height={rowH + 4}
                    rx={3}
                    style={{ fill: "none", stroke: color.accent, strokeWidth: 1.5 }}
                  />
                )}
                {Array.from({ length: n }, (_, t) => (
                  <rect
                    key={t}
                    x={labelW + t * cell}
                    y={y}
                    width={Math.max(cell - (cell > 4 ? 1 : 0), 0.8)}
                    height={rowH}
                    style={{ fill: active.has(t) ? color.accent : wash(color.muted, 14) }}
                  />
                ))}
                {Array.from({ length: warps + 1 }, (_, w) => (
                  <line
                    key={w}
                    x1={labelW + w * WARP * cell}
                    x2={labelW + w * WARP * cell}
                    y1={y - 1}
                    y2={y + rowH + 1}
                    style={{ stroke: color.lineStrong, strokeWidth: 1 }}
                  />
                ))}
                <text x={labelW + stripW + 8} y={y + rowH - 4} style={{ fill: color.muted, fontSize: 11 }}>
                  {r.warps}/{warps} warps
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "thread adds", tone: color.accent },
          { label: "thread idle", tone: wash(color.muted, 30) },
        ]}
      />
    </Figure>
  );
}
