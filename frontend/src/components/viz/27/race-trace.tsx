import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// A block of B threads sums B values in shared memory, one tree step per
// phase. Phase 0 writes s[t]. Phase k reads s[t] and s[t + d] for t < d,
// d = B >> k, and writes s[t]. A hazard is a read-after-write pair of
// different threads with no barrier between the two phases.
//
// HAZARDS is the output of scripts/figures/27-debugging-gpu-kernels.py. Each
// entry is [phase, address, writer, reader]. "free" is the model an sm_70 or
// newer GPU guarantees, where the lanes of a warp are ordered only by
// __syncwarp or a barrier. "lock" is the pre-Volta assumption that a warp runs
// in lockstep. The numbers are a trace of the algorithm, not a measurement.

type Pairs = number[][];
type Entry = { free: Pairs; lock: Pairs };
const HAZARDS: Record<string, Record<string, Entry>> = {"8":{"none":{"free":[[1,4,4,0],[1,5,5,1],[1,6,6,2],[1,7,7,3],[2,2,2,0],[2,3,3,1],[3,1,1,0]],"lock":[]},"load":{"free":[[2,2,2,0],[2,3,3,1],[3,1,1,0]],"lock":[]},"every":{"free":[],"lock":[]}},"16":{"none":{"free":[[1,8,8,0],[1,9,9,1],[1,10,10,2],[1,11,11,3],[1,12,12,4],[1,13,13,5],[1,14,14,6],[1,15,15,7],[2,4,4,0],[2,5,5,1],[2,6,6,2],[2,7,7,3],[3,2,2,0],[3,3,3,1],[4,1,1,0]],"lock":[]},"load":{"free":[[2,4,4,0],[2,5,5,1],[2,6,6,2],[2,7,7,3],[3,2,2,0],[3,3,3,1],[4,1,1,0]],"lock":[]},"every":{"free":[],"lock":[]}},"32":{"none":{"free":[[1,16,16,0],[1,17,17,1],[1,18,18,2],[1,19,19,3],[1,20,20,4],[1,21,21,5],[1,22,22,6],[1,23,23,7],[1,24,24,8],[1,25,25,9],[1,26,26,10],[1,27,27,11],[1,28,28,12],[1,29,29,13],[1,30,30,14],[1,31,31,15],[2,8,8,0],[2,9,9,1],[2,10,10,2],[2,11,11,3],[2,12,12,4],[2,13,13,5],[2,14,14,6],[2,15,15,7],[3,4,4,0],[3,5,5,1],[3,6,6,2],[3,7,7,3],[4,2,2,0],[4,3,3,1],[5,1,1,0]],"lock":[]},"load":{"free":[[2,8,8,0],[2,9,9,1],[2,10,10,2],[2,11,11,3],[2,12,12,4],[2,13,13,5],[2,14,14,6],[2,15,15,7],[3,4,4,0],[3,5,5,1],[3,6,6,2],[3,7,7,3],[4,2,2,0],[4,3,3,1],[5,1,1,0]],"lock":[]},"every":{"free":[],"lock":[]}},"64":{"none":{"free":[[1,32,32,0],[1,33,33,1],[1,34,34,2],[1,35,35,3],[1,36,36,4],[1,37,37,5],[1,38,38,6],[1,39,39,7],[1,40,40,8],[1,41,41,9],[1,42,42,10],[1,43,43,11],[1,44,44,12],[1,45,45,13],[1,46,46,14],[1,47,47,15],[1,48,48,16],[1,49,49,17],[1,50,50,18],[1,51,51,19],[1,52,52,20],[1,53,53,21],[1,54,54,22],[1,55,55,23],[1,56,56,24],[1,57,57,25],[1,58,58,26],[1,59,59,27],[1,60,60,28],[1,61,61,29],[1,62,62,30],[1,63,63,31],[2,16,16,0],[2,17,17,1],[2,18,18,2],[2,19,19,3],[2,20,20,4],[2,21,21,5],[2,22,22,6],[2,23,23,7],[2,24,24,8],[2,25,25,9],[2,26,26,10],[2,27,27,11],[2,28,28,12],[2,29,29,13],[2,30,30,14],[2,31,31,15],[3,8,8,0],[3,9,9,1],[3,10,10,2],[3,11,11,3],[3,12,12,4],[3,13,13,5],[3,14,14,6],[3,15,15,7],[4,4,4,0],[4,5,5,1],[4,6,6,2],[4,7,7,3],[5,2,2,0],[5,3,3,1],[6,1,1,0]],"lock":[[1,32,32,0],[1,33,33,1],[1,34,34,2],[1,35,35,3],[1,36,36,4],[1,37,37,5],[1,38,38,6],[1,39,39,7],[1,40,40,8],[1,41,41,9],[1,42,42,10],[1,43,43,11],[1,44,44,12],[1,45,45,13],[1,46,46,14],[1,47,47,15],[1,48,48,16],[1,49,49,17],[1,50,50,18],[1,51,51,19],[1,52,52,20],[1,53,53,21],[1,54,54,22],[1,55,55,23],[1,56,56,24],[1,57,57,25],[1,58,58,26],[1,59,59,27],[1,60,60,28],[1,61,61,29],[1,62,62,30],[1,63,63,31]]},"load":{"free":[[2,16,16,0],[2,17,17,1],[2,18,18,2],[2,19,19,3],[2,20,20,4],[2,21,21,5],[2,22,22,6],[2,23,23,7],[2,24,24,8],[2,25,25,9],[2,26,26,10],[2,27,27,11],[2,28,28,12],[2,29,29,13],[2,30,30,14],[2,31,31,15],[3,8,8,0],[3,9,9,1],[3,10,10,2],[3,11,11,3],[3,12,12,4],[3,13,13,5],[3,14,14,6],[3,15,15,7],[4,4,4,0],[4,5,5,1],[4,6,6,2],[4,7,7,3],[5,2,2,0],[5,3,3,1],[6,1,1,0]],"lock":[]},"every":{"free":[],"lock":[]}}};

const BLOCKS = [8, 16, 32, 64] as const;
type Mode = "none" | "load" | "every";
type Model = "free" | "lock";

function steps(block: number): number {
  return Math.round(Math.log2(block));
}

// barriers[k] sits between phase k and phase k + 1.
function barrierAfter(mode: Mode, k: number): boolean {
  if (mode === "every") return true;
  if (mode === "load") return k === 0;
  return false;
}

export default function RaceTrace() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [block, setBlock] = useState<number>(8);
  const [mode, setMode] = useState<Mode>("none");
  const [model, setModel] = useState<Model>("free");

  const entry = HAZARDS[String(block)][mode];
  const pairs = entry[model];
  const L = steps(block);
  const rows = L + 1;
  const gx = 64;
  const gy = 26;
  const cell = Math.max(6, Math.min(30, Math.floor((width - gx - 12) / block)));
  const rowH = 26;
  const readers = new Set(pairs.map((p) => `${p[0]}:${p[3]}`));
  const writers = new Set(pairs.map((p) => `${p[0] - 1}:${p[2]}`));

  const active = (phase: number, t: number): boolean => (phase === 0 ? true : t < block >> phase);

  return (
    <Figure
      title="Which reads race in a shared-memory tree reduction"
      caption="Each row is one phase and each column one thread. A rose cell reads a value another thread wrote one phase earlier, with no barrier between them. An amber cell is the thread that wrote it. A line between two rows is a barrier. The independent-lane model is what an sm_70 or newer GPU guarantees. The lockstep model hides every hazard inside a warp, which is the assumption that __syncwarp ends."
      controls={
        <>
          <Segmented
            label="Block size"
            value={block}
            options={BLOCKS.map((b) => ({ value: b, label: String(b) }))}
            onChange={setBlock}
          />
          <Segmented
            label="Barriers"
            value={mode}
            options={[
              { value: "none", label: "none" },
              { value: "load", label: "after load" },
              { value: "every", label: "every step" },
            ]}
            onChange={setMode}
          />
          <Segmented
            label="Warp model"
            value={model}
            options={[
              { value: "free", label: "independent lanes" },
              { value: "lock", label: "lockstep" },
            ]}
            onChange={setModel}
          />
        </>
      }
      readout={
        <>
          <Stat label="Hazard pairs" value={String(pairs.length)} tone={pairs.length ? color.d : color.ok} />
          <Stat label="Independent lanes" value={String(entry.free.length)} />
          <Stat label="Lockstep" value={String(entry.lock.length)} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, gx + block * cell + 12)}
          height={gy + rows * rowH + 8}
          role="img"
          aria-label={`Tree reduction of ${block} threads with barriers ${mode}. ${pairs.length} read-after-write hazard pairs under the ${model === "free" ? "independent-lane" : "lockstep"} model.`}
        >
          <text x={gx} y={14} fontSize={11} style={{ fill: color.muted }}>
            thread 0 to {block - 1}
          </text>
          {Array.from({ length: rows }, (_, phase) => (
            <g key={phase}>
              <text x={gx - 8} y={gy + phase * rowH + 16} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {phase === 0 ? "load" : `step ${phase}`}
              </text>
              {Array.from({ length: block }, (_, t) => {
                const isReader = readers.has(`${phase}:${t}`);
                const isWriter = writers.has(`${phase}:${t}`);
                const on = active(phase, t);
                return (
                  <rect
                    key={t}
                    x={gx + t * cell}
                    y={gy + phase * rowH}
                    width={Math.max(2, cell - 1)}
                    height={rowH - 6}
                    rx={2}
                    style={{
                      fill: isReader ? color.d : isWriter ? color.c : on ? wash(color.faint, 45) : wash(color.faint, 12),
                    }}
                  />
                );
              })}
              {phase < L && barrierAfter(mode, phase) ? (
                <line
                  x1={gx}
                  x2={gx + block * cell}
                  y1={gy + (phase + 1) * rowH - 3}
                  y2={gy + (phase + 1) * rowH - 3}
                  strokeWidth={2}
                  style={{ stroke: color.fg }}
                />
              ) : null}
            </g>
          ))}
        </svg>
      </div>
      <Legend
        items={[
          { label: "reads a racing value", tone: color.d },
          { label: "wrote it", tone: color.c },
          { label: "active thread", tone: wash(color.faint, 45) },
        ]}
      />
    </Figure>
  );
}
