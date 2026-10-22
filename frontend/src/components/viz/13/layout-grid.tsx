import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// A CuTe layout is a function from a coordinate to an offset. This figure
// evaluates it for a 4 x 6 grid of coordinates. The definitions follow the CuTe
// layout algebra document (cute/02_layout_algebra.md): offset = sum of
// coordinate * stride, an integer coordinate into a nested mode is read
// colexicographically (first mode fastest), size = product of the shape,
// cosize = largest offset + 1. Coalesce uses the pairwise rules from the same
// document. Every preset is checked by the lab's solution.py.

type Nest = number | Nest[];

const ROWS = 4;
const COLS = 6;

const prod = (x: Nest): number => (typeof x === "number" ? x : x.reduce<number>((a, b) => a * prod(b), 1));
const flat = (x: Nest): number[] => (typeof x === "number" ? [x] : x.flatMap(flat));
const fmt = (x: Nest): string => (typeof x === "number" ? String(x) : `(${x.map(fmt).join(",")})`);

/** Offset of integer coordinate `c` into one (possibly nested) mode. */
function idx(c: number, shape: Nest, stride: Nest): number {
  if (typeof shape === "number") return c * (stride as number);
  const st = stride as Nest[];
  let total = 0;
  let rest = c;
  shape.forEach((s, k) => {
    if (k === shape.length - 1) {
      total += idx(rest, s, st[k]);
    } else {
      const n = prod(s);
      total += idx(rest % n, s, st[k]);
      rest = Math.floor(rest / n);
    }
  });
  return total;
}

/** The hierarchical coordinate that integer `c` stands for in `shape`. */
function expand(c: number, shape: Nest): Nest {
  if (typeof shape === "number") return c;
  let rest = c;
  return shape.map((s, k) => {
    if (k === shape.length - 1) return expand(rest, s);
    const n = prod(s);
    const out = expand(rest % n, s);
    rest = Math.floor(rest / n);
    return out;
  });
}

function coalesce(shape: Nest, stride: Nest): string {
  const s = flat(shape);
  const d = flat(stride);
  const outS: number[] = [];
  const outD: number[] = [];
  s.forEach((size, k) => {
    if (size === 1) return;
    if (outS.length && outD[outD.length - 1] * outS[outS.length - 1] === d[k]) {
      outS[outS.length - 1] *= size;
    } else {
      outS.push(size);
      outD.push(d[k]);
    }
  });
  if (!outS.length) return "1:0";
  if (outS.length === 1) return `${outS[0]}:${outD[0]}`;
  return `(${outS.join(",")}):(${outD.join(",")})`;
}

type Preset = { key: string; label: string; shape: [Nest, Nest]; stride: [Nest, Nest] };

const PRESETS: Preset[] = [
  { key: "col", label: "Column-major", shape: [ROWS, COLS], stride: [1, ROWS] },
  { key: "row", label: "Row-major", shape: [ROWS, COLS], stride: [COLS, 1] },
  { key: "pad", label: "Padded", shape: [ROWS, COLS], stride: [1, ROWS + 1] },
  { key: "tile", label: "Tiled", shape: [[2, 2], [3, 2]], stride: [[1, 6], [2, 12]] },
  { key: "bcast", label: "Broadcast", shape: [ROWS, COLS], stride: [1, 0] },
  { key: "custom", label: "Custom", shape: [ROWS, COLS], stride: [1, ROWS] },
];

export default function LayoutGrid() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [key, setKey] = useState("tile");
  const [d0, setD0] = useState(2);
  const [d1, setD1] = useState(5);
  const [sel, setSel] = useState<[number, number]>([2, 4]);

  const preset = PRESETS.find((p) => p.key === key) ?? PRESETS[0];
  const shape = preset.shape;
  const stride: [Nest, Nest] = key === "custom" ? [d0, d1] : preset.stride;

  const offsets: number[][] = [];
  const seen = new Map<number, number>();
  for (let i = 0; i < ROWS; i++) {
    const row: number[] = [];
    for (let j = 0; j < COLS; j++) {
      const o = idx(i, shape[0], stride[0]) + idx(j, shape[1], stride[1]);
      row.push(o);
      seen.set(o, (seen.get(o) ?? 0) + 1);
    }
    offsets.push(row);
  }
  const maxOff = Math.max(...offsets.flat());
  const cosize = maxOff + 1;
  const injective = seen.size === ROWS * COLS;

  const gap = 3;
  const cell = Math.max(30, Math.min(58, Math.floor((width - 8 - gap * (COLS - 1)) / COLS)));
  const w = COLS * cell + (COLS - 1) * gap + 8;
  const h = ROWS * cell + (ROWS - 1) * gap + 8;
  const [si, sj] = sel;
  const shown = offsets[si][sj];
  const ci = expand(si, shape[0]);
  const cj = expand(sj, shape[1]);

  return (
    <Figure
      title="A layout is a function from coordinates to offsets"
      caption="Each cell shows the offset that (row, column) maps to. Click a cell to see its hierarchical coordinate. Offsets that appear twice mean the layout is not injective; the Custom preset lets you set both strides."
      controls={
        <>
          <Segmented
            label="Layout"
            value={key}
            onChange={setKey}
            options={PRESETS.map((p) => ({ value: p.key, label: p.label }))}
          />
          {key === "custom" && (
            <>
              <Slider label="Row stride" min={0} max={12} value={d0} onChange={setD0} />
              <Slider label="Column stride" min={0} max={12} value={d1} onChange={setD1} />
            </>
          )}
        </>
      }
      readout={
        <>
          <Stat label="Layout" value={`${fmt(shape)}:${fmt(stride)}`} tone={color.a} />
          <Stat label="size" value={ROWS * COLS} />
          <Stat label="cosize" value={cosize} tone={color.b} />
          <Stat label="Coalesced" value={coalesce(shape, stride)} tone={color.c} />
          <Stat label="Injective" value={injective ? "yes" : "no"} tone={injective ? color.ok : color.bad} />
          <Stat label={`Cell (${si},${sj})`} value={`${fmt(ci)}, ${fmt(cj)} → ${shown}`} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(w, 120)}
          height={h}
          role="img"
          aria-label={`A ${ROWS} by ${COLS} grid of the offsets that the layout ${fmt(shape)}:${fmt(stride)} produces`}
        >
          {offsets.map((row, i) =>
            row.map((o, j) => {
              const x = 4 + j * (cell + gap);
              const y = 4 + i * (cell + gap);
              const dup = (seen.get(o) ?? 0) > 1;
              const on = i === si && j === sj;
              return (
                <g key={`${i}-${j}`} onClick={() => setSel([i, j])} style={{ cursor: "pointer" }}>
                  <rect
                    x={x}
                    y={y}
                    width={cell}
                    height={cell}
                    rx={4}
                    strokeWidth={on ? 2.5 : 1}
                    style={{
                      fill: wash(dup ? color.bad : color.a, 12 + (60 * o) / Math.max(1, maxOff)),
                      stroke: on ? color.accent : color.line,
                    }}
                  />
                  <text
                    x={x + cell / 2}
                    y={y + cell / 2 + 5}
                    textAnchor="middle"
                    fontSize={cell < 40 ? 12 : 15}
                    fontWeight={on ? 700 : 500}
                    style={{ fill: color.fg }}
                  >
                    {o}
                  </text>
                </g>
              );
            }),
          )}
        </svg>
      </div>
    </Figure>
  );
}
