import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Halo cost of a tiled convolution. The numbers come from
// scripts/figures/31-convolution-and-stencils.py, section 1; the formulas are
//   output-tile loads per output  = ((T + k - 1) / T) ^ d
//   input-tile active fraction    = ((B - k + 1) / B) ^ d, or 0 when k > B
// for a tile or block of T or B per side, a filter of k per side, and d
// dimensions.

const KS = [1, 3, 5, 7, 9, 11, 13, 15] as const;
const OUT_TILES = [4, 8, 16, 32] as const;
const BLOCKS = [8, 16, 32] as const;

const LOADS: Record<2 | 3, Record<number, number[]>> = {
  2: {
    4: [1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25],
    8: [1.0, 1.5625, 2.25, 3.0625, 4.0, 5.0625, 6.25, 7.5625],
    16: [1.0, 1.2656, 1.5625, 1.8906, 2.25, 2.6406, 3.0625, 3.5156],
    32: [1.0, 1.1289, 1.2656, 1.4102, 1.5625, 1.7227, 1.8906, 2.0664],
  },
  3: {
    4: [1.0, 3.375, 8.0, 15.625, 27.0, 42.875, 64.0, 91.125],
    8: [1.0, 1.9531, 3.375, 5.3594, 8.0, 11.3906, 15.625, 20.7969],
    16: [1.0, 1.4238, 1.9531, 2.5996, 3.375, 4.291, 5.3594, 6.5918],
    32: [1.0, 1.1995, 1.4238, 1.6746, 1.9531, 2.261, 2.5996, 2.9705],
  },
};

const ACTIVE: Record<2 | 3, Record<number, number[]>> = {
  2: {
    8: [1.0, 0.5625, 0.25, 0.0625, 0.0, 0.0, 0.0, 0.0],
    16: [1.0, 0.7656, 0.5625, 0.3906, 0.25, 0.1406, 0.0625, 0.0156],
    32: [1.0, 0.8789, 0.7656, 0.6602, 0.5625, 0.4727, 0.3906, 0.3164],
  },
  3: {
    8: [1.0, 0.4219, 0.125, 0.0156, 0.0, 0.0, 0.0, 0.0],
    16: [1.0, 0.6699, 0.4219, 0.2441, 0.125, 0.0527, 0.0156, 0.002],
    32: [1.0, 0.824, 0.6699, 0.5364, 0.4219, 0.325, 0.2441, 0.178],
  },
};

function Bars({
  width,
  labels,
  values,
  max,
  tone,
  format,
  prefix,
}: {
  width: number;
  labels: readonly number[];
  values: number[];
  max: number;
  tone: string;
  format: (v: number) => string;
  prefix: string;
}) {
  const h = 110;
  const top = 18;
  const gap = 12;
  const bw = Math.max(24, (width - 24 - gap * (labels.length - 1)) / labels.length);
  return (
    <g>
      {labels.map((label, i) => {
        const bh = Math.max(1, (values[i] / max) * h);
        const x = 12 + i * (bw + gap);
        return (
          <g key={label}>
            <rect x={x} y={top + h - bh} width={bw} height={bh} rx={3} style={{ fill: wash(tone, 70) }} />
            <text x={x + bw / 2} y={top + h - bh - 4} fontSize={11} textAnchor="middle" style={{ fill: color.fg }}>
              {format(values[i])}
            </text>
            <text x={x + bw / 2} y={top + h + 15} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
              {prefix} {label}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export default function HaloOverhead() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [dims, setDims] = useState<2 | 3>(2);
  const [k, setK] = useState(5);
  const idx = KS.indexOf(k as (typeof KS)[number]);
  const loads = OUT_TILES.map((t) => LOADS[dims][t][idx]);
  const active = BLOCKS.map((b) => ACTIVE[dims][b][idx]);
  const loadMax = dims === 2 ? 21 : 92;
  const half = Math.max(240, width);

  return (
    <Figure
      title="Halo cost of a tiled convolution"
      caption="Top: the output-tile strategy loads (T + k - 1) to the power d elements to write T to the power d outputs. Bottom: the input-tile strategy loads one element per thread, and only the inner (B - k + 1) to the power d threads write. Small tiles and big filters pay most. A block of 8 can't hold a 9-wide filter."
      controls={
        <>
          <Segmented
            label="Dimensions"
            value={dims}
            options={[
              { value: 2, label: "2D" },
              { value: 3, label: "3D" },
            ]}
            onChange={(v) => setDims(v as 2 | 3)}
          />
          <Slider label="Filter side k" values={KS} value={k} onChange={setK} />
        </>
      }
      readout={
        <>
          <Stat label="Loads per output, tile 16" value={loads[2].toFixed(3)} tone={color.a} />
          <Stat label="Threads that write, block 16" value={`${(active[1] * 100).toFixed(1)}%`} tone={color.b} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={half}
          height={290}
          role="img"
          aria-label={`${dims}D, filter side ${k}. Output-tile loads per output for tiles 4, 8, 16, 32: ${loads
            .map((v) => v.toFixed(2))
            .join(", ")}. Input-tile writing fraction for blocks 8, 16, 32: ${active.map((v) => v.toFixed(2)).join(", ")}.`}
        >
          <text x={12} y={12} fontSize={12} style={{ fill: color.muted }}>
            Output-tile strategy: input elements loaded per output
          </text>
          <Bars width={half} labels={OUT_TILES} values={loads} max={loadMax} tone={color.a} format={(v) => v.toFixed(2)} prefix="T=" />
          <g transform="translate(0,150)">
            <text x={12} y={12} fontSize={12} style={{ fill: color.muted }}>
              Input-tile strategy: fraction of threads that write
            </text>
            <Bars width={half} labels={BLOCKS} values={active} max={1} tone={color.b} format={(v) => `${(v * 100).toFixed(0)}%`} prefix="B=" />
          </g>
        </svg>
        <Legend
          items={[
            { label: "Loads per output (lower is better)", tone: color.a },
            { label: "Active threads (higher is better)", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
