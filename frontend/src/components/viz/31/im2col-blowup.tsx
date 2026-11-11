import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// The im2col matrix against its input, on a 1 by 64 by 56 by 56 fp32 tensor
// with "same" padding (k // 2). The ratios come from
// scripts/figures/31-convolution-and-stencils.py, section 2, where each one is
// checked against the shape of F.unfold: k * k * P * Q / (H * W).

const KS = [1, 3, 5, 7, 9] as const;
const BLOWUP: Record<1 | 2, number[]> = {
  1: [1.0, 9.0, 25.0, 49.0, 81.0],
  2: [0.25, 2.25, 6.25, 12.25, 20.25],
};
const INPUT_BYTES = 64 * 56 * 56 * 4; // 802,816 B

export default function Im2colBlowup() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [stride, setStride] = useState<1 | 2>(1);
  const [k, setK] = useState(3);
  const idx = KS.indexOf(k as (typeof KS)[number]);
  const values = BLOWUP[stride];
  const max = stride === 1 ? 81 : 20.25;
  const w = Math.max(240, width);
  const h = 130;
  const gap = 14;
  const bw = Math.max(28, (w - 24 - gap * (KS.length - 1)) / KS.length);

  return (
    <Figure
      title="Memory blow-up of im2col"
      caption="Each bar is the elements of the lowered matrix divided by the elements of the input. At stride 1 it is k squared. A 3 by 3 filter writes the input nine times over, and a 9 by 9 filter writes it 81 times. Stride 2 divides the count by about four."
      controls={
        <>
          <Segmented
            label="Stride"
            value={stride}
            options={[
              { value: 1, label: "1" },
              { value: 2, label: "2" },
            ]}
            onChange={(v) => setStride(v as 1 | 2)}
          />
          <Slider label="Filter side k" values={KS} value={k} onChange={setK} />
        </>
      }
      readout={
        <>
          <Stat label="Input" value={formatBytes(INPUT_BYTES)} tone={color.b} />
          <Stat label="im2col matrix" value={formatBytes(INPUT_BYTES * values[idx])} tone={color.a} />
          <Stat label="Ratio" value={`${values[idx]}x`} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h + 50}
          role="img"
          aria-label={`Stride ${stride}. The im2col matrix holds ${values.join(", ")} times the input for filter sides 1, 3, 5, 7, 9. Selected: side ${k}, ${values[idx]} times.`}
        >
          {KS.map((kk, i) => {
            const bh = Math.max(1, (values[i] / max) * h);
            const x = 12 + i * (bw + gap);
            const on = i === idx;
            return (
              <g key={kk}>
                <rect x={x} y={20 + h - bh} width={bw} height={bh} rx={3} style={{ fill: on ? color.a : wash(color.a, 35) }} />
                <text x={x + bw / 2} y={20 + h - bh - 4} fontSize={11} textAnchor="middle" style={{ fill: color.fg }}>
                  {values[i]}x
                </text>
                <text x={x + bw / 2} y={20 + h + 16} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                  {kk}x{kk}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
