import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Mean relative error ||row - dequantized row|| / ||row|| over the 62 rows (the 64 rows minus the two outliers)
// without an outlier, for a seeded 64 by 512 Gaussian matrix whose rows 3 and
// 17 are multiplied by 10**e. Printed by scripts/figures/33-fp8-and-fp4.py
// with the functions of the chapter 33 lab. Index = e, 0 to 6.
const SCHEMES = [
  { key: "tensor", label: "Per-tensor E4M3", tone: color.bad, err: [0.0263, 0.0265, 0.0264, 0.0266, 0.0481, 0.4758, 1.0] },
  { key: "tile", label: "1x128 E4M3", tone: color.ok, err: [0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256] },
  { key: "mx", label: "MXFP4, blocks of 32", tone: color.info, err: [0.1149, 0.1149, 0.1149, 0.1149, 0.1149, 0.1149, 0.1149] },
  { key: "nv", label: "NVFP4, blocks of 16", tone: color.violet, err: [0.0951, 0.095, 0.0953, 0.0949, 0.095, 0.1406, 1.0] },
] as const;

export default function ScaleOutliers() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [e, setE] = useState(5);
  const [focus, setFocus] = useState<string>("tensor");
  const w = Math.max(width, 320);
  const h = 220;
  const left = 44;
  const right = 12;
  const top = 12;
  const bottom = 30;
  const maxErr = 1;
  const sx = (i: number) => left + (i / 6) * (w - left - right);
  const sy = (v: number) => top + (1 - v / maxErr) * (h - top - bottom);
  const chosen = SCHEMES.find((s) => s.key === focus) ?? SCHEMES[0];

  return (
    <Figure
      title="Relative error as the outlier rows grow"
      caption="Per-tensor E4M3 holds near 2.6% through an outlier of 1,000 times the rest, then the small rows fall into E4M3's subnormal range and the error reaches 48% at 100,000 and 100% at 1,000,000. A scale per 1x128 tile doesn't care. NVFP4's tensor-level scale pushes its block scales into E4M3's subnormals and then to zero at 1,000,000. MXFP4 stays flat at 11.5% because its error is the coarse 4-bit grid, and its power-of-two scales have a range of 2^254."
      controls={
        <>
          <Slider label="Outlier size" min={0} max={6} value={e} onChange={setE} format={(v) => `10^${v}`} />
          <Segmented
            label="Scheme"
            value={focus}
            onChange={setFocus}
            options={SCHEMES.map((s) => ({ value: s.key, label: s.label }))}
          />
        </>
      }
      readout={
        <>
          <Stat label={chosen.label} value={`${(chosen.err[e] * 100).toFixed(1)}%`} tone={chosen.tone} />
          {SCHEMES.filter((s) => s.key !== focus).map((s) => (
            <Stat key={s.key} label={s.label} value={`${(s.err[e] * 100).toFixed(1)}%`} />
          ))}
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Relative error of four scaling schemes with an outlier of 10 to the power ${e}. ${SCHEMES.map((s) => `${s.label} ${(s.err[e] * 100).toFixed(1)} percent`).join(", ")}.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={left} x2={w - right} y1={sy(t)} y2={sy(t)} style={{ stroke: color.line }} />
              <text x={left - 6} y={sy(t) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {[0, 1, 2, 3, 4, 5, 6].map((i) => (
            <text key={i} x={sx(i)} y={h - 10} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
              10^{i}
            </text>
          ))}
          {SCHEMES.map((s) => (
            <polyline
              key={s.key}
              fill="none"
              strokeWidth={s.key === focus ? 3 : 1.5}
              style={{ stroke: s.tone, opacity: s.key === focus ? 1 : 0.6 }}
              points={s.err.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")}
            />
          ))}
          <line x1={sx(e)} x2={sx(e)} y1={top} y2={h - bottom} style={{ stroke: color.lineStrong }} strokeDasharray="4 3" />
          {SCHEMES.map((s) => (
            <circle key={`${s.key}-d`} cx={sx(e)} cy={sy(s.err[e])} r={4} style={{ fill: s.tone }} />
          ))}
        </svg>
        <Legend items={SCHEMES.map((s) => ({ label: s.label, tone: s.tone }))} />
      </div>
    </Figure>
  );
}
