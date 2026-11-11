import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth } from "../kit";

// log10 of the cumulative product of a constant per-step decay g, and where the
// naive cumprod-ratio decay matrix breaks. CURVE holds the points at positions 8, 16,
// ..., 512 from scripts/figures/37-chunked-scans.py, and BREAKS is its UNDERFLOW
// table: the first position at which torch.cumprod in that dtype reaches zero, so that
// cumprod[i] / cumprod[j] gives 0 / 0 = NaN. float32 flushes below 1.4e-45 (log10 -44.9)
// and float64 below 4.9e-324 (log10 -323.3).

type Break = {
  g: number;
  f32_zero_at: number | null;
  f32_nan_chunk: number | null;
  f64_zero_at: number | null;
  f64_nan_chunk: number | null;
};
const BREAKS: readonly Break[] = [{"g": 0.1, "f32_zero_at": 46, "f32_nan_chunk": 46, "f64_zero_at": 324, "f64_nan_chunk": 324}, {"g": 0.5, "f32_zero_at": 150, "f32_nan_chunk": 150, "f64_zero_at": 1075, "f64_nan_chunk": 1075}, {"g": 0.9, "f32_zero_at": 987, "f32_nan_chunk": 987, "f64_zero_at": null, "f64_nan_chunk": null}, {"g": 0.99, "f32_zero_at": null, "f32_nan_chunk": null, "f64_zero_at": null, "f64_nan_chunk": null}];
const CURVE: Record<string, readonly number[]> = {"0.1": [-1.0, -9.0, -17.0, -25.0, -33.0, -41.0, -49.0, -57.0, -65.0, -73.0, -81.0, -89.0, -97.0, -105.0, -113.0, -121.0, -129.0, -137.0, -145.0, -153.0, -161.0, -169.0, -177.0, -185.0, -193.0, -201.0, -209.0, -217.0, -225.0, -233.0, -241.0, -249.0, -257.0, -265.0, -273.0, -281.0, -289.0, -297.0, -305.0, -313.0, -321.0, -329.0, -337.0, -345.0, -353.0, -361.0, -369.0, -377.0, -385.0, -393.0, -401.0, -409.0, -417.0, -425.0, -433.0, -441.0, -449.0, -457.0, -465.0, -473.0, -481.0, -489.0, -497.0, -505.0], "0.5": [-0.301, -2.7093, -5.1175, -7.5257, -9.934, -12.3422, -14.7505, -17.1587, -19.5669, -21.9752, -24.3834, -26.7917, -29.1999, -31.6081, -34.0164, -36.4246, -38.8329, -41.2411, -43.6493, -46.0576, -48.4658, -50.8741, -53.2823, -55.6905, -58.0988, -60.507, -62.9153, -65.3235, -67.7317, -70.14, -72.5482, -74.9565, -77.3647, -79.7729, -82.1812, -84.5894, -86.9977, -89.4059, -91.8141, -94.2224, -96.6306, -99.0389, -101.4471, -103.8553, -106.2636, -108.6718, -111.0801, -113.4883, -115.8965, -118.3048, -120.713, -123.1213, -125.5295, -127.9377, -130.346, -132.7542, -135.1625, -137.5707, -139.9789, -142.3872, -144.7954, -147.2037, -149.6119, -152.0201], "0.9": [-0.0458, -0.4118, -0.7779, -1.1439, -1.51, -1.8761, -2.2421, -2.6082, -2.9742, -3.3403, -3.7064, -4.0724, -4.4385, -4.8045, -5.1706, -5.5367, -5.9027, -6.2688, -6.6348, -7.0009, -7.367, -7.733, -8.0991, -8.4651, -8.8312, -9.1973, -9.5633, -9.9294, -10.2954, -10.6615, -11.0276, -11.3936, -11.7597, -12.1257, -12.4918, -12.8579, -13.2239, -13.59, -13.956, -14.3221, -14.6882, -15.0542, -15.4203, -15.7863, -16.1524, -16.5185, -16.8845, -17.2506, -17.6166, -17.9827, -18.3488, -18.7148, -19.0809, -19.4469, -19.813, -20.1791, -20.5451, -20.9112, -21.2772, -21.6433, -22.0094, -22.3754, -22.7415, -23.1075], "0.99": [-0.0044, -0.0393, -0.0742, -0.1091, -0.144, -0.179, -0.2139, -0.2488, -0.2837, -0.3186, -0.3535, -0.3885, -0.4234, -0.4583, -0.4932, -0.5281, -0.5631, -0.598, -0.6329, -0.6678, -0.7027, -0.7377, -0.7726, -0.8075, -0.8424, -0.8773, -0.9122, -0.9472, -0.9821, -1.017, -1.0519, -1.0868, -1.1218, -1.1567, -1.1916, -1.2265, -1.2614, -1.2963, -1.3313, -1.3662, -1.4011, -1.436, -1.4709, -1.5059, -1.5408, -1.5757, -1.6106, -1.6455, -1.6805, -1.7154, -1.7503, -1.7852, -1.8201, -1.855, -1.89, -1.9249, -1.9598, -1.9947, -2.0296, -2.0646, -2.0995, -2.1344, -2.1693, -2.2042]};
const STRIDE = 8;
const LENGTH = 512;
const F32_FLOOR = -44.85;
const F64_FLOOR = -323.31;
const GS = [0.1, 0.5, 0.9, 0.99] as const;
const Y_MIN = -340;

export default function DecayUnderflow() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [g, setG] = useState<number>(0.5);
  const w = Math.max(width, 320);
  const h = 240;
  const pad = { l: 56, r: 14, t: 14, b: 38 };
  const x = (pos: number) => pad.l + (pos / LENGTH) * (w - pad.l - pad.r);
  const y = (v: number) => pad.t + (Math.max(v, Y_MIN) / Y_MIN) * (h - pad.t - pad.b);
  const points: [number, number][] = [[0, 0], ...CURVE[String(g)].map((v, i): [number, number] => [(i + 1) * STRIDE, v])];
  const path = points.map(([p, v], i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const br = BREAKS.find((b) => b.g === g)!;
  const fmt = (v: number | null) => (v === null ? "none by 4096" : `${v}`);

  return (
    <Figure
      title="Where a cumulative decay product underflows"
      caption="The line is log10 of the cumulative product of a constant decay g over 512 steps, clipped at 1e-340. float32 can't hold a product below about 1e-45, and float64 can't below about 5e-324. Past that point the product is 0 and the ratio cumprod[i] / cumprod[j] is 0 / 0, a NaN. Computing the ratio in log space never forms the product."
      controls={
        <Segmented label="Decay g per step" value={g} options={GS.map((v) => ({ value: v, label: String(v) }))} onChange={setG} />
      }
      readout={
        <>
          <Stat label="float32 product is 0 from step" value={fmt(br.f32_zero_at)} tone={color.bad} />
          <Stat label="float64 product is 0 from step" value={fmt(br.f64_zero_at)} tone={color.warn} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`log10 of the cumulative product for g = ${g}. float32 underflows from step ${fmt(br.f32_zero_at)}, float64 from step ${fmt(br.f64_zero_at)}.`}
        >
          {[0, -100, -200, -300].map((v) => (
            <g key={v}>
              <line x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} style={{ stroke: color.line }} />
              <text x={pad.l - 6} y={y(v) + 4} fontSize={10} textAnchor="end" style={{ fill: color.muted }}>
                1e{v}
              </text>
            </g>
          ))}
          <line x1={pad.l} x2={w - pad.r} y1={y(F32_FLOOR)} y2={y(F32_FLOOR)} strokeDasharray="4 3" style={{ stroke: color.bad }} />
          <text x={w - pad.r} y={y(F32_FLOOR) - 4} fontSize={10} textAnchor="end" style={{ fill: color.bad }}>
            float32 floor
          </text>
          <line x1={pad.l} x2={w - pad.r} y1={y(F64_FLOOR)} y2={y(F64_FLOOR)} strokeDasharray="4 3" style={{ stroke: color.warn }} />
          <text x={w - pad.r} y={y(F64_FLOOR) - 4} fontSize={10} textAnchor="end" style={{ fill: color.warn }}>
            float64 floor
          </text>
          <path d={path} fill="none" strokeWidth={2} style={{ stroke: color.a }} />
          {[0, 128, 256, 384, 512].map((p) => (
            <text key={p} x={x(p)} y={h - pad.b + 16} fontSize={10} textAnchor="middle" style={{ fill: color.muted }}>
              {p}
            </text>
          ))}
          <text x={(pad.l + w - pad.r) / 2} y={h - 6} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            position in the chunk
          </text>
        </svg>
        <Legend items={[{ label: `cumprod of g = ${g}`, tone: color.a }]} />
      </div>
    </Figure>
  );
}
