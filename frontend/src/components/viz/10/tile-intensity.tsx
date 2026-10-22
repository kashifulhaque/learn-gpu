import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants (docs/FACTS.md section 2):
//  - FP32 peak on the CUDA cores, 19.5 TFLOP/s [A100-DS].
//  - 1935 GB/s rated HBM2e; 1275 GB/s measured device-to-device copy (learn-inference chapter 10).
// The intensity of a BM x BN output tile is the chapter's arithmetic: one K step loads BM + BN floats
// (4 bytes each) and does BM * BN FMAs (2 FLOP each), so I = 2 BM BN / (4 (BM + BN)).
const PEAK_TFLOPS = 19.5;
const BANDWIDTHS = { measured: 1275, rated: 1935 } as const;
type Ceiling = keyof typeof BANDWIDTHS;
const TILE_EDGES = [8, 16, 32, 64, 128, 256] as const;
const TILE_CHOICES = [8, 16, 32, 64, 128, 256] as const;

const intensity = (bm: number, bn: number) => (2 * bm * bn) / (4 * (bm + bn));
// GB/s times FLOP/byte is GFLOP/s; divide by 1000 for TFLOP/s.
const attainable = (i: number, gbs: number) => Math.min(PEAK_TFLOPS, (gbs * i) / 1000);

export default function TileIntensity() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [ceiling, setCeiling] = useState<Ceiling>("measured");
  const [bm, setBm] = useState<number>(128);
  const [bn, setBn] = useState<number>(128);

  const bw = BANDWIDTHS[ceiling];
  const ridge = (PEAK_TFLOPS * 1000) / bw;
  const i = intensity(bm, bn);
  const reach = attainable(i, bw);
  const smemKiB = ((bm + bn) * 8 * 4) / 1024; // one BK = 8 step of A and B, single buffered

  const h = 250;
  const m = { l: 46, r: 12, t: 16, b: 44 };
  const pw = Math.max(160, width - m.l - m.r);
  const ph = h - m.t - m.b;
  const slot = pw / TILE_EDGES.length;
  const barW = Math.min(46, slot * 0.6);
  const yMax = 22;
  const y = (t: number) => m.t + ph - (t / yMax) * ph;
  const selectedSquare = bm === bn ? TILE_EDGES.indexOf(bm as (typeof TILE_EDGES)[number]) : -1;

  return (
    <Figure
      title="How big a tile has to be"
      caption="Bars show what DRAM traffic alone allows a square tile to reach, capped by the 19.5 TFLOP/s FP32 peak. The sliders pick any rectangular tile for the readout; its bar is outlined when the tile is square."
      controls={
        <>
          <Segmented
            label="Bandwidth"
            value={ceiling}
            onChange={setCeiling}
            options={[
              { value: "measured", label: "Copy 1275 GB/s" },
              { value: "rated", label: "Rated 1935 GB/s" },
            ]}
          />
          <Slider label="Tile rows BM" values={TILE_CHOICES} value={bm} onChange={setBm} />
          <Slider label="Tile columns BN" values={TILE_CHOICES} value={bn} onChange={setBn} />
        </>
      }
      readout={
        <>
          <Stat label="Tile" value={`${bm} x ${bn}`} />
          <Stat label="Intensity" value={`${i.toFixed(1)} FLOP/B`} tone={color.a} />
          <Stat label="Ridge point" value={`${ridge.toFixed(1)} FLOP/B`} tone={color.c} />
          <Stat label="DRAM allows" value={`${reach.toFixed(1)} TFLOP/s`} tone={color.b} />
          <Stat label="Limited by" value={i < ridge ? "memory" : "arithmetic"} tone={i < ridge ? color.warn : color.ok} />
          <Stat label="Smem per K step of 8" value={`${smemKiB.toFixed(0)} KiB`} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={h}
          role="img"
          aria-label={`Bar chart of the throughput that DRAM traffic allows for square tiles of 8 to 256, at ${bw} GB/s, capped at ${PEAK_TFLOPS} TFLOP/s. The selected ${bm} by ${bn} tile has an intensity of ${i.toFixed(1)} FLOP per byte against a ridge point of ${ridge.toFixed(1)}.`}
        >
          {[0, 5, 10, 15, 20].map((t) => (
            <g key={t}>
              <line x1={m.l} x2={m.l + pw} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {t}
              </text>
            </g>
          ))}
          <text x={4} y={m.t - 4} fontSize={11} style={{ fill: color.muted }}>
            TFLOP/s
          </text>
          <line x1={m.l} x2={m.l + pw} y1={y(PEAK_TFLOPS)} y2={y(PEAK_TFLOPS)} strokeDasharray="5 4" style={{ stroke: color.ok }} />
          <text x={m.l + pw} y={y(PEAK_TFLOPS) - 4} textAnchor="end" fontSize={11} style={{ fill: color.ok }}>
            FP32 peak 19.5
          </text>
          {TILE_EDGES.map((edge, idx) => {
            const iEdge = intensity(edge, edge);
            const t = attainable(iEdge, bw);
            const mem = iEdge < ridge;
            const cx = m.l + slot * idx + slot / 2;
            const tone = mem ? color.warn : color.ok;
            return (
              <g key={edge}>
                <rect
                  x={cx - barW / 2}
                  y={y(t)}
                  width={barW}
                  height={Math.max(1, y(0) - y(t))}
                  rx={3}
                  style={{ fill: wash(tone, 38), stroke: idx === selectedSquare ? color.fg : tone, strokeWidth: idx === selectedSquare ? 2.5 : 1 }}
                />
                <text x={cx} y={y(t) - 5} textAnchor="middle" fontSize={11} style={{ fill: color.fg }}>
                  {t.toFixed(1)}
                </text>
                <text x={cx} y={h - 24} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
                  {edge} x {edge}
                </text>
                <text x={cx} y={h - 10} textAnchor="middle" fontSize={10} style={{ fill: color.subtle }}>
                  {iEdge} FLOP/B
                </text>
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "memory bound", tone: color.warn },
            { label: "reaches the arithmetic peak", tone: color.ok },
          ]}
        />
      </div>
    </Figure>
  );
}
