import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Wave efficiency on an A100 80GB PCIe: waves / ceil(waves), where
// waves = grid / (108 SMs x resident blocks per SM). The 108 comes from
// docs/FACTS.md. The series below are printed by
// scripts/figures/28-reading-profiler-reports.py, one value per grid of 54,
// 108, ... 3456 blocks, for 1, 2, 4, and 8 resident blocks per SM.
const SM_COUNT = 108;
const STEP = 54;

// WAVES: efficiency at grid = 54, 108, ... 3456
const WAVES: Record<number, number[]> = {
  1: [0.5, 1.0, 0.75, 1.0, 0.8333, 1.0, 0.875, 1.0, 0.9, 1.0, 0.9167, 1.0, 0.9286, 1.0, 0.9375, 1.0, 0.9444, 1.0, 0.95, 1.0, 0.9545, 1.0, 0.9583, 1.0, 0.9615, 1.0, 0.9643, 1.0, 0.9667, 1.0, 0.9688, 1.0, 0.9706, 1.0, 0.9722, 1.0, 0.9737, 1.0, 0.975, 1.0, 0.9762, 1.0, 0.9773, 1.0, 0.9783, 1.0, 0.9792, 1.0, 0.98, 1.0, 0.9808, 1.0, 0.9815, 1.0, 0.9821, 1.0, 0.9828, 1.0, 0.9833, 1.0, 0.9839, 1.0, 0.9844, 1.0],
  2: [0.25, 0.5, 0.75, 1.0, 0.625, 0.75, 0.875, 1.0, 0.75, 0.8333, 0.9167, 1.0, 0.8125, 0.875, 0.9375, 1.0, 0.85, 0.9, 0.95, 1.0, 0.875, 0.9167, 0.9583, 1.0, 0.8929, 0.9286, 0.9643, 1.0, 0.9062, 0.9375, 0.9688, 1.0, 0.9167, 0.9444, 0.9722, 1.0, 0.925, 0.95, 0.975, 1.0, 0.9318, 0.9545, 0.9773, 1.0, 0.9375, 0.9583, 0.9792, 1.0, 0.9423, 0.9615, 0.9808, 1.0, 0.9464, 0.9643, 0.9821, 1.0, 0.95, 0.9667, 0.9833, 1.0, 0.9531, 0.9688, 0.9844, 1.0],
  4: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 0.7083, 0.75, 0.7917, 0.8333, 0.875, 0.9167, 0.9583, 1.0, 0.7812, 0.8125, 0.8438, 0.875, 0.9062, 0.9375, 0.9688, 1.0, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 0.8542, 0.875, 0.8958, 0.9167, 0.9375, 0.9583, 0.9792, 1.0, 0.875, 0.8929, 0.9107, 0.9286, 0.9464, 0.9643, 0.9821, 1.0, 0.8906, 0.9062, 0.9219, 0.9375, 0.9531, 0.9688, 0.9844, 1.0],
  8: [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 0.5312, 0.5625, 0.5938, 0.625, 0.6562, 0.6875, 0.7188, 0.75, 0.7812, 0.8125, 0.8438, 0.875, 0.9062, 0.9375, 0.9688, 1.0, 0.6875, 0.7083, 0.7292, 0.75, 0.7708, 0.7917, 0.8125, 0.8333, 0.8542, 0.875, 0.8958, 0.9167, 0.9375, 0.9583, 0.9792, 1.0, 0.7656, 0.7812, 0.7969, 0.8125, 0.8281, 0.8438, 0.8594, 0.875, 0.8906, 0.9062, 0.9219, 0.9375, 0.9531, 0.9688, 0.9844, 1.0],
};

export default function WaveSawtooth() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [perSm, setPerSm] = useState(1);
  const [index, setIndex] = useState(1);

  const series = WAVES[perSm];
  const grid = (index + 1) * STEP;
  const waves = grid / (SM_COUNT * perSm);
  const whole = Math.ceil(waves);
  const efficiency = series[index];
  const lastFill = waves - (whole - 1);

  const w = Math.max(width, 320);
  const h = 240;
  const padL = 40;
  const padR = 12;
  const padT = 12;
  const padB = 32;
  const x = (i: number) => padL + (i / (series.length - 1)) * (w - padL - padR);
  const y = (v: number) => padT + (1 - v) * (h - padT - padB);
  const path = series.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");

  return (
    <Figure
      title="Wave quantization: the last wave decides"
      caption="Efficiency is the waves a grid needs over the whole waves it occupies. It drops just after a grid passes a whole number of waves, and it recovers as the last wave fills. A grid under one wave leaves SMs empty for the whole kernel."
      controls={
        <>
          <Segmented
            label="Resident blocks per SM"
            value={perSm}
            onChange={setPerSm}
            options={[1, 2, 4, 8].map((v) => ({ value: v, label: String(v) }))}
          />
          <Slider
            label="Grid size"
            min={0}
            max={series.length - 1}
            value={index}
            onChange={setIndex}
            format={(i) => `${((i + 1) * STEP).toLocaleString("en-US")} blocks`}
          />
        </>
      }
      readout={
        <>
          <Stat label="Blocks per wave" value={(SM_COUNT * perSm).toLocaleString("en-US")} />
          <Stat label="Waves" value={waves.toFixed(3)} tone={color.a} />
          <Stat label="Last wave full" value={`${(lastFill * 100).toFixed(0)}%`} tone={color.b} />
          <Stat label="Efficiency" value={`${(efficiency * 100).toFixed(1)}%`} tone={color.c} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={`Wave efficiency against grid size at ${perSm} resident blocks per SM. A grid of ${grid} blocks is ${waves.toFixed(3)} waves and ${(efficiency * 100).toFixed(1)} percent efficient.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={padL} x2={w - padR} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={padL - 6} y={y(t) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          <text x={(padL + w - padR) / 2} y={h - 6} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
            grid size, 54 to 3456 blocks
          </text>
          <path d={path} fill="none" strokeWidth={2} style={{ stroke: color.fg }} />
          <line x1={x(index)} x2={x(index)} y1={padT} y2={h - padB} strokeDasharray="4 4" style={{ stroke: color.faint }} />
          <circle cx={x(index)} cy={y(efficiency)} r={6} style={{ fill: color.accent, stroke: color.paper }} />
        </svg>
      </div>
    </Figure>
  );
}
