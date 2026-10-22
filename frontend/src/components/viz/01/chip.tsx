import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Every figure below comes from docs/FACTS.md, section 2.1 unless noted.
type GpuKey = "A100" | "H100" | "B200";

type Gpu = {
  label: string;
  sms: number; // SMs enabled
  dies: number; // 2 on the B200, which presents them as one device
  stacks: number; // HBM stacks
  bandwidth: string; // rated, GB/s
  l2: string;
  smem: string; // shared memory plus L1 per SM
  fp32PerSm: number; // FP32 CUDA cores per SM (B200 is derived, section 2.1)
  tensorFlopPerClock: number; // bf16 dense FLOP per SM per clock, section 2.2
  tmem: boolean; // 256 KB of tensor memory per SM
  nvlink: string; // per direction, section 7.1
  host: string; // per direction, section 7.1
  hbm: string;
};

const GPUS: Record<GpuKey, Gpu> = {
  A100: {
    label: "A100 80GB PCIe",
    sms: 108,
    dies: 1,
    stacks: 5,
    bandwidth: "1,935 GB/s",
    l2: "40 MB",
    smem: "192 KB",
    fp32PerSm: 64,
    tensorFlopPerClock: 2048,
    tmem: false,
    nvlink: "NVLink bridge, 300 GB/s per direction",
    host: "PCIe Gen4 x16, 31.5 GB/s per direction",
    hbm: "HBM2e",
  },
  H100: {
    label: "H100 SXM5",
    sms: 132,
    dies: 1,
    stacks: 5,
    bandwidth: "3,350 GB/s",
    l2: "50 MB",
    smem: "256 KB",
    fp32PerSm: 128,
    tensorFlopPerClock: 4096,
    tmem: false,
    nvlink: "NVLink 4, 450 GB/s per direction",
    host: "PCIe Gen5 x16, 63 GB/s per direction",
    hbm: "HBM3",
  },
  B200: {
    label: "B200 (HGX)",
    sms: 148,
    dies: 2,
    stacks: 8,
    bandwidth: "8,000 GB/s",
    l2: "126 MB",
    smem: "256 KB",
    fp32PerSm: 128,
    tensorFlopPerClock: 8192,
    tmem: true,
    nvlink: "NVLink 5, 900 GB/s per direction",
    host: "PCIe Gen5 x16, 63 GB/s per direction",
    hbm: "HBM3e",
  },
};

const REGISTERS_KIB_PER_SM = 256; // 64K 32-bit registers, section 2.1

type View = "chip" | "sm";

export default function ChipAnatomy() {
  const [key, setKey] = useState<GpuKey>("A100");
  const [view, setView] = useState<View>("chip");
  const [ref, width] = useWidth<HTMLDivElement>(560);
  const gpu = GPUS[key];

  const pad = 8;
  const label = (x: number, y: number, text: string, opts?: { anchor?: "start" | "middle" | "end"; tone?: string; size?: number }) => (
    <text x={x} y={y} textAnchor={opts?.anchor ?? "middle"} fontSize={opts?.size ?? 11} style={{ fill: opts?.tone ?? color.fg }}>
      {text}
    </text>
  );

  let drawing;
  let height;

  if (view === "chip") {
    const hbmH = 30;
    const gap = 6;
    const dieGap = gpu.dies === 2 ? 46 : 0;
    const dieW = (width - 2 * pad - dieGap) / gpu.dies;
    const perDie = gpu.sms / gpu.dies;
    const cols = gpu.dies === 2 ? 10 : 12;
    const rows = Math.ceil(perDie / cols);
    const cell = Math.max(6, Math.min(20, Math.floor((dieW - 20) / cols) - 2));
    const gridH = rows * (cell + 2);
    const l2H = 24;
    const dieH = 20 + l2H + 10 + gridH + 12;
    const hbmTop = pad;
    const dieTop = hbmTop + hbmH + gap;
    const linkTop = dieTop + dieH + gap;
    height = linkTop + 34;

    const stackW = (width - 2 * pad - (gpu.stacks - 1) * 4) / gpu.stacks;
    drawing = (
      <g>
        {Array.from({ length: gpu.stacks }, (_, i) => (
          <g key={`hbm${i}`}>
            <rect x={pad + i * (stackW + 4)} y={hbmTop} width={stackW} height={hbmH} rx={3} style={{ fill: wash(color.b, 30), stroke: color.b }} />
            {label(pad + i * (stackW + 4) + stackW / 2, hbmTop + 19, stackW > 46 ? "HBM stack" : "HBM")}
          </g>
        ))}
        {Array.from({ length: gpu.dies }, (_, d) => {
          const x0 = pad + d * (dieW + dieGap);
          return (
            <g key={`die${d}`}>
              <rect x={x0} y={dieTop} width={dieW} height={dieH} rx={6} style={{ fill: color.well, stroke: color.lineStrong }} />
              <rect x={x0 + 8} y={dieTop + 8} width={dieW - 16} height={l2H} rx={3} style={{ fill: wash(color.c, 30), stroke: color.c }} />
              {label(x0 + dieW / 2, dieTop + 8 + 16, gpu.dies === 2 ? "L2 cache (126 MB across both dies)" : `L2 cache, ${gpu.l2}`)}
              {Array.from({ length: perDie }, (_, i) => {
                const r = Math.floor(i / cols);
                const c = i % cols;
                const gridW = cols * (cell + 2);
                return (
                  <rect
                    key={i}
                    x={x0 + (dieW - gridW) / 2 + c * (cell + 2)}
                    y={dieTop + 8 + l2H + 10 + r * (cell + 2)}
                    width={cell}
                    height={cell}
                    rx={2}
                    style={{ fill: wash(color.a, 55), stroke: color.a }}
                  />
                );
              })}
              {label(x0 + dieW / 2, dieTop + dieH - 4, `${perDie} SMs`, { tone: color.muted })}
            </g>
          );
        })}
        {gpu.dies === 2 && (
          <g>
            <line x1={pad + dieW + 4} x2={pad + dieW + dieGap - 4} y1={dieTop + dieH / 2} y2={dieTop + dieH / 2} style={{ stroke: color.warn, strokeWidth: 3 }} />
            {label(pad + dieW + dieGap / 2, dieTop + dieH / 2 - 8, "NV-HBI", { size: 10 })}
            {label(pad + dieW + dieGap / 2, dieTop + dieH / 2 + 18, "10 TB/s", { size: 10 })}
          </g>
        )}
        <rect x={pad} y={linkTop} width={width - 2 * pad} height={26} rx={4} style={{ fill: wash(color.violet, 18), stroke: color.violet }} />
        {label(width / 2, linkTop + 17, width > 520 ? `${gpu.nvlink}  ·  ${gpu.host}` : "NVLink and PCIe to the outside", { size: 10.5 })}
      </g>
    );
  } else {
    const narrow = width < 520;
    const cols = narrow ? 2 : 4;
    const rows = narrow ? 2 : 1;
    const boxGap = 8;
    const boxW = (width - 2 * pad - (cols - 1) * boxGap) / cols;
    const boxH = 176;
    const top = pad + 18;
    const lanes = gpu.fp32PerSm / 4;
    const baseY = top + rows * (boxH + boxGap);
    const barH = 26;
    height = baseY + barH + (gpu.tmem ? barH + 6 : 0) + 6 + 10;

    const inner = (x: number, y: number, w: number, h: number, text: string, tone: string) => (
      <g>
        <rect x={x} y={y} width={w} height={h} rx={3} style={{ fill: wash(tone, 28), stroke: tone }} />
        {label(x + w / 2, y + h / 2 + 4, text, { size: 10.5 })}
      </g>
    );

    drawing = (
      <g>
        {label(width / 2, pad + 8, "One streaming multiprocessor (SM): four identical sub-partitions", { tone: color.muted })}
        {Array.from({ length: 4 }, (_, i) => {
          const x0 = pad + (i % cols) * (boxW + boxGap);
          const y0 = top + Math.floor(i / cols) * (boxH + boxGap);
          const ix = x0 + 8;
          const iw = boxW - 16;
          return (
            <g key={i}>
              <rect x={x0} y={y0} width={boxW} height={boxH} rx={6} style={{ fill: color.well, stroke: color.lineStrong }} />
              {label(x0 + boxW / 2, y0 + 15, `Sub-partition ${i}`, { tone: color.muted, size: 10.5 })}
              {inner(ix, y0 + 22, iw, 24, "Warp scheduler", color.accent)}
              {inner(ix, y0 + 52, iw, 24, `Registers ${REGISTERS_KIB_PER_SM / 4} KB`, color.c)}
              {inner(ix, y0 + 82, iw, 24, `${lanes} FP32 lanes`, color.a)}
              {inner(ix, y0 + 112, iw, 28, `Tensor core · ${gpu.tensorFlopPerClock / 4} FLOP/clk`, color.d)}
              {inner(ix, y0 + 146, iw, 22, "LD/ST and SFU", color.muted)}
            </g>
          );
        })}
        <rect x={pad} y={baseY} width={width - 2 * pad} height={barH} rx={4} style={{ fill: wash(color.b, 28), stroke: color.b }} />
        {label(width / 2, baseY + 17, `Shared memory and L1 cache, ${gpu.smem} per SM`)}
        {gpu.tmem && (
          <g>
            <rect x={pad} y={baseY + barH + 6} width={width - 2 * pad} height={barH} rx={4} style={{ fill: wash(color.d, 28), stroke: color.d }} />
            {label(width / 2, baseY + barH + 6 + 17, "Tensor memory (TMEM), 256 KB per SM")}
          </g>
        )}
      </g>
    );
  }

  const registerMiB = (gpu.sms * REGISTERS_KIB_PER_SM) / 1024;

  return (
    <Figure
      title="The chip, top-down"
      caption="Counts are from NVIDIA's whitepapers and product pages. The B200 presents two dies as one CUDA device, and its 148 SMs split as 74 per die. The sub-partition labels use per-SM totals divided by four, and the B200's FP32 lane count is derived from its rated FP32 rate, not published."
      controls={
        <>
          <Segmented
            label="GPU"
            value={key}
            options={[
              { value: "A100", label: "A100" },
              { value: "H100", label: "H100" },
              { value: "B200", label: "B200" },
            ]}
            onChange={setKey}
          />
          <Segmented
            label="View"
            value={view}
            options={[
              { value: "chip", label: "Whole chip" },
              { value: "sm", label: "One SM" },
            ]}
            onChange={setView}
          />
        </>
      }
      readout={
        <>
          <Stat label="GPU" value={gpu.label} />
          <Stat label="SMs" value={gpu.sms} tone={color.a} />
          <Stat label="HBM" value={`${gpu.stacks} stacks of ${gpu.hbm}, ${gpu.bandwidth}`} tone={color.b} />
          <Stat label="L2" value={gpu.l2} tone={color.c} />
          <Stat label="Register files, all SMs" value={`${registerMiB.toFixed(0)} MiB`} />
          <Stat label="bf16 tensor FLOP per SM per clock" value={gpu.tensorFlopPerClock.toLocaleString("en-US")} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg width={width} height={height} role="img" aria-label={`${gpu.label}: ${view === "chip" ? "SMs, L2 cache, HBM stacks, and links" : "the four sub-partitions of one SM"}`}>
          {drawing}
        </svg>
      </div>
    </Figure>
  );
}
