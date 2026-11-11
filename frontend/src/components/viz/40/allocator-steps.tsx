import { Figure, Legend, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

// The lab's recorded allocation trace (content/labs/40-memory-budget/fixture_snapshot.py,
// ALLOC_TRACE) run through the lab solution's caching-allocator simulator.
// scripts/figures/40-memory-management-and-oom.py prints this array. Sizes are MiB.
// A block is [size, name]; an empty name is a free block.

type Block = [number, string];
type Step = { label: string; segments: Block[][]; allocated: number; reserved: number };

const STEPS: Step[] = [
  { label: "start", segments: [], allocated: 0, reserved: 0 },
  { label: "alloc a 12 MiB", segments: [[[12, "a"]]], allocated: 12, reserved: 12 },
  { label: "alloc b 12 MiB", segments: [[[12, "a"]], [[12, "b"]]], allocated: 24, reserved: 24 },
  { label: "alloc c 12 MiB", segments: [[[12, "a"]], [[12, "b"]], [[12, "c"]]], allocated: 36, reserved: 36 },
  { label: "alloc d 12 MiB", segments: [[[12, "a"]], [[12, "b"]], [[12, "c"]], [[12, "d"]]], allocated: 48, reserved: 48 },
  { label: "free a", segments: [[[12, ""]], [[12, "b"]], [[12, "c"]], [[12, "d"]]], allocated: 36, reserved: 48 },
  { label: "free c", segments: [[[12, ""]], [[12, "b"]], [[12, ""]], [[12, "d"]]], allocated: 24, reserved: 48 },
  { label: "alloc e 16 MiB", segments: [[[12, ""]], [[12, "b"]], [[12, ""]], [[12, "d"]], [[16, "e"]]], allocated: 40, reserved: 64 },
  { label: "alloc f 4 MiB", segments: [[[4, "f"], [8, ""]], [[12, "b"]], [[12, ""]], [[12, "d"]], [[16, "e"]]], allocated: 44, reserved: 64 },
  { label: "free e", segments: [[[4, "f"], [8, ""]], [[12, "b"]], [[12, ""]], [[12, "d"]], [[16, ""]]], allocated: 28, reserved: 64 },
  { label: "alloc g 20 MiB", segments: [[[4, "f"], [8, ""]], [[12, "b"]], [[12, ""]], [[12, "d"]], [[16, ""]], [[20, "g"]]], allocated: 48, reserved: 84 },
];

const MAX_MIB = 20;

export default function AllocatorSteps() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const stepper = useStepper(STEPS.length, 900);
  const step = STEPS[stepper.step];

  const gx = 28;
  const rowH = 26;
  const scale = Math.max(8, Math.floor((width - gx - 16) / MAX_MIB));
  const height = Math.max(1, step.segments.length) * rowH + 10;

  let largestFree = 0;
  let totalFree = 0;
  for (const seg of step.segments) {
    for (const [size, name] of seg) {
      if (name === "") {
        largestFree = Math.max(largestFree, size);
        totalFree += size;
      }
    }
  }

  return (
    <Figure
      title="A caching allocator keeps its segments"
      caption="Each row is one cudaMalloc segment. Filled blocks are live tensors and open blocks are cached. After free e the allocator holds 36 MiB of free blocks but no 20 MiB block, so alloc g reserves a new segment and reserved climbs to 84 MiB while allocated is 48 MiB."
      controls={<StepControls stepper={stepper} count={STEPS.length} describe={(i) => STEPS[i].label} />}
      readout={
        <>
          <Stat label="Allocated" value={`${step.allocated} MiB`} tone={color.a} />
          <Stat label="Reserved" value={`${step.reserved} MiB`} tone={color.b} />
          <Stat label="Free in cache" value={`${totalFree} MiB`} />
          <Stat label="Largest free block" value={`${largestFree} MiB`} />
        </>
      }
    >
      <div ref={ref}>
        <Legend
          items={[
            { label: "Live tensor", tone: color.a },
            { label: "Cached free block", tone: color.faint },
          ]}
        />
        <svg
          width={Math.max(width, gx + MAX_MIB * scale + 8)}
          height={height}
          role="img"
          aria-label={`After ${step.label}: ${step.allocated} MiB allocated, ${step.reserved} MiB reserved in ${step.segments.length} segments.`}
        >
          {step.segments.map((seg, row) => {
            let x = gx;
            return (
              <g key={row}>
                <text x={gx - 6} y={row * rowH + 17} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                  {row}
                </text>
                {seg.map(([size, name], i) => {
                  const w = size * scale;
                  const left = x;
                  x += w;
                  const live = name !== "";
                  return (
                    <g key={i}>
                      <rect
                        x={left}
                        y={row * rowH + 4}
                        width={w - 2}
                        height={rowH - 6}
                        rx={3}
                        style={{
                          fill: live ? color.a : wash(color.faint, 30),
                          stroke: live ? "none" : color.lineStrong,
                        }}
                      />
                      <text
                        x={left + (w - 2) / 2}
                        y={row * rowH + 20}
                        fontSize={11}
                        textAnchor="middle"
                        style={{ fill: live ? color.paper : color.muted }}
                      >
                        {live ? `${name} ${size}` : `${size}`}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
