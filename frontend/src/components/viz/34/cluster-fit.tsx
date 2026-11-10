import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, useWidth, wash } from "../kit";

// The H100 SXM5 has 8 GPCs and 132 SMs (NVIDIA Hopper in-depth blog). NVIDIA
// does not publish the SM count of each GPC. This split is an assumption: six
// GPCs of 16 SMs and two of 18. A cluster sits inside one GPC, so a GPC with S
// SMs holds floor(S * b / c) clusters of size c, with b blocks per SM.
// scripts/figures/34-clusters-and-distributed-shared-memory.py prints the same
// table. Sizes above 8 need cudaFuncAttributeNonPortableClusterSizeAllowed.

const GPC_SMS = [16, 16, 16, 16, 16, 16, 18, 18];
const PORTABLE_MAX = 8;

export default function ClusterFit() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [size, setSize] = useState(3);
  const [perSm, setPerSm] = useState<1 | 2>(1);

  const perGpc = GPC_SMS.map((sms) => Math.floor((sms * perSm) / size));
  const clusters = perGpc.reduce((a, b) => a + b, 0);
  const totalSlots = GPC_SMS.reduce((a, b) => a + b, 0) * perSm;
  const idleSlots = totalSlots - clusters * size;

  const cols = 8;
  const cell = Math.max(10, Math.min(18, Math.floor((width - 24) / (cols * 2.2))));
  const gpcW = Math.floor((width - 16) / cols);
  const perRow = 3;

  return (
    <Figure
      title="Clusters packed into the GPCs of an H100 SXM5"
      caption="Each column is one GPC and each square is one SM. A filled square holds a block of a resident cluster. A cluster never spans two GPCs, so SMs left over in a GPC stay idle. The split of 132 SMs into 16 and 18 per GPC is an assumption. The real count comes from cudaOccupancyMaxActiveClusters."
      controls={
        <>
          <Slider label="Cluster size" min={1} max={16} value={size} onChange={setSize} />
          <Segmented
            label="Blocks per SM"
            value={perSm}
            options={[
              { value: 1, label: "1" },
              { value: 2, label: "2" },
            ]}
            onChange={setPerSm}
          />
        </>
      }
      readout={
        <>
          <Stat label="Resident clusters" value={String(clusters)} tone={color.a} />
          <Stat label="Resident blocks" value={String(clusters * size)} />
          <Stat label="Idle block slots" value={String(idleSlots)} tone={idleSlots ? color.bad : color.ok} />
          <Stat label="Portable" value={size <= PORTABLE_MAX ? "yes" : "needs opt-in"} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={Math.max(width, 200)}
          height={150}
          role="img"
          aria-label={`Clusters of ${size} with ${perSm} block per SM: ${clusters} resident clusters, ${idleSlots} idle block slots.`}
        >
          {GPC_SMS.map((sms, g) => {
            const used = perGpc[g] * size;
            return (
              <g key={g} transform={`translate(${8 + g * gpcW}, 8)`}>
                <text x={0} y={8} fontSize={11} style={{ fill: color.muted }}>
                  GPC {g}
                </text>
                {Array.from({ length: sms }, (_, i) => {
                  const slotsUsed = Math.max(0, Math.min(perSm, used - i * perSm));
                  const on = slotsUsed === perSm;
                  const partial = slotsUsed > 0 && !on;
                  return (
                    <rect
                      key={i}
                      x={(i % perRow) * (cell + 2)}
                      y={16 + Math.floor(i / perRow) * (cell + 2)}
                      width={cell}
                      height={cell}
                      rx={2}
                      style={{
                        fill: on ? color.a : partial ? wash(color.a, 45) : wash(color.faint, 35),
                      }}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "SM holds a resident block", tone: color.a },
            { label: "idle", tone: color.faint },
          ]}
        />
      </div>
    </Figure>
  );
}
