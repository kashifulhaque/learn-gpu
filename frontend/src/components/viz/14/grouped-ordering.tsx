import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";

// Constants. The grid is the output of a 4096 x 4096 x 4096 fp16 matmul cut
// into 128 x 256 tiles: 32 tile rows by 16 tile columns. One K step of a
// program reads a 128 x 64 tile of A and a 64 x 256 tile of B, 2 bytes per
// element. The resident count is the 108 SMs of the A100 (docs/FACTS.md
// section 2), at one program per SM, which the 144 KiB of shared memory per
// program of this tile shape allows.
const TILES_M = 32;
const TILES_N = 16;
const A_TILE_KIB = (128 * 64 * 2) / 1024; // 16
const B_TILE_KIB = (64 * 256 * 2) / 1024; // 32

/** The tile a program computes: the same arithmetic as the kernel's first lines. */
function groupedTile(pid: number, group: number): [number, number] {
  const perGroup = group * TILES_N;
  const groupId = Math.floor(pid / perGroup);
  const firstM = groupId * group;
  const size = Math.min(TILES_M - firstM, group);
  return [firstM + (pid % size), Math.floor((pid % perGroup) / size)];
}

export default function GroupedOrdering() {
  const [group, setGroup] = useState(8);
  const [resident, setResident] = useState(108);
  const [ref, width] = useWidth<HTMLDivElement>();

  const rows = new Set<number>();
  const cols = new Set<number>();
  const order = new Map<string, number>();
  for (let pid = 0; pid < TILES_M * TILES_N; pid++) {
    const [m, n] = groupedTile(pid, group);
    order.set(`${m},${n}`, pid);
    if (pid < resident) {
      rows.add(m);
      cols.add(n);
    }
  }
  const requestedKiB = resident * (A_TILE_KIB + B_TILE_KIB);
  const uniqueKiB = rows.size * A_TILE_KIB + cols.size * B_TILE_KIB;

  const margin = 22;
  const cell = Math.max(6, Math.min(12, Math.floor((width - margin - 8) / TILES_N)));
  const gridW = cell * TILES_N;
  const height = margin + cell * TILES_M + 4;

  return (
    <Figure
      title="Which tiles of A and B the resident programs read"
      caption={
        <>
          Each cell is one 128 x 256 output tile of a 4096 x 4096 x 4096 matmul. Blue cells are the programs that are
          resident together, one per SM. The bars on the left and top mark the rows of A and the columns of B that those
          programs read. The fewer marked bars, the more of each K step's loads are shared.
        </>
      }
      controls={
        <>
          <Slider label="GROUP_M" value={group} values={[1, 2, 4, 8, 16, 32]} onChange={setGroup} />
          <Slider label="Resident programs" value={resident} values={[27, 54, 108, 216]} onChange={setResident} />
        </>
      }
      readout={
        <>
          <Stat label="Tile rows of A read" value={rows.size} tone={color.a} />
          <Stat label="Tile columns of B read" value={cols.size} tone={color.b} />
          <Stat label="Requested per K step" value={`${requestedKiB.toLocaleString("en-US")} KiB`} />
          <Stat label="Distinct per K step" value={`${uniqueKiB.toLocaleString("en-US")} KiB`} tone={color.accent} />
          <Stat label="Distinct share" value={`${((100 * uniqueKiB) / requestedKiB).toFixed(1)}%`} />
        </>
      }
    >
      <div ref={ref} style={{ width: "100%" }}>
        <svg
          width={Math.max(width, margin + gridW)}
          height={height}
          role="img"
          aria-label="Grid of output tiles with the resident programs highlighted"
        >
          {Array.from({ length: TILES_N }, (_, n) => (
            <rect
              key={`c${n}`}
              x={margin + n * cell}
              y={4}
              width={Math.max(cell - 1, 1)}
              height={8}
              style={{ fill: cols.has(n) ? color.b : wash(color.muted, 12) }}
            />
          ))}
          {Array.from({ length: TILES_M }, (_, m) => (
            <rect
              key={`r${m}`}
              x={4}
              y={margin + m * cell}
              width={8}
              height={Math.max(cell - 1, 1)}
              style={{ fill: rows.has(m) ? color.a : wash(color.muted, 12) }}
            />
          ))}
          {Array.from({ length: TILES_M }, (_, m) =>
            Array.from({ length: TILES_N }, (_, n) => {
              const pid = order.get(`${m},${n}`) ?? 0;
              const on = pid < resident;
              return (
                <rect
                  key={`${m}-${n}`}
                  x={margin + n * cell}
                  y={margin + m * cell}
                  width={Math.max(cell - 1, 1)}
                  height={Math.max(cell - 1, 1)}
                  style={{ fill: on ? color.info : wash(color.muted, 14) }}
                />
              );
            }),
          )}
        </svg>
      </div>
      <Legend
        items={[
          { label: "resident program", tone: color.info },
          { label: "row of A read", tone: color.a },
          { label: "column of B read", tone: color.b },
        ]}
      />
    </Figure>
  );
}
