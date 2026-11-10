import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// BFS from vertex 0 on the seeded 1,520-vertex preferential-attachment graph of
// lab 32 (scripts/figures/32-sparse-and-irregular-work.py, seed 32): 9,020 directed
// edges, 20 detached vertices, deepest level 4. Per level: the edges a top-down round
// examines (m_f, the sum of the frontier's degrees), the edges a bottom-up round
// examines from the same state, and m_u, the edges of the vertices not yet in a level
// at or below that depth. The rule is the lab's should_switch: top-down to bottom-up
// when m_f * alpha > m_u, back when frontier size * beta < n, with beta = 18 and n = 1520.
// Model counts from the lab's reference solution, not GPU measurements.

const N = 1520;
const BETA = 18;

const LEVELS = [
  { depth: 0, frontier: 1, topDown: 85, bottomUp: 8028, unexplored: 8935 },
  { depth: 1, frontier: 85, topDown: 992, bottomUp: 4659, unexplored: 7943 },
  { depth: 2, frontier: 605, topDown: 4282, bottomUp: 1094, unexplored: 3661 },
  { depth: 3, frontier: 766, topDown: 3479, bottomUp: 81, unexplored: 182 },
  { depth: 4, frontier: 43, topDown: 144, bottomUp: 38, unexplored: 38 },
];

type Direction = "top" | "bottom";

function plan(alpha: number): Direction[] {
  let direction: Direction = "top";
  return LEVELS.map((level) => {
    if (direction === "top") {
      direction = level.topDown * alpha > level.unexplored ? "bottom" : "top";
    } else {
      direction = level.frontier * BETA < N ? "top" : "bottom";
    }
    return direction;
  });
}

export default function BfsDirections() {
  const [alpha, setAlpha] = useState(4);
  const [ref, width] = useWidth<HTMLDivElement>(560);

  const directions = plan(alpha);
  const cost = LEVELS.map((level, i) => (directions[i] === "top" ? level.topDown : level.bottomUp));
  const total = cost.reduce((a, b) => a + b, 0);
  const allTop = LEVELS.reduce((a, level) => a + level.topDown, 0);
  const best = LEVELS.reduce((a, level) => a + Math.min(level.topDown, level.bottomUp), 0);

  // Log scale from 10 to 10,000 edges, so 38 and 8,028 are both visible.
  const padL = 56;
  const padR = 12;
  const height = 220;
  const plotH = height - 56;
  const plotW = Math.max(220, width - padL - padR);
  const group = plotW / LEVELS.length;
  const barW = Math.min(34, group / 2 - 4);
  const y = (v: number) => 10 + plotH - (Math.log10(Math.max(v, 10)) - 1) * (plotH / 3);

  return (
    <Figure
      title="Top-down against bottom-up, level by level"
      caption="The bars are the edges each direction would examine in that round, on a log scale. Top-down is cheap while the frontier is small and bottom-up is cheap once most vertices are visited. The outlined bar is the direction the rule picks. At alpha 4 it picks the cheaper direction at four of the five levels, 2,396 edges in all. At alpha 15 it switches one level early and examines 6,063."
      controls={<Slider label="Alpha" min={1} max={30} value={alpha} onChange={setAlpha} />}
      readout={
        <>
          <Stat label="Edges examined" value={total.toLocaleString("en-US")} tone={color.accent} />
          <Stat label="All top-down" value={allTop.toLocaleString("en-US")} />
          <Stat label="Cheapest per level" value={best.toLocaleString("en-US")} />
          <Stat label="Directions" value={directions.map((d) => d[0].toUpperCase()).join(" ")} />
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={padL + plotW + padR}
          height={height}
          role="img"
          aria-label={`BFS cost per level. With alpha ${alpha} the rule examines ${total} edges against ${allTop} for top-down alone.`}
        >
          {[10, 100, 1000, 10000].map((tick) => (
            <g key={tick}>
              <line x1={padL} x2={padL + plotW} y1={y(tick)} y2={y(tick)} style={{ stroke: color.line }} />
              <text x={padL - 6} y={y(tick) + 4} fontSize={11} textAnchor="end" style={{ fill: color.muted }}>
                {tick.toLocaleString("en-US")}
              </text>
            </g>
          ))}
          {LEVELS.map((level, i) => {
            const x0 = padL + i * group + group / 2;
            const picked = directions[i];
            return (
              <g key={level.depth}>
                <rect
                  x={x0 - barW - 2}
                  y={y(level.topDown)}
                  width={barW}
                  height={10 + plotH - y(level.topDown)}
                  style={{
                    fill: color.accent,
                    stroke: picked === "top" ? color.fg : "none",
                    strokeWidth: 2,
                  }}
                />
                <rect
                  x={x0 + 2}
                  y={y(level.bottomUp)}
                  width={barW}
                  height={10 + plotH - y(level.bottomUp)}
                  style={{
                    fill: color.b,
                    stroke: picked === "bottom" ? color.fg : "none",
                    strokeWidth: 2,
                  }}
                />
                <text x={x0} y={height - 28} fontSize={11} textAnchor="middle" style={{ fill: color.fg }}>
                  level {level.depth}
                </text>
                <text x={x0} y={height - 14} fontSize={11} textAnchor="middle" style={{ fill: color.muted }}>
                  frontier {level.frontier.toLocaleString("en-US")}
                </text>
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Top-down edges", tone: color.accent },
            { label: "Bottom-up edges", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
