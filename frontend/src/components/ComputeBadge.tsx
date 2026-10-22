import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { Dot } from "./ui";

const POLL_MS = 60_000;

/**
 * How many things are running on a GPU provider, or null while unknown.
 *
 * The point is that a container nobody remembers starting is visible from every
 * page, not only from the compute panel. The shell reads this once and hands it
 * to everything that shows it.
 */
export function useActiveCompute(enabled: boolean): number | null {
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let live = true;

    const read = () => {
      api
        .infra()
        .then((result) => {
          if (live) setActive(result.active);
        })
        .catch(() => {
          if (live) setActive(null);
        });
    };

    // The first read happens even in a background tab, so the count is right
    // the moment the page is looked at.
    read();
    const onVisible = () => {
      if (!document.hidden) read();
    };
    document.addEventListener("visibilitychange", onVisible);
    const timer = window.setInterval(() => {
      if (!document.hidden) read();
    }, POLL_MS);

    return () => {
      live = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [enabled]);

  return active;
}

export function computeLabel(active: number | null): string {
  if (active === null) return "Checking";
  return active > 0 ? `${active} running` : "Idle";
}

export function computeTitle(active: number | null): string {
  if (active === null) return "Checking the GPU providers";
  return active > 0
    ? `${active} thing${active === 1 ? "" : "s"} running on a GPU provider`
    : "Nothing is running on a GPU provider";
}

/** The status light alone, for the collapsed rail and the sidebar's nav row. */
export function ComputeDot({ active }: { active: number | null }) {
  const running = (active ?? 0) > 0;
  return <Dot tone={active === null ? "neutral" : running ? "bad" : "ok"} pulse={running} />;
}

/** A compact link to the compute panel, for the narrow-screen top bar. */
export default function ComputeBadge({ active }: { active: number | null }) {
  const running = (active ?? 0) > 0;
  return (
    <Link
      to="/compute"
      title={computeTitle(active)}
      className={`flex h-8 shrink-0 items-center gap-2 whitespace-nowrap rounded-md border px-2.5 text-[12.5px] font-medium transition-colors ${
        running
          ? "border-bad/35 bg-bad/8 text-bad hover:bg-bad/14"
          : "border-line bg-card text-fg-subtle hover:border-line-strong hover:text-fg"
      }`}
    >
      <ComputeDot active={active} />
      <span className="tabular-nums">GPU {computeLabel(active).toLowerCase()}</span>
    </Link>
  );
}
