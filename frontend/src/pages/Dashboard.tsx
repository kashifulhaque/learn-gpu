import { useEffect, useMemo, useState } from "react";

import { Loading, Tag } from "../components/ui";
import { api, type ChapterMeta, type Run } from "../lib/api";
import { chapterNumbers, groupByPart, splitPart } from "../lib/chapters";
import { formatAgo, formatMinutes } from "../lib/format";
import { doneCount, nextChapter, partNumber } from "../lib/shellData";
import Overview, { OverviewBlock, OverviewList } from "./Overview";

type Props = {
  /** Null until the course list has loaded. */
  chapters: ChapterMeta[] | null;
  progress: Record<string, string>;
  /** The GPU the labs run on, such as "H100". Empty when unknown. */
  gpu: string;
};

const RECENT_RUNS = 5;

function RunOutcome({ run }: { run: Run }) {
  if (run.passed === true) return <Tag tone="ok">Passed</Tag>;
  if (run.passed === false) return <Tag tone="bad">Failed</Tag>;
  return <Tag>{run.status.replace(/_/g, " ")}</Tag>;
}

/**
 * The learn-gpu overview at "/": the shared Overview page filled from the
 * chapter index, with the reader's recent lab runs after the part cards.
 */
export default function Dashboard({ chapters, progress, gpu }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    api.runs().then((result) => setRuns(result.runs)).catch(() => undefined);
  }, []);

  const list = useMemo(() => chapters ?? [], [chapters]);
  const numbers = useMemo(() => chapterNumbers(list), [list]);
  const byLab = useMemo(
    () => new Map(list.filter((c) => c.lab).map((c) => [c.lab as string, c])),
    [list],
  );

  if (!chapters) {
    return (
      <div className="min-h-0 flex-1">
        <Loading label="Loading the course" />
      </div>
    );
  }

  const done = doneCount(list, progress);
  const next = nextChapter(list, progress);
  const resuming = next ? Boolean(progress[next.slug]) : false;
  const labs = list.filter((c) => c.lab).length;
  // A lab counts once however many times it passed.
  const labsPassed = new Set(runs.filter((r) => r.passed).map((r) => r.lab)).size;
  const minutes = list.reduce((sum, c) => sum + (c.minutes ?? 0), 0);
  const minutesLeft = list
    .filter((c) => progress[c.slug] !== "done")
    .reduce((sum, c) => sum + (c.minutes ?? 0), 0);
  const recent = [...runs]
    .sort((a, b) => b.started_at - a.started_at)
    .filter((run) => byLab.has(run.lab))
    .slice(0, RECENT_RUNS);
  const parts = groupByPart(list);
  // Tag a chapter's lab only where labs differ. When every chapter has a GPU
  // lab, a tag on every row says nothing.
  const labKind = (chapter: ChapterMeta) => (chapter.lab ? (chapter.gpu ? "GPU lab" : "Lab") : "");
  const tagLabs = new Set(list.map(labKind)).size > 1;

  return (
    <Overview
      eyebrow="Field notes · learn/gpu"
      pill={`${list.length} chapters · ${labs} labs`}
      title={
        <>
          GPU programming <em>for ML</em>
        </>
      }
      intro={`From the streaming multiprocessor to the kernels that train and serve models. You write CUDA C++ and Triton, and each lab runs on a real GPU${gpu ? `: an ${gpu}.` : "."}`}
      primary={
        next
          ? {
              href: `/c/${next.slug}`,
              label: resuming ? `Continue: ${next.title}` : "Start the course",
            }
          : list[0]
            ? { href: `/c/${list[0].slug}`, label: "Read it again" }
            : undefined
      }
      secondary={[{ href: "/slides", label: "Slides" }]}
      next={
        next
          ? {
              href: `/c/${next.slug}`,
              kicker: `${resuming ? "Pick up where you left off" : "Start here"} · Chapter ${numbers.get(next.slug) ?? ""}`,
              title: next.title,
              summary: next.summary,
              action: resuming ? "Continue" : "Start chapter",
            }
          : undefined
      }
      progress={{
        done,
        total: list.length,
        label: "Chapters done",
        detail: [
          `${labsPassed} of ${labs} labs passed`,
          `${runs.length} lab run${runs.length === 1 ? "" : "s"}`,
          minutesLeft ? `${formatMinutes(minutesLeft)} of reading left` : "Nothing left to read",
        ].join(" · "),
      }}
      facts={[
        { icon: "book", value: String(list.length), label: "chapters" },
        { icon: "flask", value: String(labs), label: "labs" },
        { icon: "layers", value: String(parts.length), label: "parts" },
        ...(minutes > 0 ? [{ icon: "clock" as const, value: formatMinutes(minutes), label: "of reading" }] : []),
      ]}
      path={{
        eyebrow: "The course",
        title: "The course, part by part",
        aside: `${parts.length} parts · ${list.length} chapters`,
      }}
      noun="chapter"
      sections={parts.map(([part, items], index) => {
        const partMinutes = items.reduce((sum, c) => sum + (c.minutes ?? 0), 0);
        return {
          id: part,
          number: partNumber(part, index),
          title: splitPart(part).name,
          // A part's chapter numbers aren't contiguous (27 and 28 sit in
          // part 1), so the meta gives a count rather than a range.
          meta: [
            `${items.length} chapter${items.length === 1 ? "" : "s"}`,
            partMinutes > 0 ? formatMinutes(partMinutes) : "",
          ]
            .filter(Boolean)
            .join(" · "),
          lessons: items.map((chapter) => ({
            id: chapter.slug,
            href: `/c/${chapter.slug}`,
            title: chapter.title,
            number: numbers.get(chapter.slug),
            done: progress[chapter.slug] === "done",
            tag: tagLabs ? labKind(chapter) || undefined : undefined,
          })),
        };
      })}
    >
      {recent.length > 0 && (
        <OverviewBlock
          eyebrow="Activity"
          title="Recent lab runs"
          aside={`${runs.length} in total`}
          id="sh-runs-heading"
        >
          <OverviewList
            items={recent.map((run) => {
              const chapter = byLab.get(run.lab)!;
              const seconds =
                run.finished_at !== null ? Math.round(run.finished_at - run.started_at) : null;
              return {
                id: run.id,
                href: `/c/${chapter.slug}`,
                kicker: [formatAgo(run.started_at), run.provider, seconds !== null ? `${seconds}s` : ""]
                  .filter(Boolean)
                  .join(" · "),
                title: chapter.title,
                trailing: <RunOutcome run={run} />,
              };
            })}
          />
        </OverviewBlock>
      )}
    </Overview>
  );
}
