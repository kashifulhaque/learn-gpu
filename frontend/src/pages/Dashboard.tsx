import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ChapterMeta, type Run } from "../lib/api";
import { chapterNumbers, groupByPart, labTitle, splitPart } from "../lib/chapters";
import { formatAgo, formatMinutes } from "../lib/format";
import { ArrowRightIcon, CheckIcon, ClockIcon, FlaskIcon } from "../components/icons";
import { BlockHeading, buttonClass, Dot, Loading, Panel, Tag } from "../components/ui";

type Props = {
  /** Null until the course list has loaded. */
  chapters: ChapterMeta[] | null;
  progress: Record<string, string>;
  gpu: string;
};

const RECENT_RUNS = 5;

function RunOutcome({ run }: { run: Run }) {
  if (run.passed === true) return <Tag tone="ok">Passed</Tag>;
  if (run.passed === false) return <Tag tone="bad">Failed</Tag>;
  return <Tag>{run.status.replace(/_/g, " ")}</Tag>;
}

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

  if (!chapters) return <Loading label="Loading the course" />;

  const done = list.filter((c) => progress[c.slug] === "done");
  const next = list.find((c) => progress[c.slug] !== "done");
  const labs = list.filter((c) => c.lab).length;
  // A lab counts once however many times it passed.
  const labsPassed = new Set(runs.filter((r) => r.passed).map((r) => r.lab)).size;
  const minutesLeft = list
    .filter((c) => progress[c.slug] !== "done")
    .reduce((sum, c) => sum + (c.minutes ?? 0), 0);
  const recent = [...runs].sort((a, b) => b.started_at - a.started_at).slice(0, RECENT_RUNS);

  const stats = [
    { label: "Chapters done", value: `${done.length}`, of: `of ${list.length}` },
    { label: "Labs passed", value: `${labsPassed}`, of: `of ${labs}` },
    { label: "Lab runs", value: `${runs.length}`, of: "" },
    { label: "Reading left", value: minutesLeft ? formatMinutes(minutesLeft) : "None", of: "" },
  ];

  return (
    <div className="mx-auto max-w-4xl px-4 pb-16 pt-8 sm:px-8 lg:pt-14">
      <header className="max-w-2xl">
        <h1 className="font-serif text-[2rem] font-semibold leading-[1.15] tracking-[-0.01em] text-fg sm:text-[2.4rem]">
          GPU programming for ML
        </h1>
        <p className="mt-3 text-[15px] leading-7 text-fg-muted">
          From the streaming multiprocessor to the kernels that train and serve models. You write
          CUDA C++ and Triton, and each lab runs on a real GPU
          {gpu ? `: an ${gpu}.` : "."}
        </p>
      </header>

      {next ? (
        <Panel className="mt-8 flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:gap-6 sm:p-6">
          <div className="min-w-0 flex-1">
            <p className="text-[12.5px] font-medium text-accent">
              {progress[next.slug] ? "Pick up where you left off" : "Start here"}
            </p>
            <p className="mt-1.5 flex items-baseline gap-2.5">
              <span className="font-mono text-[13px] text-fg-faint">{numbers.get(next.slug)}</span>
              <span className="font-serif text-[1.35rem] font-semibold leading-snug text-fg">
                {next.title}
              </span>
            </p>
            <p className="mt-1.5 text-[13.5px] leading-6 text-fg-subtle">{next.summary}</p>
            <p className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px] text-fg-subtle">
              <span>{splitPart(next.part).name}</span>
              {next.minutes && (
                <span className="flex items-center gap-1.5">
                  <ClockIcon className="size-3.5 text-fg-faint" />
                  {next.minutes} min
                </span>
              )}
              {next.lab && (
                <span className="flex items-center gap-1.5">
                  <FlaskIcon className="size-3.5 text-fg-faint" />
                  {next.gpu ? "GPU lab" : "Lab"}
                </span>
              )}
            </p>
          </div>
          <Link to={`/c/${next.slug}`} className={buttonClass("primary", "md", "self-start sm:self-center")}>
            {progress[next.slug] ? "Continue" : "Start chapter"}
            <ArrowRightIcon className="size-3.5" />
          </Link>
        </Panel>
      ) : (
        <Panel className="mt-8 p-5 sm:p-6">
          <p className="text-[12.5px] font-medium text-ok">Course complete</p>
          <p className="mt-1.5 font-serif text-[1.35rem] font-semibold text-fg">
            You've worked through every chapter.
          </p>
          <p className="mt-1.5 text-[13.5px] leading-6 text-fg-subtle">
            The labs stay open, so you can go back and push any of them further.
          </p>
        </Panel>
      )}

      <dl className="mt-4 grid grid-cols-2 overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4 [&>div]:bg-card gap-px">
        {stats.map((stat) => (
          <div key={stat.label} className="px-4 py-3.5">
            <dt className="text-[12px] text-fg-subtle">{stat.label}</dt>
            <dd className="mt-1 flex items-baseline gap-1.5">
              <span className="text-xl font-semibold tabular-nums tracking-tight text-fg">
                {stat.value}
              </span>
              {stat.of && <span className="text-[12.5px] tabular-nums text-fg-faint">{stat.of}</span>}
            </dd>
          </div>
        ))}
      </dl>

      {recent.length > 0 && (
        <section className="mt-10">
          <BlockHeading aside={`${runs.length} in total`}>Recent lab runs</BlockHeading>
          <Panel className="divide-y divide-line overflow-hidden">
            {recent.map((run) => {
              const chapter = byLab.get(run.lab);
              const seconds =
                run.finished_at !== null ? Math.round(run.finished_at - run.started_at) : null;
              const body = (
                <>
                  <RunOutcome run={run} />
                  <span className="min-w-0 flex-1 truncate text-[13px] text-fg">
                    {chapter ? chapter.title : run.lab}
                  </span>
                  <span className="hidden font-mono text-[12px] text-fg-faint sm:inline">
                    {run.provider}
                  </span>
                  {seconds !== null && (
                    <span className="hidden w-12 text-right font-mono text-[12px] tabular-nums text-fg-subtle sm:inline">
                      {seconds}s
                    </span>
                  )}
                  <span className="w-20 shrink-0 text-right text-[12px] text-fg-subtle">
                    {formatAgo(run.started_at)}
                  </span>
                </>
              );
              return chapter ? (
                <Link
                  key={run.id}
                  to={`/c/${chapter.slug}`}
                  className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-tint/60"
                >
                  {body}
                </Link>
              ) : (
                <div key={run.id} className="flex items-center gap-3 px-4 py-2.5">
                  {body}
                </div>
              );
            })}
          </Panel>
        </section>
      )}

      <section className="mt-12">
        <BlockHeading aside={`${list.length} chapters`}>Contents</BlockHeading>
        <div className="space-y-8">
          {groupByPart(list).map(([part, items]) => {
            const { label, name } = splitPart(part);
            const partDone = items.filter((c) => progress[c.slug] === "done").length;
            return (
              <div key={part}>
                <div className="flex items-baseline gap-3 border-b border-line pb-2">
                  {label && <span className="font-mono text-[12px] text-fg-faint">{label}</span>}
                  <h3 className="text-[15px] font-semibold text-fg">{name}</h3>
                  <span className="ml-auto text-[12px] tabular-nums text-fg-subtle">
                    {partDone} of {items.length} done
                  </span>
                </div>
                <ol>
                  {items.map((chapter) => {
                    const status = progress[chapter.slug];
                    return (
                      <li key={chapter.slug} className="border-b border-line/70 last:border-b-0">
                        <Link
                          to={`/c/${chapter.slug}`}
                          className="group -mx-2 flex items-start gap-3 rounded-md px-2 py-3 transition-colors hover:bg-tint/50 sm:gap-4"
                        >
                          <span
                            className={`mt-px flex w-8 shrink-0 justify-start font-mono text-[12.5px] tabular-nums ${
                              status === "done" ? "text-ok" : "text-fg-faint"
                            }`}
                          >
                            {status === "done" ? (
                              <CheckIcon className="mt-0.5 size-3.5" />
                            ) : (
                              numbers.get(chapter.slug)
                            )}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="flex items-center gap-2">
                              <span className="text-[14px] font-medium text-fg group-hover:text-accent">
                                {chapter.title}
                              </span>
                              {status === "in_progress" && (
                                <span className="flex items-center gap-1.5 text-[12px] text-fg-subtle">
                                  <Dot tone="accent" />
                                  <span className="hidden sm:inline">In progress</span>
                                </span>
                              )}
                            </span>
                            <span className="mt-0.5 block text-[13px] leading-5 text-fg-subtle">
                              {chapter.summary}
                            </span>
                          </span>
                          <span className="mt-0.5 hidden shrink-0 items-center gap-3 text-[12px] text-fg-faint sm:flex">
                            {chapter.lab && (
                              <span title={labTitle(chapter)}>
                                <FlaskIcon className="size-3.5" />
                              </span>
                            )}
                            {chapter.minutes && (
                              <span className="w-14 text-right tabular-nums">{chapter.minutes} min</span>
                            )}
                          </span>
                        </Link>
                      </li>
                    );
                  })}
                </ol>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
