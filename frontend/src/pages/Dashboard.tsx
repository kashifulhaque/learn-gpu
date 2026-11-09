import { useEffect, useMemo, useState, type CSSProperties } from "react";
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

  const parts = groupByPart(list);

  return (
    <div className="mx-auto max-w-[1320px] px-4 pb-16 pt-8 sm:px-8 lg:px-12 lg:pt-12">
      <section className="hero">
        <div className="min-w-0">
          <p className="kicker mb-4">
            Course · {list.length} chapters · {labs} labs
          </p>
          <h1>
            GPU programming <em>for ML</em>
          </h1>
          <p className="lede">
            From the streaming multiprocessor to the kernels that train and serve models. You write
            CUDA C++ and Triton, and each lab runs on a real GPU
            {gpu ? `: an ${gpu}.` : "."}
          </p>
          <div className="hero-actions">
            {next ? (
              <Link to={`/c/${next.slug}`} className="button button-primary">
                {progress[next.slug] ? "Continue" : "Start the course"}
                <ArrowRightIcon className="size-3.5" />
              </Link>
            ) : (
              <Link to={`/c/${list[0]?.slug ?? ""}`} className="button button-primary">
                Read it again
              </Link>
            )}
            <Link to="/slides" className="button">
              Slides
            </Link>
          </div>
          <p className="hero-meta">
            <span>{done.length} of {list.length} chapters done</span>
            <span>{labsPassed} of {labs} labs passed</span>
            <span>{minutesLeft ? `${formatMinutes(minutesLeft)} of reading left` : "Nothing left to read"}</span>
          </p>
        </div>

        <div className="min-w-0">
          {next ? (
            <Panel className="p-5 sm:p-6">
              <p className="kicker">
                {progress[next.slug] ? "Pick up where you left off" : "Start here"}
              </p>
              <p className="mt-3 flex items-baseline gap-2.5">
                <span className="font-mono text-[12px] text-fg-subtle">{numbers.get(next.slug)}</span>
                <span className="font-serif text-[1.6rem] font-medium leading-tight tracking-[-0.025em] text-fg">
                  {next.title}
                </span>
              </p>
              <p className="mt-2 text-[13.5px] leading-6 text-fg-muted">{next.summary}</p>
              <p className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10.5px] text-fg-subtle">
                <span>{splitPart(next.part).name}</span>
                {next.minutes && (
                  <span className="flex items-center gap-1.5">
                    <ClockIcon className="size-3.5" />
                    {next.minutes} min
                  </span>
                )}
                {next.lab && (
                  <span className="flex items-center gap-1.5">
                    <FlaskIcon className="size-3.5" />
                    {next.gpu ? "GPU lab" : "Lab"}
                  </span>
                )}
              </p>
              <Link to={`/c/${next.slug}`} className={buttonClass("primary", "md", "mt-5")}>
                {progress[next.slug] ? "Continue" : "Start chapter"}
                <ArrowRightIcon className="size-3.5" />
              </Link>
            </Panel>
          ) : (
            <Panel className="p-5 sm:p-6">
              <p className="kicker">Course complete</p>
              <p className="mt-3 font-serif text-[1.6rem] font-medium tracking-[-0.025em] text-fg">
                You've worked through every chapter.
              </p>
              <p className="mt-2 text-[13.5px] leading-6 text-fg-muted">
                The labs stay open, so you can go back and push any of them further.
              </p>
            </Panel>
          )}

          <dl className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-[7px] border border-line bg-line [&>div]:bg-surface">
            {stats.map((stat) => (
              <div key={stat.label} className="px-4 py-3">
                <dt className="font-mono text-[10px] uppercase tracking-[0.06em] text-fg-subtle">
                  {stat.label}
                </dt>
                <dd className="mt-1 flex items-baseline gap-1.5">
                  <span className="font-serif text-[1.5rem] font-medium tabular-nums tracking-[-0.03em] text-fg">
                    {stat.value}
                  </span>
                  {stat.of && <span className="text-[12.5px] tabular-nums text-fg-subtle">{stat.of}</span>}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <div className="section-heading">
        <div>
          <p className="kicker mb-2">The course</p>
          <h2>Modules</h2>
        </div>
        <p>
          {parts.length} parts · {list.length} chapters
        </p>
      </div>

      <div className="module-grid">
        {parts.map(([part, items], partIndex) => {
          const { label, name } = splitPart(part);
          const partDone = items.filter((c) => progress[c.slug] === "done").length;
          const partNext = items.find((c) => progress[c.slug] !== "done") ?? items[0];
          const minutes = items.reduce((sum, c) => sum + (c.minutes ?? 0), 0);
          const percent = items.length ? Math.round((partDone / items.length) * 100) : 0;
          return (
            <article
              key={part}
              className="module-card"
              style={{ "--progress": `${percent}%` } as CSSProperties}
            >
              <div className="module-card-header">
                <span className="module-number">{label || `Part ${partIndex + 1}`}</span>
                <span className="module-icon">{String(partIndex + 1).padStart(2, "0")}</span>
              </div>
              <h3>
                <Link to={`/c/${partNext.slug}`} className="hover:text-accent">
                  {name}
                </Link>
              </h3>
              <ol className="mb-5 space-y-1 text-[13px] leading-5">
                {items.map((chapter) => {
                  const status = progress[chapter.slug];
                  return (
                    <li key={chapter.slug}>
                      <Link
                        to={`/c/${chapter.slug}`}
                        className="group flex items-baseline gap-2.5 text-fg-muted hover:text-accent"
                      >
                        <span
                          className={`w-6 shrink-0 font-mono text-[10.5px] tabular-nums ${
                            status === "done" ? "text-accent" : "text-fg-subtle"
                          }`}
                        >
                          {status === "done" ? (
                            <CheckIcon className="inline size-3" aria-label="Done" />
                          ) : (
                            numbers.get(chapter.slug)
                          )}
                        </span>
                        <span className="min-w-0 flex-1">
                          {chapter.title}
                          {status === "in_progress" && (
                            <span className="ml-1.5 inline-flex translate-y-[-1px] align-middle">
                              <Dot tone="accent" />
                              <span className="sr-only">In progress</span>
                            </span>
                          )}
                        </span>
                        {chapter.lab && (
                          <span title={labTitle(chapter)} className="text-fg-faint">
                            <FlaskIcon className="size-3" />
                          </span>
                        )}
                      </Link>
                    </li>
                  );
                })}
              </ol>
              <div className="module-card-footer">
                <span className="module-count">
                  {partDone}/{items.length} done{minutes ? ` · ${formatMinutes(minutes)}` : ""}
                </span>
                <Link
                  to={`/c/${partNext.slug}`}
                  className="flex items-center gap-1 text-[12.5px] font-semibold text-accent hover:underline"
                >
                  {partDone === items.length ? "Review" : partDone || progress[partNext.slug] ? "Continue" : "Start"}
                  <ArrowRightIcon className="size-3.5" />
                </Link>
              </div>
              <div
                className="module-progress"
                role="progressbar"
                aria-label={`${name}: chapters done`}
                aria-valuenow={partDone}
                aria-valuemin={0}
                aria-valuemax={items.length}
              />
            </article>
          );
        })}
      </div>

      {recent.length > 0 && (
        <section className="mt-12">
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
                  <span className="hidden font-mono text-[12px] text-fg-subtle sm:inline">
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
                  className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-surface-muted"
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
    </div>
  );
}
