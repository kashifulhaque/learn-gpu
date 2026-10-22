import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api, runLab, type Lab, type ProviderInfo, type RunEvent } from "../lib/api";
import SplitPane from "./SplitPane";
import { useIsWide } from "../lib/useMediaQuery";
import { Button, Dot, Kbd, Notice, Tag } from "./ui";
import { CheckIcon, CloseIcon, FileIcon, PlayIcon, ResetIcon, StopIcon } from "./icons";

// Monaco is heavy, so the editor is its own chunk and loads with the first lab.
const CodeEditor = lazy(() => import("./CodeEditor"));

// The editor binds Cmd+Enter on a Mac and Ctrl+Enter elsewhere; the hint says
// whichever this reader will actually press.
const RUN_KEYS =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)
    ? "⌘↵"
    : "Ctrl ↵";

/** "wall_seconds" reads as "Wall seconds". */
function metricLabel(key: string): string {
  const words = key.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

type Check = { name: string; passed: boolean; detail: string };
type SaveState = "idle" | "dirty" | "saving" | "saved" | "failed";
type Tab = "console" | "checks" | "hints" | "solution";

type Props = {
  lab: Lab;
  onPassed: () => void;
};

export default function LabPane({ lab, onPassed }: Props) {
  const [code, setCode] = useState(lab.draft || lab.starter);
  const [logs, setLogs] = useState<string[]>([]);
  const [checks, setChecks] = useState<Check[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<"idle" | "running" | "passed" | "failed" | "error">(
    "idle",
  );
  const [message, setMessage] = useState("");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [creditWarning, setCreditWarning] = useState("");
  const [solution, setSolution] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>(lab.draft ? "saved" : "idle");
  const [tab, setTab] = useState<Tab>("console");
  // Hints and the solution are opened on purpose, one step at a time, so a
  // learner takes only as much help as they need.
  const [revealed, setRevealed] = useState(0);
  const [solutionShown, setSolutionShown] = useState(false);
  const labId = useRef(lab.id);
  labId.current = lab.id;

  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimer = useRef<number | null>(null);
  const wide = useIsWide();

  useEffect(() => {
    setCode(lab.draft || lab.starter);
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setStatus("idle");
    setMessage("");
    setSolution("");
    setTab("console");
    setSaveState(lab.draft ? "saved" : "idle");
  }, [lab.id, lab.draft, lab.starter]);

  useEffect(() => {
    setRevealed(0);
    setSolutionShown(false);
  }, [lab.id]);

  useEffect(() => {
    api.providers().then((result) => {
      setProviders(result.providers);
      // The configured default, unless it is not set up on this deployment, in
      // which case pick something that can actually run.
      const usable = result.providers.filter((item) => item.available);
      const preset = usable.find((item) => item.name === result.default);
      setProvider((preset ?? usable[0])?.name ?? result.default);
    });
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs]);

  useEffect(() => {
    if (status !== "running") return;
    const started = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      500,
    );
    return () => window.clearInterval(timer);
  }, [status]);

  // Autosave the draft a second after typing stops.
  const scheduleSave = useCallback(
    (next: string) => {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
      setSaveState("dirty");
      saveTimer.current = window.setTimeout(() => {
        setSaveState("saving");
        api
          .saveDraft(lab.id, next)
          .then(() => setSaveState("saved"))
          .catch(() => setSaveState("failed"));
      }, 1000);
    },
    [lab.id],
  );

  function handleChange(next: string) {
    setCode(next);
    scheduleSave(next);
  }

  function resetToStarter() {
    if (code === lab.starter) return;
    if (!window.confirm("Replace your code with the starter file? This overwrites your draft.")) {
      return;
    }
    handleChange(lab.starter);
  }

  function loadSolution() {
    if (!solution || code === solution) return;
    if (!window.confirm("Replace your code with the worked solution? This overwrites your draft.")) {
      return;
    }
    handleChange(solution);
  }

  async function run() {
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setMessage("");
    setCreditWarning("");
    setStatus("running");
    setElapsed(0);
    setTab("console");

    const controller = new AbortController();
    abortRef.current = controller;

    const handle = (event: RunEvent) => {
      switch (event.type) {
        case "start":
          setLogs((lines) => [...lines, `— running on ${event.provider} —`]);
          break;
        case "log":
          setLogs((lines) => [...lines, event.line]);
          break;
        case "result":
          setChecks(event.checks);
          setMetrics(event.metrics);
          break;
        case "done":
          setStatus(event.passed ? "passed" : "failed");
          setMessage(`Finished in ${event.seconds}s`);
          setTab("checks");
          if (event.passed) onPassed();
          break;
        case "out_of_credits":
          setStatus("error");
          setCreditWarning(event.hint);
          setMessage(event.message);
          break;
        case "error":
          setStatus("error");
          setMessage(event.message);
          break;
      }
    };

    try {
      await runLab(lab.id, code, provider, handle, controller.signal);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setStatus("error");
        setMessage(String(error));
      }
    } finally {
      abortRef.current = null;
      setStatus((current) => (current === "running" ? "idle" : current));
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setStatus("idle");
    setMessage("Cancelled. The GPU job may still be finishing.");
  }

  // The Solution tab only explains itself; the code is fetched and shown once
  // the learner asks for it.
  async function showSolution() {
    setSolutionShown(true);
    if (!solution) {
      const id = lab.id;
      const result = await api.solution(id);
      // The reader may have moved to another chapter while this loaded.
      if (labId.current === id) setSolution(result.solution);
    }
  }

  const runnable = providers.find((p) => p.name === provider);
  const canRun = status !== "running" && Boolean(runnable?.available);
  const passedCount = checks.filter((c) => c.passed).length;

  const saveLabel: Record<SaveState, string> = {
    idle: "Autosaves as you type",
    dirty: "Unsaved edits",
    saving: "Saving…",
    saved: "Draft saved",
    failed: "Could not save the draft",
  };

  const tabs: { id: Tab; label: string; badge?: string }[] = [
    { id: "console", label: "Console" },
    { id: "checks", label: "Checks", badge: checks.length ? `${passedCount}/${checks.length}` : undefined },
    { id: "hints", label: "Hints", badge: lab.hints.length ? String(lab.hints.length) : undefined },
    { id: "solution", label: "Solution" },
  ];


  const editorPane = (
    <>
      <div className="flex h-9 shrink-0 items-center gap-3 border-b border-line bg-well px-3">
        <span className="flex min-w-0 items-center gap-1.5 font-mono text-[12px] text-fg-muted">
          <FileIcon className="size-3.5 text-fg-faint" />
          solution.py
        </span>
        <span
          className={`truncate text-[12px] ${
            saveState === "failed"
              ? "text-bad"
              : saveState === "dirty"
                ? "text-fg-subtle"
                : "text-fg-faint"
          }`}
          aria-live="polite"
        >
          {saveLabel[saveState]}
        </span>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <span className="hidden items-center gap-1.5 text-[12px] text-fg-faint xl:flex">
            <Kbd>{RUN_KEYS}</Kbd>
            to run
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetToStarter}
            disabled={code === lab.starter}
            title="Replace your code with the starter file"
          >
            <ResetIcon className="size-3.5" />
            Reset
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 bg-paper">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center bg-paper text-[12.5px] text-fg-faint">
              Loading the editor…
            </div>
          }
        >
          <CodeEditor
            fill
            path={`${lab.id}/solution.py`}
            value={code}
            onChange={handleChange}
            onRun={() => {
              if (canRun) run();
            }}
          />
        </Suspense>
      </div>
    </>
  );

  const consolePane = (
    <>
      <div className="flex shrink-0 items-stretch overflow-x-auto border-y border-line bg-well">
        <div className="flex shrink-0 items-stretch px-1" role="tablist" aria-label="Lab output">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={tab === item.id}
              onClick={() => setTab(item.id)}
              className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-2.5 py-2 text-[12.5px] font-medium transition-colors sm:px-3 ${
                tab === item.id
                  ? "border-accent text-fg"
                  : "border-transparent text-fg-subtle hover:text-fg"
              }`}
            >
              {item.label}
              {item.badge && (
                <span className="rounded bg-tint px-1 font-mono text-[10.5px] leading-4 tabular-nums text-fg-subtle">
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2 whitespace-nowrap px-3 text-[12px]">
          {status === "running" && (
            <span className="flex items-center gap-1.5 font-mono tabular-nums text-fg-subtle">
              <Dot tone="accent" pulse />
              {elapsed}s
            </span>
          )}
          {status === "passed" && <span className="font-medium text-ok">All checks passed</span>}
          {status === "failed" && <span className="font-medium text-bad">Checks failed</span>}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto bg-paper">
        {tab === "console" && (
          <div
            ref={logRef}
            className="h-full overflow-y-auto px-4 py-3 font-mono text-[12px] leading-relaxed"
          >
            {logs.length === 0 && status !== "running" && (
              <div className="max-w-md font-sans text-[13px] leading-5 text-fg-subtle">
                <p>
                  Run the lab to send your code to a GPU. Output streams here as it runs, and the
                  checks open when it finishes.
                </p>
                <p className="mt-2 flex items-center gap-1.5 text-[12px] text-fg-faint">
                  <Kbd>{RUN_KEYS}</Kbd> runs it from the editor.
                </p>
              </div>
            )}
            {logs.map((line, index) => (
              <div
                key={index}
                className={
                  line.startsWith("[PASS]")
                    ? "text-ok"
                    : line.startsWith("[FAIL]")
                      ? "text-bad"
                      : line.startsWith("—")
                        ? "text-fg-faint"
                        : "whitespace-pre-wrap text-fg-muted"
                }
              >
                {line}
              </div>
            ))}
            {status === "running" && <div className="text-fg-faint">▍ waiting on the GPU…</div>}
            {status === "error" && message && (
              <Notice tone="bad" className="mt-3 font-sans">
                {message}
              </Notice>
            )}
          </div>
        )}

        {tab === "checks" && (
          <div className="px-4 py-3.5">
            {checks.length === 0 ? (
              <p className="text-[13px] text-fg-subtle">No results yet. Run the lab.</p>
            ) : (
              <>
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <Tag tone={status === "passed" ? "ok" : "bad"} className="tabular-nums">
                    {passedCount}/{checks.length} checks passed
                  </Tag>
                  {message && (
                    <span className="text-[12px] tabular-nums text-fg-subtle">{message}</span>
                  )}
                </div>
                <ul className="space-y-2">
                  {checks.map((check) => (
                    <li key={check.name} className="flex gap-2.5 text-[13px] leading-5">
                      <span
                        className={`mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full ${
                          check.passed ? "bg-ok/12 text-ok" : "bg-bad/12 text-bad"
                        }`}
                      >
                        {check.passed ? (
                          <CheckIcon className="size-3" />
                        ) : (
                          <CloseIcon className="size-3" />
                        )}
                        <span className="sr-only">{check.passed ? "Passed:" : "Failed:"}</span>
                      </span>
                      <span className="min-w-0 text-fg">
                        {check.name}
                        {check.detail && <span className="text-fg-subtle"> — {check.detail}</span>}
                      </span>
                    </li>
                  ))}
                </ul>

                {Object.keys(metrics).length > 0 && (
                  <dl className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-3">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="min-w-0 bg-card px-3 py-2">
                        <dt className="truncate text-[12px] text-fg-subtle" title={key}>
                          {metricLabel(key)}
                        </dt>
                        <dd className="mt-0.5 truncate font-mono text-[13px] tabular-nums text-fg">
                          {String(value)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                )}
              </>
            )}
          </div>
        )}

        {tab === "hints" && (
          <div className="px-4 py-3.5">
            {lab.hints.length === 0 ? (
              <p className="text-[13px] text-fg-subtle">This lab ships without hints.</p>
            ) : (
              <>
                {revealed === 0 && (
                  <p className="mb-3 max-w-md text-[13px] leading-5 text-fg-subtle">
                    Each hint is a nudge, not the answer. Open one, try again, and open the next
                    only if you're still stuck.
                  </p>
                )}
                {revealed > 0 && (
                  <ol className="mb-3 space-y-2.5">
                    {lab.hints.slice(0, revealed).map((hint, index) => (
                      <li key={hint} className="flex gap-2.5 text-[13px] leading-5 text-fg-muted">
                        <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded bg-tint font-mono text-[10.5px] tabular-nums text-fg-subtle">
                          {index + 1}
                        </span>
                        <span className="min-w-0">{hint}</span>
                      </li>
                    ))}
                  </ol>
                )}
                {revealed < lab.hints.length ? (
                  <Button size="sm" onClick={() => setRevealed((count) => count + 1)}>
                    Show hint {revealed + 1} of {lab.hints.length}
                  </Button>
                ) : (
                  <p className="text-[12px] text-fg-faint">That's every hint for this lab.</p>
                )}
              </>
            )}
          </div>
        )}

        {tab === "solution" && (
          <div className="px-4 py-3.5">
            {!solutionShown ? (
              <>
                <p className="mb-3 max-w-md text-[13px] leading-5 text-fg-subtle">
                  The worked solution passes every check. Looking at it before you've tried the lab
                  skips the part that teaches.
                </p>
                <Button size="sm" onClick={showSolution}>
                  Show the solution
                </Button>
              </>
            ) : !solution ? (
              <p className="text-[13px] text-fg-subtle">Loading the worked solution…</p>
            ) : (
              <>
                <Button
                  size="sm"
                  onClick={loadSolution}
                  disabled={code === solution}
                  className="mb-3"
                >
                  Load into the editor
                </Button>
                <pre className="overflow-auto rounded-md border border-line bg-code p-3.5 font-mono text-[12px] leading-relaxed text-fg-muted">
                  {solution}
                </pre>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );

  return (
    <section className="@container flex h-full min-h-0 flex-col bg-paper">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-well px-3 sm:gap-2.5">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h2 className="truncate text-[13px] font-semibold text-fg" title={lab.brief}>
            {lab.title}
          </h2>
          {/* Tag sets its own display, so the breakpoint lives on a wrapper. */}
          <span className="hidden shrink-0 @lg:inline-flex">
            <Tag mono>{lab.gpu}</Tag>
          </span>
        </div>

        {/* Spelling out "(not configured)" inside the closed select took 43% of
            a 375px header and truncated the lab's own title to four letters.
            A narrow screen gets a marker instead; either way the reason is a
            line of its own under this bar. An option's text cannot be styled
            per breakpoint, so this branch is in JavaScript. With one provider
            there is nothing to pick, so the select stays hidden. */}
        {providers.length > 1 && (
          <select
            value={provider ?? ""}
            onChange={(event) => setProvider(event.target.value)}
            className="h-8 min-w-0 max-w-28 shrink rounded-md border border-line bg-card px-2 text-[12.5px] text-fg-muted outline-none transition-colors hover:border-line-strong hover:text-fg focus-visible:border-accent @lg:max-w-none"
            aria-label="GPU provider"
          >
            {providers.map((item) => (
              <option key={item.name} value={item.name} disabled={!item.available}>
                {item.name}
                {item.available ? "" : wide ? " (not configured)" : " ⚠"}
              </option>
            ))}
          </select>
        )}

        {status === "running" ? (
          <Button variant="danger" size="sm" className="h-8 px-3" onClick={cancel}>
            <StopIcon className="size-3.5" />
            <span className="tabular-nums">Stop · {elapsed}s</span>
          </Button>
        ) : (
          <Button
            variant="primary"
            size="sm"
            className="h-8 px-3"
            onClick={run}
            disabled={!canRun}
            title={`Run on the GPU (${RUN_KEYS})`}
          >
            <PlayIcon className="size-3.5" />
            Run
          </Button>
        )}
      </header>

      {runnable && !runnable.available && (
        <p className="shrink-0 border-b border-line bg-well px-3 py-1.5 text-[12px] text-fg-subtle">
          {runnable.reason}
        </p>
      )}
      {creditWarning && (
        <p className="shrink-0 border-b border-line bg-warn/8 px-3 py-1.5 text-[12px] text-warn">
          {creditWarning}
        </p>
      )}

      <SplitPane
        direction="column"
        storageKey="lg.lab.console"
        initial={62}
        min={20}
        max={88}
        className="flex-1"
        label="Resize the editor and the console"
        first={editorPane}
        second={consolePane}
      />
    </section>
  );
}
