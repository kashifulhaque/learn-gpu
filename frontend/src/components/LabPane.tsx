import { lazy, Suspense, useCallback, useEffect, useId, useRef, useState } from "react";
import { api, runLab, type Lab, type ProviderInfo, type RunEvent } from "../lib/api";
import { useEditorLook } from "../lib/editorPrefs";
import { useTheme } from "../lib/theme";
import { OutputHeader, useOutputCollapse } from "./CollapsibleOutput";
import EditorToolbar, { ToolbarPanel, type ToolbarAction } from "./EditorToolbar";
import { CheckIcon, CloseIcon } from "./icons";
import { modKey } from "./shellIcons";
import SplitPane from "./SplitPane";
import { useIsWide } from "../lib/useMediaQuery";
import { Button, Kbd, Notice, Tag } from "./ui";

// Monaco is heavy, so the editor is its own chunk and loads with the first lab.
const CodeEditor = lazy(() => import("./CodeEditor"));

/** "wall_seconds" reads as "Wall seconds". */
function metricLabel(key: string): string {
  const words = key.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

type Check = { name: string; passed: boolean; detail: string };
type SaveState = "idle" | "dirty" | "saving" | "saved" | "failed";
type Status = "idle" | "running" | "passed" | "failed" | "error";

type Props = {
  lab: Lab;
  onPassed: () => void;
  /** Folds the lab away so the chapter takes the whole width. */
  onCollapse?: () => void;
};

const SAVE_LABEL: Record<SaveState, string> = {
  idle: "",
  dirty: "Unsaved",
  saving: "Saving…",
  saved: "Saved",
  failed: "Not saved",
};

export default function LabPane({ lab, onPassed, onCollapse }: Props) {
  const [code, setCode] = useState(lab.draft || lab.starter);
  const [logs, setLogs] = useState<string[]>([]);
  const [checks, setChecks] = useState<Check[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [ranOn, setRanOn] = useState("");
  const [creditWarning, setCreditWarning] = useState("");
  const [unavailable, setUnavailable] = useState("");
  const [solution, setSolution] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>(lab.draft ? "saved" : "idle");
  // Which help panel is open under the toolbar. Hints and the solution are
  // opened on purpose, one step at a time, so a learner takes only as much
  // help as they need.
  const [panel, setPanel] = useState<"hints" | "solution" | null>(null);
  const [revealed, setRevealed] = useState(0);
  const [solutionShown, setSolutionShown] = useState(false);
  // Counts runs, so the output panel opens when one starts.
  const [runId, setRunId] = useState(0);
  const labId = useRef(lab.id);
  labId.current = lab.id;

  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimer = useRef<number | null>(null);
  const wide = useIsWide();
  const look = useEditorLook(useTheme());
  const output = useOutputCollapse(runId);
  const outputId = useId();
  const panelId = useId();
  const mod = modKey();

  useEffect(() => {
    setCode(lab.draft || lab.starter);
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setStatus("idle");
    setMessage("");
    setRanOn("");
    setUnavailable("");
    setSolution("");
    setSaveState(lab.draft ? "saved" : "idle");
  }, [lab.id, lab.draft, lab.starter]);

  useEffect(() => {
    setPanel(null);
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
    setRanOn("");
    setCreditWarning("");
    setUnavailable("");
    setStatus("running");
    setElapsed(0);
    setRunId((n) => n + 1);

    const controller = new AbortController();
    abortRef.current = controller;

    const handle = (event: RunEvent) => {
      switch (event.type) {
        case "start":
          setRanOn(event.provider);
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
          if (event.passed) onPassed();
          break;
        case "out_of_credits":
          setStatus("error");
          setCreditWarning(event.hint);
          setMessage(event.message);
          break;
        case "unavailable":
          // The GPU endpoint is missing. CPU checks still stream after this.
          setUnavailable(event.message);
          setLogs((lines) => [...lines, event.message]);
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

  // The solution panel only explains itself at first; the code is fetched and
  // shown once the learner asks for it.
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
  const running = status === "running";

  const actions: ToolbarAction[] = [
    ...(lab.hints.length > 0
      ? [
          {
            id: "hints",
            label: panel === "hints" ? "Hide the hints" : "Show a hint",
            icon: "bulb" as const,
            pressed: panel === "hints",
            controls: panelId,
            onClick: () => {
              setPanel((open) => (open === "hints" ? null : "hints"));
              setRevealed((count) => Math.max(count, 1));
            },
          },
        ]
      : []),
    {
      id: "reset",
      label: "Reset to the starter file",
      icon: "reset",
      priority: "low",
      disabled: code === lab.starter,
      onClick: resetToStarter,
    },
    {
      id: "solution",
      label: panel === "solution" ? "Hide the solution" : "Show the solution",
      icon: "key",
      priority: "low",
      pressed: panel === "solution",
      controls: panelId,
      onClick: () => setPanel((open) => (open === "solution" ? null : "solution")),
    },
  ];

  const outcome =
    status === "passed"
      ? `All ${checks.length} checks passed`
      : status === "failed"
        ? checks.length
          ? `${passedCount} of ${checks.length} checks passed`
          : "Checks failed"
        : status === "error"
          ? "Run failed"
          : message
            ? "Stopped"
            : "";

  const header = (collapsed: boolean) => (
    <OutputHeader
      label={checks.length ? "Checks" : "Output"}
      detail={running ? `${elapsed}s` : ranOn ? `on ${ranOn}` : undefined}
      status={outcome ? `${outcome}${message && status !== "error" && status !== "idle" ? ` · ${message.replace(/^Finished in /, "")}` : ""}` : undefined}
      running={running}
      collapsed={collapsed}
      onToggle={output.toggle}
      controls={outputId}
    />
  );

  const editorPane = (
    <div className="min-h-0 flex-1 bg-code">
      <Suspense
        fallback={
          <div className="flex h-full items-center justify-center bg-code text-[12.5px] text-fg-subtle">
            Loading the editor…
          </div>
        }
      >
        <CodeEditor
          path={`${lab.id}/solution.py`}
          value={code}
          onChange={handleChange}
          onRun={() => {
            if (canRun) run();
          }}
        />
      </Suspense>
    </div>
  );

  const outputPane = (
    <>
      {header(false)}
      <div
        id={outputId}
        ref={logRef}
        className="min-h-0 flex-1 overflow-y-auto bg-paper"
        data-testid="lab-output"
      >
        {checks.length > 0 && (
          <div className="border-b border-line px-4 py-3.5">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Tag tone={status === "passed" ? "ok" : "bad"} className="tabular-nums">
                {passedCount}/{checks.length} checks passed
              </Tag>
              {message && <span className="text-[12px] tabular-nums text-fg-subtle">{message}</span>}
            </div>
            <ul className="space-y-2">
              {checks.map((check) => (
                <li key={check.name} className="flex gap-2.5 text-[13px] leading-5">
                  <span
                    className={`mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full ${
                      check.passed ? "bg-ok/12 text-ok" : "bg-bad/12 text-bad"
                    }`}
                  >
                    {check.passed ? <CheckIcon className="size-3" /> : <CloseIcon className="size-3" />}
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
          </div>
        )}

        <div className="px-4 py-3 font-mono text-[12px] leading-[1.75]">
          {logs.length === 0 && status !== "running" && (
            <div className="max-w-md font-sans text-[13px] leading-5 text-fg-subtle">
              <p>
                Run the lab to send your code to a GPU. Output streams here as it runs, and the
                checks appear above it when it finishes.
              </p>
              <p className="mt-2 flex items-center gap-1.5 text-[12px] text-fg-faint">
                <Kbd>{`${mod} ↵`}</Kbd> runs it from the editor.
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
                      ? "text-fg-subtle"
                      : "whitespace-pre-wrap text-fg-muted"
              }
            >
              {line}
            </div>
          ))}
          {status === "running" && <div className="text-fg-subtle">▍ waiting on the GPU…</div>}
          {status === "error" && message && (
            <Notice tone="bad" className="mt-3 font-sans">
              {message}
            </Notice>
          )}
          {status === "idle" && message && (
            <p className="mt-2 font-sans text-[12.5px] text-fg-subtle">{message}</p>
          )}
        </div>
      </div>
    </>
  );

  return (
    <section
      className={`slab flex h-full min-h-0 flex-col ${look === "paper" ? "code-light sh-slab-paper" : ""}`}
      aria-label={`Lab: ${lab.title}`}
    >
      <EditorToolbar
        title={lab.title}
        info={
          <>
            <p>{lab.brief}</p>
            <p className="mt-2 text-[12px] text-fg-subtle">
              Runs on {lab.gpu}, with a {lab.timeout}s limit.
            </p>
          </>
        }
        meta={lab.gpu}
        status={SAVE_LABEL[saveState]}
        running={running}
        onRun={() => {
          if (canRun) run();
        }}
        onStop={cancel}
        runLabel="Run"
        runDisabled={!runnable?.available}
        actions={actions}
        onCollapse={onCollapse}
        collapseShortcut={`${mod} .`}
      >
        {/* With one provider there is nothing to pick, so the select stays
            hidden. A narrow screen marks an unconfigured provider with a sign
            instead of words; either way the reason is a line under the bar. */}
        {providers.length > 1 && (
          <select
            value={provider ?? ""}
            onChange={(event) => setProvider(event.target.value)}
            className="sh-select max-w-28!"
            aria-label="GPU provider"
            title="GPU provider"
          >
            {providers.map((item) => (
              <option key={item.name} value={item.name} disabled={!item.available}>
                {item.name}
                {item.available ? "" : wide ? " (not configured)" : " ⚠"}
              </option>
            ))}
          </select>
        )}
      </EditorToolbar>

      {runnable && !runnable.available && (
        <p className="shrink-0 border-b border-line bg-card px-3 py-1.5 text-[12px] text-fg-muted">
          {runnable.reason}
        </p>
      )}
      {unavailable && (
        <p className="shrink-0 border-b border-line bg-warn/8 px-3 py-1.5 text-[12px] text-warn" role="status">
          {unavailable}
        </p>
      )}
      {creditWarning && (
        <p className="shrink-0 border-b border-line bg-warn/8 px-3 py-1.5 text-[12px] text-warn">
          {creditWarning}
        </p>
      )}

      {panel === "hints" && (
        <ToolbarPanel id={panelId} title="Hints" onClose={() => setPanel(null)}>
          <ol className="space-y-2.5">
            {lab.hints.slice(0, revealed).map((hint, index) => (
              <li key={hint} className="flex gap-2.5 text-[13px] leading-5 text-fg-muted">
                <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded bg-tint font-mono text-[10.5px] tabular-nums text-fg-subtle">
                  {index + 1}
                </span>
                <span className="min-w-0">{hint}</span>
              </li>
            ))}
          </ol>
          {revealed < lab.hints.length ? (
            <button
              type="button"
              onClick={() => setRevealed((count) => count + 1)}
              className="mt-2.5 text-[12.5px] font-medium text-accent hover:underline"
            >
              Show hint {revealed + 1} of {lab.hints.length}
            </button>
          ) : (
            <p className="mt-2.5 text-[12px] text-fg-faint">That's every hint for this lab.</p>
          )}
        </ToolbarPanel>
      )}

      {panel === "solution" && (
        <ToolbarPanel
          id={panelId}
          title="Worked solution"
          onClose={() => setPanel(null)}
          footer={
            solution ? (
              <Button size="sm" onClick={loadSolution} disabled={code === solution} className="ml-auto">
                Load into the editor
              </Button>
            ) : undefined
          }
        >
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
            <pre className="overflow-auto rounded-md border border-line bg-card p-3.5 font-mono text-[12px] leading-relaxed text-fg">
              {solution}
            </pre>
          )}
        </ToolbarPanel>
      )}

      <SplitPane
        direction="column"
        storageKey="lg.lab.console"
        initial={62}
        min={20}
        max={88}
        className="min-h-0 flex-1"
        label="Resize the editor and the output"
        collapsed={output.collapsed}
        rail={header(true)}
        first={editorPane}
        second={outputPane}
      />
    </section>
  );
}
