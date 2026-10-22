import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  api,
  type Infra,
  type Instance,
  type ProviderInfra,
} from "../lib/api";
import { formatAge } from "../lib/format";
import { BlockHeading, Button, Dot, Notice, Panel, Tag, buttonClass } from "../components/ui";
import { ExternalIcon, RefreshIcon } from "../components/icons";

const REFRESH_MS = 20_000;

const PROVIDER_LABELS: Record<string, string> = { runpod: "RunPod" };

export default function Compute() {
  const [infra, setInfra] = useState<Infra | null>(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [, setClock] = useState(0);

  const load = useCallback(async () => {
    try {
      setInfra(await api.infra());
      setError("");
    } catch (problem) {
      setError(String((problem as Error).message ?? problem));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while the tab is visible, so a forgotten GPU shows up without a click.
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) load();
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [live, load]);

  // Re-render once a second so the "up for" clocks keep counting.
  useEffect(() => {
    const timer = window.setInterval(() => setClock((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  async function act(
    provider: string,
    action: string,
    target: string,
    confirmText?: string,
  ) {
    if (confirmText && !window.confirm(confirmText)) return;
    const key = `${provider}:${action}:${target}`;
    setBusy(key);
    setMessage("");
    try {
      const result = await api.infraAction(provider, action, target);
      setMessage(result.message);
      await load();
    } catch (problem) {
      setError(String((problem as Error).message ?? problem));
    } finally {
      setBusy(null);
    }
  }

  const active = infra?.active ?? 0;
  const facts = (infra?.providers ?? []).flatMap((provider) =>
    provider.facts
      .filter((fact) => ["Balance", "Burn rate", "Spent, 24h"].includes(fact.label))
      .map((fact) => ({ ...fact, provider: provider.name })),
  );

  // The first /api/infra call can take many seconds, so the title waits for it
  // rather than claiming nothing is running.
  let status: ReactNode;
  if (infra) {
    status = (
      <>
        <Dot tone={active ? "bad" : "neutral"} pulse={active > 0} />
        {active ? `${active} thing${active === 1 ? "" : "s"} running` : "Nothing is running"}
      </>
    );
  } else if (loading) {
    status = (
      <>
        <span
          className="size-4 flex-none animate-spin rounded-full border-[1.5px] border-line-strong border-t-fg-subtle"
          aria-hidden
        />
        Checking the providers…
      </>
    );
  } else {
    status = (
      <>
        <Dot tone="bad" />
        Couldn’t reach the providers
      </>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-8 lg:py-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 max-w-2xl">
          <h1
            className="flex items-center gap-2.5 text-2xl font-semibold tracking-tight text-fg"
            aria-live="polite"
          >
            {status}
          </h1>
          <p className="mt-2 text-[14px] leading-6 text-fg-muted">
            Every pod and worker pool this app can start. Stopping something doesn’t delete
            it: the RunPod endpoint stays deployed and costs nothing while idle.
          </p>
        </div>

        <div className="flex flex-none items-center gap-3">
          <label className="flex cursor-pointer select-none items-center gap-2 text-[13px] text-fg-muted">
            <input
              type="checkbox"
              checked={live}
              onChange={(event) => setLive(event.target.checked)}
              className="size-3.5 accent-[var(--accent)]"
            />
            Auto-refresh
          </label>
          <Button onClick={load}>
            <RefreshIcon className="size-3.5" />
            Refresh
          </Button>
        </div>
      </header>

      {infra && (
        <p className="mt-3 text-[12px] tabular-nums text-fg-faint">
          Updated {formatAge((Date.now() - infra.generated_at * 1000) / 1000)} ago
          {live ? ` · refreshing every ${REFRESH_MS / 1000}s` : " · auto-refresh off"}
        </p>
      )}

      {facts.length > 0 && (
        <CellGrid className="mt-6 grid-cols-2 sm:grid-cols-4">
          <div className="border-b border-r border-line px-4 py-3">
            <div className="text-[12px] text-fg-subtle">Active now</div>
            <div
              className={`mt-0.5 text-lg font-semibold tabular-nums ${
                active ? "text-bad" : "text-fg"
              }`}
            >
              {active}
            </div>
            <div className="text-[12px] text-fg-faint">Containers, pods, workers</div>
          </div>
          {facts.map((fact) => (
            <div
              key={`${fact.provider}-${fact.label}`}
              className="min-w-0 border-b border-r border-line px-4 py-3"
            >
              <div className="truncate text-[12px] text-fg-subtle">{fact.label}</div>
              <div className="mt-0.5 truncate text-lg font-semibold tabular-nums text-fg">
                {fact.value}
              </div>
              <div className="text-[12px] text-fg-faint">
                {PROVIDER_LABELS[fact.provider] ?? fact.provider}
              </div>
            </div>
          ))}
        </CellGrid>
      )}

      {message && (
        <Notice tone="ok" className="mt-6">
          {message}
        </Notice>
      )}
      {error && (
        <Notice tone="bad" className="mt-6 break-words font-mono">
          {error}
        </Notice>
      )}

      {infra && infra.runs.length > 0 && (
        <Panel className="mt-8 p-4 sm:p-5">
          <h2 className="text-[15px] font-semibold text-warn">Runs this app never saw finish</h2>
          <p className="mt-1 max-w-2xl text-[13px] leading-5 text-fg-muted">
            A run stays here when the browser tab closed mid-stream. The GPU job may still be
            going.
          </p>
          <ul className="mt-4 divide-y divide-line rounded-md border border-line">
            {infra.runs.map((run) => (
              <li
                key={run.id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-3 py-2.5"
              >
                <Dot tone="warn" pulse />
                <span className="text-[13.5px] font-medium text-fg">{run.lab}</span>
                <span className="min-w-0 break-all font-mono text-[12px] tabular-nums text-fg-subtle">
                  {run.provider} · {run.user} · {formatAge(run.age)}
                </span>
                <span className="min-w-0 flex-1 basis-40 text-[12px] text-fg-subtle">
                  {run.hint}
                </span>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() =>
                    act(
                      run.provider,
                      "cancel-run",
                      run.id,
                      "Cancel this run? Any job still on the GPU is cancelled too.",
                    )
                  }
                  disabled={busy !== null}
                >
                  {busy === `${run.provider}:cancel-run:${run.id}` ? "Working…" : "Cancel run"}
                </Button>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {infra?.providers.map((provider) => (
        <ProviderSection
          key={provider.name}
          provider={provider}
          busy={busy}
          onAct={act}
        />
      ))}
    </div>
  );
}

/**
 * A bordered grid of label-over-value cells. Each cell draws its own right and
 * bottom rule, and the outer box clips the last ones, so a row that is not full
 * shows card, not a block of divider colour.
 */
function CellGrid({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-card">
      <div className={`-mb-px -mr-px grid ${className}`}>{children}</div>
    </div>
  );
}

type SectionProps = {
  provider: ProviderInfra;
  busy: string | null;
  onAct: (
    provider: string,
    action: string,
    target: string,
    confirmText?: string,
  ) => void;
};

function ProviderSection({ provider, busy, onAct }: SectionProps) {
  const label = PROVIDER_LABELS[provider.name] ?? provider.name;

  return (
    <Panel className="mt-8 overflow-hidden">
      <header className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3 sm:px-5">
        <h2 className="mr-1 text-[15px] font-semibold text-fg">{label}</h2>
        {provider.default && <Tag tone="accent">Default</Tag>}
        {provider.available ? <Tag tone="ok">Configured</Tag> : <Tag>Unavailable</Tag>}
        {provider.console_url && (
          <a
            href={provider.console_url}
            target="_blank"
            rel="noreferrer"
            className={buttonClass("ghost", "sm", "-mr-1.5 ml-auto")}
          >
            Console
            <ExternalIcon className="size-3.5" />
          </a>
        )}
      </header>

      {!provider.available && (
        <p className="border-b border-line bg-well px-4 py-2.5 text-[13px] leading-5 text-fg-muted sm:px-5">
          {provider.reason}
        </p>
      )}

      {provider.notices.map((notice) => (
        <p
          key={notice}
          className="flex gap-2.5 border-b border-line bg-warn/6 px-4 py-2.5 text-[13px] leading-5 text-fg-muted sm:px-5"
        >
          <span className="mt-1.5">
            <Dot tone="warn" />
          </span>
          <span className="min-w-0">{notice}</span>
        </p>
      ))}

      {provider.facts.length > 0 && (
        // The panel's own border frames this grid, so it only needs a bottom rule.
        <div className="overflow-hidden border-b border-line">
          <div className="-mb-px -mr-px grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
            {provider.facts.map((fact) => (
              <div key={fact.label} className="min-w-0 border-b border-r border-line px-4 py-2.5 sm:px-5">
                <div className="truncate text-[12px] text-fg-subtle">{fact.label}</div>
                <div className="mt-0.5 truncate font-mono text-[12.5px] tabular-nums text-fg">
                  {fact.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="px-4 py-4 sm:px-5">
        <BlockHeading as="h3" aside={provider.instances.length || null}>
          Instances
        </BlockHeading>
        {provider.instances.length === 0 ? (
          <p className="rounded-md border border-dashed border-line px-4 py-6 text-center text-[13px] text-fg-subtle">
            Nothing running on {label}.
          </p>
        ) : (
          <ul className="divide-y divide-line rounded-md border border-line">
            {provider.instances.map((instance) => (
              <InstanceRow
                key={`${instance.kind}-${instance.id}`}
                instance={instance}
                busy={busy}
                onAct={onAct}
              />
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

function InstanceRow({
  instance,
  busy,
  onAct,
}: {
  instance: Instance;
  busy: string | null;
  onAct: SectionProps["onAct"];
}) {
  const uptime = instance.started_at
    ? formatAge(Date.now() / 1000 - instance.started_at)
    : null;

  return (
    <li className="px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <Dot tone={instance.active ? "bad" : "neutral"} pulse={instance.active} />
        <span className="min-w-0 break-words text-[13.5px] font-medium text-fg">
          {instance.label}
        </span>
        <Tag mono>{instance.kind}</Tag>
        <span className="text-[12.5px] text-fg-muted">{instance.status}</span>
        {uptime && (
          <span className="text-[12px] text-fg-subtle">
            {instance.age_label}{" "}
            <span className="font-mono tabular-nums">{uptime}</span>
          </span>
        )}
        {instance.actions.length > 0 && (
          <div className="ml-auto flex flex-wrap gap-2">
            {instance.actions.map((action) => (
              <Button
                key={action.action}
                variant="danger"
                size="sm"
                onClick={() =>
                  onAct(instance.provider, action.action, instance.id, action.confirm)
                }
                disabled={busy !== null}
              >
                {busy === `${instance.provider}:${action.action}:${instance.id}`
                  ? "Working…"
                  : action.label}
              </Button>
            ))}
          </div>
        )}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-5 text-[12px] leading-5 text-fg-subtle">
        <span className="min-w-0 break-all font-mono text-fg-faint">{instance.id}</span>
        {instance.gpu && <span className="font-mono text-fg-muted">{instance.gpu}</span>}
        <span className="min-w-0">{instance.detail}</span>
      </div>
    </li>
  );
}
