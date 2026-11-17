import { useCallback, useEffect, useState, type FormEvent } from "react";

import { Icon } from "../components/shellIcons";
import { BlockHeading, Button, Loading, Notice, Panel } from "../components/ui";
import { api, ApiError, type McpToken, type McpTokenCreated } from "../lib/api";
import { formatAgo } from "../lib/format";

const MAX_TOKENS = 10;

function messageOf(error: unknown): string {
  if (error instanceof ApiError && error.status === 409) {
    return `You have reached the limit of ${MAX_TOKENS} tokens. Revoke one to create another.`;
  }
  return error instanceof Error && error.message ? error.message : "Something went wrong. Try again.";
}

/** A copy button that confirms for two seconds. */
function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can be blocked; the text stays selectable.
    }
  }
  return (
    <Button onClick={copy} aria-label={label}>
      <Icon name={copied ? "check" : "download"} size={14} />
      <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
    </Button>
  );
}

function Snippet({ title, text }: { title: string; text: string }) {
  return (
    <div className="mt-4">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <h3 className="text-[13px] font-semibold text-fg">{title}</h3>
        <CopyButton text={text} label={`Copy the ${title} snippet`} />
      </div>
      <pre className="m-0 overflow-x-auto rounded-md border border-line bg-well p-3.5 font-mono text-[12.5px] leading-[1.6] text-fg">
        <code>{text}</code>
      </pre>
    </div>
  );
}

function CodeLine({ text, label }: { text: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded-md border border-line bg-well px-3 py-2 font-mono text-[13px] text-fg">
        {text}
      </code>
      <CopyButton text={text} label={label} />
    </div>
  );
}

/**
 * Agent access: create and revoke the tokens that let an AI agent use the
 * course's MCP server on your behalf.
 */
export default function AgentsPage() {
  const [url, setUrl] = useState("");
  const [tokens, setTokens] = useState<McpToken[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [created, setCreated] = useState<McpTokenCreated | null>(null);
  const [revoking, setRevoking] = useState<number | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoadError(null);
    return api
      .mcpTokens()
      .then((body) => {
        setUrl(body.url);
        setTokens(body.tokens);
      })
      .catch((error) => setLoadError(messageOf(error)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function onCreate(event: FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || creating) return;
    setCreating(true);
    setFormError(null);
    try {
      const token = await api.createMcpToken(trimmed);
      setCreated(token);
      setName("");
      await load();
    } catch (error) {
      setFormError(messageOf(error));
    } finally {
      setCreating(false);
    }
  }

  async function onRevoke(token: McpToken) {
    if (!window.confirm(`Revoke "${token.name}"? Agents that use it lose access immediately.`)) return;
    setRevoking(token.id);
    setListError(null);
    try {
      await api.revokeMcpToken(token.id);
      if (created?.id === token.id) setCreated(null);
      await load();
    } catch (error) {
      setListError(messageOf(error));
    } finally {
      setRevoking(null);
    }
  }

  const atLimit = tokens.length >= MAX_TOKENS;
  const claudeCode = created
    ? `claude mcp add --transport http learn-gpu ${url} --header "Authorization: Bearer ${created.token}"`
    : "";
  const json = created
    ? JSON.stringify(
        {
          mcpServers: {
            "learn-gpu": { type: "http", url, headers: { Authorization: `Bearer ${created.token}` } },
          },
        },
        null,
        2,
      )
    : "";

  return (
    <div className="mx-auto max-w-3xl px-4 pb-16 pt-8 sm:px-8 lg:pt-10">
      <header>
        <p className="kicker mb-3">Agent access · MCP</p>
        <h1 className="font-serif text-[2rem] font-medium leading-tight tracking-[-0.035em] text-fg sm:text-[2.4rem]">
          Agent access
        </h1>
        <p className="mt-2 max-w-2xl text-[14px] leading-6 text-fg-muted">
          Connect an AI agent, such as Claude Code, Claude Desktop, or Cursor, to read chapters,
          labs, slides, and your progress, and to run labs. Each agent uses its own token, which
          you can revoke at any time.
        </p>
      </header>

      <Notice tone="warn" className="mt-6">
        A token acts as you. The <code className="font-mono">run_lab</code> tool does what the
        lab pane's Run button does, so an agent that holds a token can start paid GPU jobs on
        RunPod. Treat a token like a password.
      </Notice>

      {loadError && (
        <Notice tone="bad" className="mt-4">
          {loadError}{" "}
          <button
            type="button"
            className="font-semibold underline"
            onClick={() => {
              setLoading(true);
              void load();
            }}
          >
            Try again
          </button>
        </Notice>
      )}

      <Panel className="mt-6 p-4 sm:p-5">
        <BlockHeading>Server URL</BlockHeading>
        {loading ? <Loading /> : url ? <CodeLine text={url} label="Copy the server URL" /> : null}
      </Panel>

      <Panel className="mt-6 p-4 sm:p-5">
        <BlockHeading>Create a token</BlockHeading>
        <form onSubmit={onCreate} className="flex flex-col gap-2 sm:flex-row">
          <label className="sr-only" htmlFor="mcp-name">
            Token name
          </label>
          <input
            id="mcp-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={64}
            required
            disabled={atLimit}
            placeholder="Name, such as Claude Code on my laptop"
            className="h-9 min-w-0 flex-1 rounded-md border border-line bg-card px-3 text-[13px] text-fg placeholder:text-fg-faint"
          />
          <Button type="submit" variant="primary" disabled={creating || atLimit || !name.trim()}>
            <Icon name="plus" size={14} />
            {creating ? "Creating…" : "Create token"}
          </Button>
        </form>
        {atLimit && (
          <p className="mt-2 text-[13px] text-fg-muted">
            You have {MAX_TOKENS} tokens, the maximum. Revoke one to create another.
          </p>
        )}
        {formError && (
          <p role="alert" className="mt-2 text-[13px] text-bad">
            {formError}
          </p>
        )}

        {created && (
          <Notice tone="accent" className="mt-5">
            <div role="status">
              <h3 className="m-0 text-[14px] font-semibold text-fg">Copy your token now</h3>
              <p className="mt-1 text-[13px] leading-5">
                This is the only time the token for “{created.name}” is shown. Store it somewhere
                safe. Anyone who has it can act as you.
              </p>
              <div className="mt-3">
                <CodeLine text={created.token} label="Copy the token" />
              </div>
              <Snippet title="Claude Code" text={claudeCode} />
              <Snippet title="JSON config (Claude Desktop, Cursor, and others)" text={json} />
              <Button className="mt-4" onClick={() => setCreated(null)}>
                I have copied it
              </Button>
            </div>
          </Notice>
        )}
      </Panel>

      <section className="mt-8" aria-label="Your tokens">
        <BlockHeading aside={loading ? null : `${tokens.length} of ${MAX_TOKENS}`}>Your tokens</BlockHeading>
        {listError && (
          <p role="alert" className="mb-2 text-[13px] text-bad">
            {listError}
          </p>
        )}
        {loading ? (
          <Loading label="Loading your tokens" />
        ) : tokens.length === 0 ? (
          <p className="text-[14px] text-fg-muted">You have no tokens yet. Create one to connect an agent.</p>
        ) : (
          <Panel>
            <ul className="m-0 list-none divide-y divide-line p-0">
              {tokens.map((token) => (
                <li key={token.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <div className="truncate text-[14px] font-semibold text-fg">{token.name}</div>
                    <div className="mt-1 text-[12.5px] text-fg-subtle">
                      <code className="font-mono">{token.prefix}…</code>
                      {" · "}Created {formatAgo(token.created_at)}
                      {" · "}
                      {token.last_used_at ? `Last used ${formatAgo(token.last_used_at)}` : "Never used"}
                    </div>
                  </div>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => onRevoke(token)}
                    disabled={revoking === token.id}
                    aria-label={`Revoke ${token.name}`}
                  >
                    {revoking === token.id ? "Revoking…" : "Revoke"}
                  </Button>
                </li>
              ))}
            </ul>
          </Panel>
        )}
      </section>
    </div>
  );
}
