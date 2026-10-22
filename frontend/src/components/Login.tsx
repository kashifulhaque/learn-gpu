import { useState } from "react";
import { api } from "../lib/api";
import { Mark } from "./Brand";
import { ArrowRightIcon } from "./icons";
import { Button, Notice } from "./ui";

const FIELD =
  "h-10 w-full rounded-md border border-line bg-card px-3 text-[14px] text-fg outline-none transition-colors placeholder:text-fg-faint focus:border-line-strong";

export default function Login({ onSignedIn }: { onSignedIn: (name: string) => void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api.login(name.trim(), password);
      onSignedIn(result.name);
    } catch (problem) {
      // A wrong password is a 401. Anything else is the server, not the reader.
      setError(
        (problem as { status?: number }).status === 401
          ? "That password isn't right."
          : "Couldn't reach the server. Try again in a moment.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full flex-col bg-paper px-5">
      <div className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center py-12">
        <div className="mb-10 flex items-center gap-2.5">
          <Mark className="size-7" />
          <span className="text-[15px] font-semibold tracking-tight text-fg">learn-gpu</span>
        </div>

        <h1 className="font-serif text-[2rem] font-semibold leading-[1.15] tracking-[-0.01em] text-fg">
          GPU programming for ML
        </h1>
        <p className="mt-3 text-[14px] leading-6 text-fg-subtle">
          From the streaming multiprocessor to the kernels that train and serve models. Each lab
          compiles and runs on a real A100.
        </p>

        <form onSubmit={submit} className="mt-9 space-y-4">
          <div>
            <label className="mb-1.5 block text-[13px] font-medium text-fg-muted" htmlFor="name">
              Name
            </label>
            <input
              id="name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
              maxLength={40}
              autoComplete="username"
              autoFocus
              className={FIELD}
            />
            <p className="mt-1.5 text-[12px] text-fg-faint">Keeps your progress and drafts separate.</p>
          </div>

          <div>
            <label className="mb-1.5 block text-[13px] font-medium text-fg-muted" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              autoComplete="current-password"
              className={FIELD}
            />
          </div>

          {error && (
            <Notice tone="bad" className="py-2">
              <span role="alert">{error}</span>
            </Notice>
          )}

          <Button
            type="submit"
            variant="primary"
            disabled={busy || !name.trim() || !password}
            className="h-10 w-full text-[14px]"
          >
            {busy ? "Signing in…" : "Sign in"}
            {!busy && <ArrowRightIcon className="size-3.5" />}
          </Button>
        </form>
      </div>
    </div>
  );
}
