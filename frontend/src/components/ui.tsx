import type { ButtonHTMLAttributes, ReactNode } from "react";

// The handful of primitives every page shares. Each is a class string first,
// so a router <Link> or an <a> can wear a button's look without wrapping it.

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

const BUTTON_BASE =
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-on-accent hover:bg-accent-strong disabled:hover:bg-accent",
  secondary:
    "border border-line bg-card text-fg-muted hover:border-line-strong hover:text-fg disabled:hover:border-line disabled:hover:text-fg-muted",
  ghost: "text-fg-subtle hover:bg-tint hover:text-fg disabled:hover:bg-transparent",
  danger:
    "border border-bad/35 bg-bad/8 text-bad hover:border-bad/60 hover:bg-bad/14 disabled:hover:bg-bad/8",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-[12.5px]",
  md: "h-9 px-3.5 text-[13px]",
};

export function buttonClass(
  variant: ButtonVariant = "secondary",
  size: ButtonSize = "md",
  extra = "",
): string {
  return `${BUTTON_BASE} ${BUTTON_VARIANTS[variant]} ${BUTTON_SIZES[size]} ${extra}`;
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  type = "button",
  ...rest
}: ButtonProps) {
  return <button type={type} className={buttonClass(variant, size, className)} {...rest} />;
}

/** A square button that holds only an icon. Always give it an aria-label. */
export function IconButton({
  className = "",
  type = "button",
  size = "md",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { size?: ButtonSize }) {
  return (
    <button
      type={type}
      className={`inline-flex shrink-0 items-center justify-center rounded-md text-fg-subtle transition-colors hover:bg-tint hover:text-fg disabled:cursor-default disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-fg-subtle ${
        size === "sm" ? "size-7" : "size-9"
      } ${className}`}
      {...rest}
    />
  );
}

export type Tone = "neutral" | "accent" | "ok" | "bad" | "warn" | "info";

const TAG_TONES: Record<Tone, string> = {
  neutral: "border-line bg-well text-fg-subtle",
  accent: "border-accent/25 bg-accent/8 text-accent",
  ok: "border-ok/25 bg-ok/8 text-ok",
  bad: "border-bad/25 bg-bad/8 text-bad",
  warn: "border-warn/25 bg-warn/8 text-warn",
  info: "border-info/25 bg-info/8 text-info",
};

/** A small label for a kind, a state, or a count. */
export function Tag({
  tone = "neutral",
  mono = false,
  className = "",
  children,
  title,
}: {
  tone?: Tone;
  mono?: boolean;
  className?: string;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded border px-1.5 py-px text-[11px] font-medium leading-4 ${
        mono ? "font-mono" : ""
      } ${TAG_TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

const DOT_TONES: Record<Tone, string> = {
  neutral: "bg-fg-faint",
  accent: "bg-accent",
  ok: "bg-ok",
  bad: "bg-bad",
  warn: "bg-warn",
  info: "bg-info",
};

/** A status light. `pulse` is for something live, such as a running GPU. */
export function Dot({ tone = "neutral", pulse = false }: { tone?: Tone; pulse?: boolean }) {
  return (
    <span className="relative flex size-2 shrink-0" aria-hidden>
      {pulse && (
        <span
          className={`absolute inline-flex size-full animate-ping rounded-full opacity-50 ${DOT_TONES[tone]}`}
        />
      )}
      <span className={`relative inline-flex size-2 rounded-full ${DOT_TONES[tone]}`} />
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-[18px] min-w-[18px] items-center justify-center rounded border border-line bg-card px-1 font-mono text-[10.5px] leading-none text-fg-subtle">
      {children}
    </kbd>
  );
}

/** A heading for a block of a page, with room for a count or a control on the right. */
export function BlockHeading({
  children,
  aside,
  as: Tag = "h2",
}: {
  children: ReactNode;
  aside?: ReactNode;
  as?: "h2" | "h3";
}) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <Tag className="text-[13px] font-semibold text-fg">{children}</Tag>
      {aside && <div className="text-[12px] text-fg-subtle">{aside}</div>}
    </div>
  );
}

/** A bordered surface. Most blocks on a page are one of these. */
export function Panel({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`rounded-lg border border-line bg-card ${className}`}>{children}</div>
  );
}

/** A callout inside the app chrome: an error, a notice, or a confirmation. */
export function Notice({
  tone = "neutral",
  className = "",
  children,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
}) {
  const tones: Record<Tone, string> = {
    neutral: "border-line bg-well text-fg-muted",
    accent: "border-accent/25 bg-accent/6 text-fg-muted",
    ok: "border-ok/25 bg-ok/6 text-fg-muted",
    bad: "border-bad/30 bg-bad/6 text-bad",
    warn: "border-warn/30 bg-warn/6 text-fg-muted",
    info: "border-info/25 bg-info/6 text-fg-muted",
  };
  return (
    <div className={`rounded-md border px-3 py-2.5 text-[13px] leading-5 ${tones[tone]} ${className}`}>
      {children}
    </div>
  );
}

/** A quiet placeholder for a page or pane that is still loading. */
export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 p-8 text-[13px] text-fg-subtle" role="status">
      <span className="size-3.5 animate-spin rounded-full border-[1.5px] border-line-strong border-t-fg-subtle" />
      {label}…
    </div>
  );
}
