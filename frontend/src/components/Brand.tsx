import { setThemeChoice, useThemeChoice, type ThemeChoice } from "../lib/theme";
import { MonitorIcon, MoonIcon, SunIcon } from "./icons";

/** The mark from the favicon, drawn in the accent colour. */
export function Mark({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={`shrink-0 ${className}`} aria-hidden>
      <rect width="64" height="64" rx="14" className="fill-accent" />
      <path
        d="M19 15 Q27 13 30.5 20 L46.5 50"
        fill="none"
        className="stroke-on-accent"
        strokeWidth="7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M34 28 L17.5 50" fill="none" className="stroke-on-accent" strokeWidth="7" strokeLinecap="round" />
    </svg>
  );
}

const CHOICES: { value: ThemeChoice; label: string; Icon: typeof SunIcon }[] = [
  { value: "system", label: "Match the system", Icon: MonitorIcon },
  { value: "light", label: "Light", Icon: SunIcon },
  { value: "dark", label: "Dark", Icon: MoonIcon },
];

/** System, light, or dark, as a three-way switch. */
export function ThemeSwitch() {
  const choice = useThemeChoice();
  return (
    <div role="radiogroup" aria-label="Theme" className="flex rounded-md border border-line bg-paper p-0.5">
      {CHOICES.map(({ value, label, Icon }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={choice === value}
          aria-label={label}
          title={label}
          onClick={() => setThemeChoice(value)}
          className={`flex size-6 items-center justify-center rounded transition-colors ${
            choice === value ? "bg-tint text-fg" : "text-fg-faint hover:text-fg-muted"
          }`}
        >
          <Icon className="size-3.5" />
        </button>
      ))}
    </div>
  );
}

/** One button that steps through the three, for the collapsed rail. */
export function ThemeCycle() {
  const choice = useThemeChoice();
  const index = CHOICES.findIndex((item) => item.value === choice);
  const { Icon, label } = CHOICES[index];
  const next = CHOICES[(index + 1) % CHOICES.length];
  return (
    <button
      type="button"
      onClick={() => setThemeChoice(next.value)}
      title={`Theme: ${label}. Switch to ${next.label.toLowerCase()}.`}
      aria-label={`Theme: ${label}. Switch to ${next.label.toLowerCase()}.`}
      className="flex size-9 items-center justify-center rounded-md text-fg-subtle transition-colors hover:bg-tint hover:text-fg"
    >
      <Icon className="size-4" />
    </button>
  );
}
