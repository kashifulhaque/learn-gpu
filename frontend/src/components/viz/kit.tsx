// The parts every chapter figure is built from: a frame, a few controls, a
// readout, and theme colours for SVG. A figure is a React component with no
// props, registered under a name in its chapter's `viz/<NN>/index.ts`, and
// placed in a chapter or a slide with a ```viz fence. This file is shared by
// every chapter's figures: do not edit it while you write one. Put anything
// chapter-specific in your own folder.
//
// Figures render on screen and in the printed book. On paper the controls are
// hidden and the figure prints in its initial state, so every value a control
// sets must also appear in the drawing or the readout.

import { useEffect, useId, useRef, useState, type ReactNode } from "react";

/**
 * Theme colours for SVG. Each one is a CSS variable, so a figure follows the
 * light and dark themes without code of its own. Apply them through `style`
 * (`style={{ fill: color.a }}`) rather than the `fill` attribute.
 */
export const color = {
  fg: "var(--fg)",
  muted: "var(--fg-muted)",
  subtle: "var(--fg-subtle)",
  faint: "var(--fg-faint)",
  paper: "var(--paper)",
  well: "var(--well)",
  card: "var(--card)",
  tint: "var(--tint)",
  line: "var(--line)",
  lineStrong: "var(--line-strong)",
  accent: "var(--accent)",
  ok: "var(--ok)",
  bad: "var(--bad)",
  warn: "var(--warn)",
  info: "var(--info)",
  violet: "var(--violet)",
  // The maths colours, matching \hla to \hld in the chapter's equations.
  a: "var(--hl-a)",
  b: "var(--hl-b)",
  c: "var(--hl-c)",
  d: "var(--hl-d)",
} as const;

/** A theme colour mixed toward transparent, for fills behind a stroke. */
export function wash(tone: string, percent: number): string {
  return `color-mix(in srgb, ${tone} ${percent}%, transparent)`;
}

/** Bytes in binary units: "147.8 MiB", "2.14 GiB". */
export function formatBytes(bytes: number, digits = 1): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(index === 0 ? 0 : digits)} ${units[index]}`;
}

/** A count with thin grouping: 32768 as "32,768", or 32k when `compact`. */
export function formatCount(value: number, compact = false): string {
  if (compact && Math.abs(value) >= 1000) {
    const k = value / 1024;
    if (Number.isInteger(k)) return `${k}k`;
    return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  }
  return value.toLocaleString("en-US");
}

/**
 * The pixel width of the element `ref` points at, kept current as it resizes.
 * Draw an SVG at this width, with no viewBox scaling, so its text stays the
 * same size on a phone as on a desktop. Before the first measurement it is
 * `fallback`.
 */
export function useWidth<T extends HTMLElement = HTMLDivElement>(fallback = 560) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry.contentRect.width);
      if (next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** True when the reader asked the system for less motion. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

/**
 * Steps through `count` frames of an animation. Nothing plays until the reader
 * presses play, so a figure never moves on its own. Playback stops on the last
 * frame; pressing play again starts over.
 */
export function useStepper(count: number, intervalMs = 700) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setStep((current) => {
        if (current >= count - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer.current);
  }, [playing, count, intervalMs]);

  useEffect(() => {
    setStep((current) => Math.min(current, Math.max(0, count - 1)));
  }, [count]);

  return {
    step,
    setStep: (next: number) => {
      setPlaying(false);
      setStep(Math.max(0, Math.min(count - 1, next)));
    },
    playing,
    toggle: () => {
      if (playing) return setPlaying(false);
      if (step >= count - 1) setStep(0);
      setPlaying(true);
    },
    reset: () => {
      setPlaying(false);
      setStep(0);
    },
  };
}

type FigureProps = {
  /** A short name for the figure, in sentence case. */
  title: string;
  /** One or two sentences under the figure that say what to notice. */
  caption?: ReactNode;
  /** Sliders, segmented controls, and buttons. Hidden in the printed book. */
  controls?: ReactNode;
  /** A row of <Stat>s under the drawing. Printed. */
  readout?: ReactNode;
  children: ReactNode;
};

/** The frame every figure sits in. */
export function Figure({ title, caption, controls, readout, children }: FigureProps) {
  return (
    <figure className="viz">
      <div className="viz-head">
        <span className="viz-label">Figure</span>
        <span className="viz-title">{title}</span>
      </div>
      {controls && <div className="viz-controls">{controls}</div>}
      <div className="viz-body">{children}</div>
      {readout && <div className="viz-readout">{readout}</div>}
      {caption && <figcaption className="viz-caption">{caption}</figcaption>}
    </figure>
  );
}

type SliderProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  /** The shown value; defaults to the number itself. */
  format?: (value: number) => string;
} & (
  | { min: number; max: number; step?: number; values?: undefined }
  /** Discrete stops, for example powers of two. The slider moves by index. */
  | { values: readonly number[]; min?: undefined; max?: undefined; step?: undefined }
);

/** A labelled range input with its value beside it. */
export function Slider(props: SliderProps) {
  const id = useId();
  const { label, value, onChange, format = (v: number) => v.toLocaleString("en-US") } = props;
  const discrete = props.values;
  const index = discrete ? Math.max(0, discrete.indexOf(value)) : 0;
  return (
    <div className="viz-slider">
      <label htmlFor={id} className="viz-control-label">
        {label}
      </label>
      <input
        id={id}
        type="range"
        min={discrete ? 0 : props.min}
        max={discrete ? discrete.length - 1 : props.max}
        step={discrete ? 1 : (props.step ?? 1)}
        value={discrete ? index : value}
        onChange={(event) => {
          const raw = Number(event.target.value);
          onChange(discrete ? discrete[raw] : raw);
        }}
      />
      <output htmlFor={id} className="viz-value">
        {format(value)}
      </output>
    </div>
  );
}

type SegmentedProps<T extends string | number> = {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
};

/** A row of mutually exclusive buttons, for a handful of named choices. */
export function Segmented<T extends string | number>({ label, value, options, onChange }: SegmentedProps<T>) {
  return (
    <div className="viz-segmented" role="radiogroup" aria-label={label}>
      <span className="viz-control-label">{label}</span>
      <div className="viz-segments">
        {options.map((option) => (
          <button
            key={String(option.value)}
            type="button"
            role="radio"
            aria-checked={option.value === value}
            className={option.value === value ? "is-on" : ""}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** A small text button: play, step, reset. */
export function VizButton({
  children,
  onClick,
  pressed,
  label,
}: {
  children: ReactNode;
  onClick: () => void;
  pressed?: boolean;
  /** An accessible name, when the visible text is a symbol. */
  label?: string;
}) {
  return (
    <button type="button" className="viz-button" onClick={onClick} aria-pressed={pressed} aria-label={label}>
      {children}
    </button>
  );
}

/** Play, step back, step forward, and a step counter, for a useStepper. */
export function StepControls({
  stepper,
  count,
  describe,
}: {
  stepper: ReturnType<typeof useStepper>;
  count: number;
  /** What the current step shows, in a few words. */
  describe?: (step: number) => string;
}) {
  return (
    <div className="viz-steps">
      <VizButton onClick={stepper.toggle} pressed={stepper.playing}>
        {stepper.playing ? "Pause" : "Play"}
      </VizButton>
      <VizButton onClick={() => stepper.setStep(stepper.step - 1)} label="Previous step">
        ‹
      </VizButton>
      <VizButton onClick={() => stepper.setStep(stepper.step + 1)} label="Next step">
        ›
      </VizButton>
      <span className="viz-step-count">
        Step {stepper.step + 1} of {count}
        {describe ? ` · ${describe(stepper.step)}` : ""}
      </span>
    </div>
  );
}

/** One number in a figure's readout. */
export function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: ReactNode;
  /** A colour from `color`, to tie the number to a mark in the drawing. */
  tone?: string;
}) {
  return (
    <div className="viz-stat">
      <span className="viz-stat-label">{label}</span>
      <span className="viz-stat-value" style={tone ? { color: tone } : undefined}>
        {value}
      </span>
    </div>
  );
}

/** A colour key: a swatch and a name per series. */
export function Legend({ items }: { items: readonly { label: string; tone: string; dashed?: boolean }[] }) {
  return (
    <div className="viz-legend">
      {items.map((item) => (
        <span key={item.label} className="viz-legend-item">
          <span
            className="viz-swatch"
            style={
              item.dashed
                ? { borderTop: `2px dashed ${item.tone}`, background: "transparent", height: 0 }
                : { background: item.tone }
            }
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}
