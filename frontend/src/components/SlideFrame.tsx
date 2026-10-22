import { useLayoutEffect, useRef, type RefObject } from "react";

import Markdown, { type MarkdownVariant } from "./Markdown";
import type { Slide } from "../lib/api";

/** The smallest the type shrinks to fit a slide, before the slide scrolls instead. */
const MIN_FIT = 0.62;
const FIT_STEP = 0.05;

/** The event a late-drawing block, such as a diagram, sends up when its size changes. */
export const CONTENT_CHANGE = "lg:contentchange";

/**
 * Shrinks a slide's type until its content fits the 16:9 stage, down to
 * `MIN_FIT` of the normal size; past that the frame scrolls. It measures
 * again whenever the frame or its content changes size, such as when a
 * diagram finishes drawing or the window is resized.
 */
function useAutoFit(frame: RefObject<HTMLDivElement | null>, key: string) {
  useLayoutEffect(() => {
    const node = frame.current;
    if (!node) return;
    let timer = 0;
    const fit = () => {
      let scale = 1;
      node.style.setProperty("--fit", "1");
      while (node.scrollHeight > node.clientHeight + 1 && scale > MIN_FIT) {
        scale = Math.max(MIN_FIT, Math.round((scale - FIT_STEP) * 100) / 100);
        node.style.setProperty("--fit", String(scale));
      }
    };
    // A timeout rather than an animation frame, so a slide fits in a tab that
    // isn't painting, such as a notes window behind the stage.
    const schedule = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(fit, 0);
    };
    fit();
    // Content that draws late, such as a diagram, says so with this event,
    // because a tab that isn't painting delivers no resize observations.
    node.addEventListener(CONTENT_CHANGE, schedule);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    observer?.observe(node);
    for (const child of Array.from(node.children)) observer?.observe(child);
    return () => {
      window.clearTimeout(timer);
      node.removeEventListener(CONTENT_CHANGE, schedule);
      observer?.disconnect();
    };
  }, [frame, key]);
}

/**
 * One slide on a 16:9 stage.
 *
 * - The title slide sits in the middle, with the deck's title and subtitle.
 * - A section divider sits in the middle a size up, with its body below.
 * - A content slide has its section and subsection above the title.
 *
 * Sizes are in container units, so the slide looks the same on a thumbnail, a
 * projector, and the printed page. The body is chapter Markdown: mathematics,
 * callouts, tables, Mermaid diagrams, and `viz` figures.
 */
export default function SlideFrame({
  slide,
  variant = "slide",
  frame,
}: {
  slide: Pick<Slide, "id" | "kind" | "title" | "body"> & Partial<Pick<Slide, "section" | "subsection">>;
  variant?: MarkdownVariant;
  /** The scrolling frame, so the caller can scroll it back to the top. */
  frame?: RefObject<HTMLDivElement | null>;
}) {
  const own = useRef<HTMLDivElement>(null);
  const ref = frame ?? own;
  useAutoFit(ref, `${slide.id}\n${slide.body}`);

  const eyebrow =
    slide.kind === "title"
      ? null
      : slide.kind === "section"
        ? "Section"
        : [slide.section, slide.subsection].filter(Boolean).join(" · ");

  return (
    <div ref={ref} className="slide-frame" data-kind={slide.kind}>
      {eyebrow && <p className="slide-eyebrow">{eyebrow}</p>}
      <h2 className="slide-title">
        <Markdown variant="inline">{slide.title}</Markdown>
      </h2>
      {slide.body?.trim() && (
        <div className="slide-body">
          <Markdown variant={variant}>{slide.body}</Markdown>
        </div>
      )}
    </div>
  );
}
