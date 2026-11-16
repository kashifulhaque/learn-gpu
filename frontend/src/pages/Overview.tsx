import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { Icon, type IconName } from "../components/shellIcons";

export type OverviewLesson = {
  id: string;
  href: string;
  title: string;
  /** The number shown in the card, such as "12" or "P3". */
  number?: string;
  done: boolean;
  /** A small tag after the title, such as "Project" or "GPU". */
  tag?: string;
};

export type OverviewSection = {
  id: string;
  /** Two-digit section number, such as "01". */
  number: string;
  title: string;
  /** One short sentence about the section, taken from the course data. */
  description?: string;
  /** Quiet detail after the number, such as "11 lessons · 4 h". */
  meta?: string;
  lessons: OverviewLesson[];
};

type Props = {
  /** The mono kicker above the hero, such as "Weights and wires · Machine learning". */
  eyebrow: string;
  title: ReactNode;
  intro: ReactNode;
  /** The pill above the title. */
  pill?: string;
  primary?: { href: string; label: string };
  secondary?: { href: string; label: string }[];
  /** The "Up next" card beside the hero copy. */
  next?: { href: string; kicker: string; title: string; summary?: string; action: string };
  progress: { done: number; total: number; label?: string; detail?: string };
  facts?: { icon: IconName; value: string; label: string }[];
  path: { eyebrow: string; title: string; aside?: string };
  sections: OverviewSection[];
  /** How many lessons a card lists before "Show all". */
  previewCount?: number;
  /** What a section's items are called, for "Show all 8 lessons". Default "lesson". */
  noun?: string;
  /** More blocks after the cards, such as projects or recent activity. */
  children?: ReactNode;
};

/**
 * The course overview at "/", in the tpu reference site's style: a hero with
 * the next step and progress, a row of course facts, one card per section
 * with its lessons, then any course-specific blocks.
 *
 * Portable: the course builds every string and list from its own data.
 */
export default function Overview({
  eyebrow,
  title,
  intro,
  pill,
  primary,
  secondary = [],
  next,
  progress,
  facts = [],
  path,
  sections,
  previewCount = 5,
  noun = "lesson",
  children,
}: Props) {
  const share = progress.total ? (progress.done / progress.total) * 100 : 0;
  return (
    <div className="sh-page-scroll">
      <div className="sh-page sh-overview">
        <div className="sh-eyebrow">
          {eyebrow}
          <span className="sh-eyebrow-line" />
        </div>

        <section className="sh-hero">
          <div className="sh-hero-copy">
            {pill && <span className="sh-hero-pill">{pill}</span>}
            <h1>{title}</h1>
            <p>{intro}</p>
            <div className="sh-hero-actions">
              {primary && (
                <Link className="sh-button sh-button-primary" to={primary.href}>
                  {primary.label}
                  <Icon name="arrow" size={15} />
                </Link>
              )}
              {secondary.map((action) => (
                <Link key={action.href} className="sh-button sh-button-secondary" to={action.href}>
                  {action.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="sh-hero-side">
            <div className="sh-progress-label">
              <span>{progress.label ?? "Your progress"}</span>
              <strong>
                {progress.done} / {progress.total}
              </strong>
            </div>
            <div
              className="sh-progress-track"
              role="progressbar"
              aria-label={progress.label ?? "Your progress"}
              aria-valuemin={0}
              aria-valuemax={progress.total}
              aria-valuenow={progress.done}
            >
              <div style={{ width: `${share}%` }} />
            </div>
            {progress.detail && <p className="sh-hero-side-detail">{progress.detail}</p>}
            {next && (
              <Link to={next.href} className="sh-next-card">
                <small>{next.kicker}</small>
                <strong>{next.title}</strong>
                {next.summary && <span>{next.summary}</span>}
                <em>
                  {next.action}
                  <Icon name="arrow" size={14} />
                </em>
              </Link>
            )}
          </div>
        </section>

        {facts.length > 0 && (
          <div className="sh-facts">
            {facts.map((fact) => (
              <span key={fact.label}>
                <Icon name={fact.icon} size={15} />
                <strong>{fact.value}</strong> {fact.label}
              </span>
            ))}
          </div>
        )}

        <OverviewBlock eyebrow={path.eyebrow} title={path.title} aside={path.aside} id="sh-path-heading">
          <div className="sh-module-grid">
            {sections.map((section) => (
              <ModuleCard key={section.id} section={section} previewCount={previewCount} noun={noun} />
            ))}
          </div>
        </OverviewBlock>

        {children}
      </div>
    </div>
  );
}

function ModuleCard({ section, previewCount, noun }: { section: OverviewSection; previewCount: number; noun: string }) {
  const [expanded, setExpanded] = useState(false);
  const done = section.lessons.filter((lesson) => lesson.done).length;
  const total = section.lessons.length;
  const complete = total > 0 && done === total;
  // A long section lists its first few lessons, plus the next unfinished one
  // when it sits further down, until the reader asks for all of them.
  const nextIndex = section.lessons.findIndex((lesson) => !lesson.done);
  const shown =
    expanded || total <= previewCount + 1
      ? section.lessons
      : section.lessons.filter((_, index) => index < previewCount || index === nextIndex);
  const listId = `sh-card-${section.id.replace(/[^a-z0-9]+/gi, "-")}`;

  return (
    <article className="sh-module-card">
      <div className="sh-module-card-top">
        <span className="sh-module-card-number">{section.number}</span>
        {section.meta && <span className="sh-module-card-meta">{section.meta}</span>}
        {complete ? (
          <span className="sh-module-card-done">
            <Icon name="check" size={12} />
            Complete
          </span>
        ) : (
          <span className="sh-module-card-count">
            {done}/{total}
          </span>
        )}
      </div>
      <h3>{section.title}</h3>
      {section.description && <p>{section.description}</p>}
      <div className="sh-module-card-progress" aria-hidden="true">
        <span style={{ width: `${total ? (done / total) * 100 : 0}%` }} />
      </div>
      <div className="sh-module-card-lessons" id={listId}>
        {shown.map((lesson) => (
          <Link key={lesson.id} to={lesson.href}>
            <span className={`sh-card-index ${lesson.done ? "is-done" : ""}`}>
              {lesson.done ? <Icon name="check" size={12} /> : lesson.number}
            </span>
            <span className="sh-card-title">
              {lesson.title}
              {lesson.done && <span className="sh-sr-only"> (done)</span>}
            </span>
            {lesson.tag && <span className="sh-card-tag">{lesson.tag}</span>}
            <Icon name="arrow" size={12} />
          </Link>
        ))}
      </div>
      {shown.length < total || expanded ? (
        <button
          type="button"
          className="sh-card-more"
          aria-expanded={expanded}
          aria-controls={listId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show fewer" : `Show all ${total} ${noun}s`}
          <Icon name={expanded ? "chevronUp" : "chevronDown"} size={13} />
        </button>
      ) : null}
    </article>
  );
}

/** A titled block on the overview page, with tpu's section heading. */
export function OverviewBlock({
  eyebrow,
  title,
  aside,
  id,
  children,
}: {
  eyebrow: string;
  title: string;
  aside?: string;
  id?: string;
  children: ReactNode;
}) {
  return (
    <section className="sh-block" aria-labelledby={id}>
      <div className="sh-section-heading">
        <div>
          <span className="sh-kicker">{eyebrow}</span>
          <h2 id={id}>{title}</h2>
        </div>
        {aside && <span className="sh-section-aside">{aside}</span>}
      </div>
      {children}
    </section>
  );
}

/** Rows of links in a bordered list, for projects, recent runs, or lectures. */
export function OverviewList({
  items,
}: {
  items: { id: string; href: string; title: string; kicker?: string; detail?: string; trailing?: ReactNode }[];
}) {
  return (
    <ul className="sh-list">
      {items.map((item) => (
        <li key={item.id}>
          <Link to={item.href}>
            <span className="sh-list-text">
              {item.kicker && <small>{item.kicker}</small>}
              <strong>{item.title}</strong>
              {item.detail && <span>{item.detail}</span>}
            </span>
            {item.trailing}
            <Icon name="arrow" size={14} />
          </Link>
        </li>
      ))}
    </ul>
  );
}
