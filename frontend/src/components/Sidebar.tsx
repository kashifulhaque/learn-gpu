import { useEffect, useMemo, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import type { ChapterMeta } from "../lib/api";
import { chapterNumbers, groupByPart, labTitle, splitPart } from "../lib/chapters";
import { useIsWide } from "../lib/useMediaQuery";
import { Mark, ThemeCycle, ThemeSwitch } from "./Brand";
import { ComputeDot, computeLabel, computeTitle } from "./ComputeBadge";
import { DownloadBookItem, DownloadBookRailItem, useBook } from "./DownloadBook";
import {
  CheckIcon,
  ChevronDownIcon,
  ChipIcon,
  CloseIcon,
  FlaskIcon,
  HomeIcon,
  PanelIcon,
  PresentIcon,
  SearchIcon,
} from "./icons";
import { IconButton, Kbd } from "./ui";

type Props = {
  chapters: ChapterMeta[];
  progress: Record<string, string>;
  /** Drawer state on a narrow screen. */
  open: boolean;
  /** Rail state on a wide screen. */
  collapsed: boolean;
  onNavigate: () => void;
  /** Shuts the narrow-screen drawer without navigating anywhere. */
  onClose: () => void;
  onToggleCollapsed: () => void;
  name: string;
  /** How many things are running on a GPU provider, or null while unknown. */
  active: number | null;
  /** Opens the "/" search dialog. */
  onSearch: () => void;
};

const SHUT_PARTS_KEY = "lg.sidebar.shutParts";

function readShutParts(): string[] {
  try {
    const raw = window.localStorage.getItem(SHUT_PARTS_KEY);
    const value = raw ? JSON.parse(raw) : [];
    return Array.isArray(value) ? value.filter((p) => typeof p === "string") : [];
  } catch {
    return [];
  }
}

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
const TOGGLE_KEY = IS_MAC ? "⌘B" : "Ctrl B";

function NavItem({
  to,
  end,
  icon,
  children,
  aside,
  onClick,
}: {
  to: string;
  end?: boolean;
  icon: ReactNode;
  children: ReactNode;
  aside?: ReactNode;
  onClick: () => void;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onClick}
      className={({ isActive }) =>
        `flex h-9 items-center gap-2.5 rounded-md px-2.5 text-[13px] font-medium transition-colors lg:h-8 ${
          isActive ? "bg-accent-soft text-accent" : "text-fg-muted hover:bg-surface-muted hover:text-fg"
        }`
      }
    >
      <span className="opacity-80">{icon}</span>
      <span className="flex-1">{children}</span>
      {aside}
    </NavLink>
  );
}

export default function Sidebar({
  chapters,
  progress,
  open,
  collapsed,
  onNavigate,
  onClose,
  onToggleCollapsed,
  name,
  active,
  onSearch,
}: Props) {
  const [filter, setFilter] = useState("");
  const [shutParts, setShutParts] = useState<string[]>(readShutParts);
  const wide = useIsWide();
  const book = useBook();

  useEffect(() => {
    try {
      window.localStorage.setItem(SHUT_PARTS_KEY, JSON.stringify(shutParts));
    } catch {
      // Losing the open/shut state is harmless.
    }
  }, [shutParts]);

  const numbers = useMemo(() => chapterNumbers(chapters), [chapters]);

  const needle = filter.trim().toLowerCase();
  const matches = needle
    ? chapters.filter(
        (chapter) =>
          chapter.title.toLowerCase().includes(needle) ||
          chapter.part.toLowerCase().includes(needle) ||
          chapter.summary.toLowerCase().includes(needle) ||
          numbers.get(chapter.slug) === needle,
      )
    : chapters;
  const parts = groupByPart(matches);

  const done = chapters.filter((c) => progress[c.slug] === "done").length;
  const completion = chapters.length ? (done / chapters.length) * 100 : 0;

  function togglePart(part: string) {
    setShutParts((current) =>
      current.includes(part) ? current.filter((p) => p !== part) : [...current, part],
    );
  }

  // The rail: chapter numbers only, enough to keep your place while the screen
  // belongs to the chapter and its lab.
  if (collapsed) {
    const railLink = ({ isActive }: { isActive: boolean }) =>
      `relative flex size-9 items-center justify-center rounded-md transition-colors ${
        isActive ? "bg-tint text-fg" : "text-fg-subtle hover:bg-tint hover:text-fg"
      }`;
    return (
      <aside className="hidden w-14 shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <div className="flex h-12 shrink-0 items-center justify-center">
          <IconButton
            onClick={onToggleCollapsed}
            title={`Expand the sidebar (${TOGGLE_KEY})`}
            aria-label="Expand the sidebar"
          >
            <PanelIcon />
          </IconButton>
        </div>

        <div className="flex min-h-0 flex-1 flex-col items-center gap-0.5 overflow-y-auto pb-3">
          <NavLink to="/" end title="Overview" aria-label="Overview" className={railLink}>
            <HomeIcon />
          </NavLink>
          <NavLink to="/slides" title="Slides" aria-label="Slides" className={railLink}>
            <PresentIcon />
          </NavLink>
          <NavLink to="/compute" title={computeTitle(active)} aria-label="Compute" className={railLink}>
            <ChipIcon />
            <span className="absolute right-1.5 top-1.5">
              <ComputeDot active={active} />
            </span>
          </NavLink>
          {book && <DownloadBookRailItem book={book} />}

          <div className="my-2 h-px w-6 bg-line" />

          {chapters.map((chapter) => {
            const status = progress[chapter.slug];
            return (
              <NavLink
                key={chapter.slug}
                to={`/c/${chapter.slug}`}
                title={`${numbers.get(chapter.slug)} · ${chapter.title}`}
                className={({ isActive }) =>
                  `flex h-7 w-9 shrink-0 items-center justify-center rounded-md font-mono text-[11px] tabular-nums transition-colors ${
                    isActive
                      ? "bg-accent text-on-accent"
                      : status === "done"
                        ? "text-ok hover:bg-tint"
                        : status
                          ? "text-fg hover:bg-tint"
                          : "text-fg-faint hover:bg-tint hover:text-fg"
                  }`
                }
              >
                {numbers.get(chapter.slug)}
              </NavLink>
            );
          })}
        </div>

        <div className="flex shrink-0 flex-col items-center gap-0.5 border-t border-line py-2">
          <ThemeCycle />
        </div>
      </aside>
    );
  }

  const next = chapters.find((c) => progress[c.slug] !== "done");

  return (
    <aside
      className={`${
        open ? "translate-x-0 shadow-pop" : "-translate-x-full"
      } fixed inset-y-0 left-0 z-30 flex w-[min(17.5rem,86vw)] shrink-0 flex-col border-r border-line bg-surface transition-transform duration-200 lg:static lg:w-[var(--rail-width)] lg:translate-x-0 lg:shadow-none`}
      // A shut drawer is only translated out of view, so without this its links
      // stay in the tab order and a screen reader still reads them out. From
      // `lg` up the same element is the static sidebar, which is never shut.
      inert={!wide && !open ? true : undefined}
    >
      <div className="flex h-16 shrink-0 items-center gap-2 border-b border-line pl-4 pr-2">
        <NavLink to="/" onClick={onNavigate} className="flex min-w-0 items-center gap-2.5 rounded-md">
          <Mark className="size-8" />
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-[17px] font-bold leading-tight tracking-[-0.045em] text-fg">
              learn<span className="text-terracotta">/</span>gpu
            </span>
            <span className="truncate font-mono text-[9px] uppercase leading-tight tracking-[0.2em] text-fg-subtle">
              Field notes
            </span>
          </span>
        </NavLink>
        <IconButton
          onClick={onToggleCollapsed}
          title={`Collapse the sidebar (${TOGGLE_KEY})`}
          aria-label="Collapse the sidebar"
          className="ml-auto hidden lg:inline-flex"
          size="sm"
        >
          <PanelIcon />
        </IconButton>
        <IconButton onClick={onClose} aria-label="Close the course list" className="ml-auto lg:hidden">
          <CloseIcon />
        </IconButton>
      </div>

      <nav className="shrink-0 space-y-px px-2.5 pb-2 pt-3" aria-label="App">
        <NavItem to="/" end icon={<HomeIcon />} onClick={onNavigate}>
          Overview
        </NavItem>
        <NavItem to="/slides" icon={<PresentIcon />} onClick={onNavigate}>
          Slides
        </NavItem>
        <NavItem
          to="/compute"
          icon={<ChipIcon />}
          onClick={onNavigate}
          aside={
            <span
              className={`flex items-center gap-1.5 text-[11.5px] font-normal tabular-nums ${
                (active ?? 0) > 0 ? "text-bad" : "text-fg-faint"
              }`}
              title={computeTitle(active)}
            >
              <ComputeDot active={active} />
              {computeLabel(active)}
            </span>
          }
        >
          Compute
        </NavItem>
        {book && <DownloadBookItem book={book} />}
      </nav>

      <div className="flex shrink-0 gap-1.5 px-2.5 pb-2.5">
        <label className="relative block min-w-0 flex-1">
          <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-faint" />
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") setFilter("");
            }}
            placeholder="Filter chapters"
            aria-label="Filter chapters"
            className="h-8 w-full rounded-[5px] border border-line bg-paper pl-8 pr-2.5 text-[13px] text-fg outline-none transition-colors placeholder:text-fg-faint focus:border-line-strong"
          />
        </label>
        <button
          type="button"
          onClick={onSearch}
          title="Search chapters and sections (/)"
          aria-label="Search chapters and sections"
          className="flex h-8 shrink-0 items-center rounded-[5px] border border-line bg-paper px-2 font-mono text-[11px] text-fg-subtle transition-colors hover:border-line-strong hover:text-fg"
        >
          /
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto overscroll-contain border-t border-line px-2.5 pb-4 pt-2" aria-label="Chapters">
        {parts.map(([part, items]) => {
          const shut = shutParts.includes(part) && !needle;
          const { label, name: partName } = splitPart(part);
          const partDone = items.filter((c) => progress[c.slug] === "done").length;
          return (
            <div key={part} className="mt-2 first:mt-0">
              <button
                onClick={() => togglePart(part)}
                className="sidebar-group-title group flex w-full items-center gap-1.5 rounded-[5px] px-1.5 py-1.5 text-left transition-colors hover:bg-surface-muted !mx-0 !my-0.5"
                aria-expanded={!shut}
              >
                <ChevronDownIcon
                  className={`size-3 shrink-0 text-fg-faint transition-transform duration-200 ${shut ? "-rotate-90" : ""}`}
                />
                <span className="min-w-0 flex-1 truncate">
                  {label && <span className="text-kicker">{label} · </span>}
                  <span>{partName}</span>
                </span>
                <span className="tabular-nums text-fg-faint">
                  {partDone}/{items.length}
                </span>
              </button>

              {!shut && (
                <ul className="mb-1 space-y-px">
                  {items.map((chapter) => {
                    const status = progress[chapter.slug];
                    return (
                      <li key={chapter.slug}>
                        <NavLink
                          to={`/c/${chapter.slug}`}
                          onClick={onNavigate}
                          className={({ isActive }) =>
                            `sidebar-lesson !items-center ${isActive ? "is-active" : ""} ${
                              status === "done" ? "is-complete" : ""
                            }`
                          }
                        >
                          <span className="sidebar-lesson-number tabular-nums" aria-hidden>
                            {numbers.get(chapter.slug)}
                          </span>
                          <span className="min-w-0 flex-1 truncate" title={chapter.title}>
                            {chapter.title}
                            {status === "done" && <span className="sr-only"> (done)</span>}
                          </span>
                          {chapter.lab && status !== "done" && (
                            <span title={labTitle(chapter)} className="text-fg-faint">
                              <FlaskIcon className="size-3.5" />
                              <span className="sr-only">{labTitle(chapter)}</span>
                            </span>
                          )}
                          {status === "done" && (
                            <CheckIcon className="size-3.5 shrink-0 text-accent" aria-hidden />
                          )}
                        </NavLink>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          );
        })}

        {matches.length === 0 && (
          <p className="px-3 py-6 text-[13px] text-fg-subtle">No chapter matches “{filter}”.</p>
        )}
      </nav>

      <div className="shrink-0 border-t border-line px-4 pb-3 pt-3.5">
        <div className="mb-2 flex items-baseline justify-between">
          <span className="kicker !text-fg-subtle">Your progress</span>
          <span className="font-mono text-[11px] tabular-nums text-fg-muted">
            {done} / {chapters.length}
          </span>
        </div>
        <div
          className="progress-track"
          role="progressbar"
          aria-label="Chapters complete"
          aria-valuenow={done}
          aria-valuemin={0}
          aria-valuemax={chapters.length}
        >
          <div className="progress-fill" style={{ width: `${completion}%` }} />
        </div>
        <div className="mt-2.5 flex items-center justify-between gap-2 text-[12px]">
          {next ? (
            <NavLink
              to={`/c/${next.slug}`}
              onClick={onNavigate}
              className="font-semibold text-accent hover:underline"
            >
              {progress[next.slug] ? "Resume" : "Start"} {numbers.get(next.slug)} →
            </NavLink>
          ) : (
            <span className="font-semibold text-accent">Course complete</span>
          )}
          <ThemeSwitch />
        </div>
        <p
          className="mt-2.5 truncate text-[11.5px] text-fg-subtle"
          title={`Signed in as ${name}. Progress, drafts, and notes are saved to your account.`}
        >
          Saved for <span className="font-medium text-fg-muted">{name}</span>
        </p>
      </div>

      <p className="hidden shrink-0 items-center gap-1.5 px-4 pb-2.5 text-[11px] text-fg-faint lg:flex">
        <Kbd>{TOGGLE_KEY}</Kbd> hides the sidebar · <Kbd>/</Kbd> searches
      </p>
    </aside>
  );
}
