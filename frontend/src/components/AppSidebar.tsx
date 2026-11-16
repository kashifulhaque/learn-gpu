import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";

import { usePopover } from "../lib/usePopover";
import { Icon, modKey, type IconName } from "./shellIcons";

export type ShellThemeMode = "system" | "light" | "dark";

export type SidebarNavItem = {
  to: string;
  label: string;
  icon: IconName;
  /** A small count badge, such as the number of projects. */
  count?: number;
  active: boolean;
};

export type SidebarLesson = {
  id: string;
  to: string;
  title: string;
  /** The number shown before the title, such as "12" or "P3". */
  number?: string;
  done: boolean;
  current: boolean;
  /** The status glyph. Defaults to a tick or an empty circle from `done`. */
  mark?: ReactNode;
};

export type SidebarSection = {
  id: string;
  /** The section number, such as "01". */
  number?: string;
  title: string;
  lessons: SidebarLesson[];
};

export type SidebarUtility = {
  id: string;
  label: string;
  /** Quiet second line, such as "PDF · 412 pages". */
  detail?: string;
  icon: IconName;
  href: string;
  download?: boolean;
};

export type SidebarStatus = {
  label: string;
  title?: string;
  /** "ok" (default) is green, "alert" flags something that needs attention, and "neutral" is a quiet grey. */
  tone?: "ok" | "alert" | "neutral";
};

type Props = {
  brand: { name: string; kicker: string; mark: ReactNode; homeLabel?: string };
  /** The search button's text, such as "Find a lesson". */
  searchLabel: string;
  onSearch: () => void;
  nav: SidebarNavItem[];
  pathLabel?: string;
  sections: SidebarSection[];
  /** Prefix for the localStorage keys that remember open sections. */
  storageKey: string;
  progress: { done: number; total: number; label?: string; caption?: string };
  utilities?: SidebarUtility[];
  /** A compute status chip, such as "GPU available". */
  status?: SidebarStatus | null;
  theme: { mode: ShellThemeMode; onChange: (mode: ShellThemeMode) => void };
  account?: { username: string; signOutHref: string } | null;
  /** Whether the narrow-screen drawer is open. */
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  /** Whether the wide-screen sidebar is folded to its icon rail. */
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

const WIDE = "(min-width: 1024px)";

const THEMES: { mode: ShellThemeMode; label: string; icon: IconName }[] = [
  { mode: "system", label: "Match the system", icon: "monitor" },
  { mode: "light", label: "Light", icon: "sun" },
  { mode: "dark", label: "Dark", icon: "moon" },
];

/**
 * The app's only navigation: brand, search, primary pages, the lesson tree
 * in collapsible sections, and a footer with progress and utilities.
 *
 * On a wide screen it folds to a 56px icon rail. On a narrow screen it is an
 * off-canvas drawer behind a floating menu button. It is modelled on the tpu
 * reference site's sidebar.
 *
 * Portable: every course-specific string, link, and count comes in as props.
 */
export default function AppSidebar(props: Props) {
  const { open, onOpen, onClose, collapsed } = props;
  const wide = useWide();
  const rail = collapsed && wide;
  const fab = useRef<HTMLButtonElement>(null);
  const aside = useRef<HTMLElement>(null);

  // Escape closes the drawer and returns focus to the menu button.
  useEffect(() => {
    if (!open || wide) return;
    aside.current?.querySelector<HTMLElement>("a, button")?.focus();
    function onKey(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      onClose();
      fab.current?.focus();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, wide, onClose]);

  const drawerHidden = !wide && !open;

  return (
    <>
      {!wide && (
        <button
          ref={fab}
          type="button"
          className="sh-fab"
          onClick={onOpen}
          aria-label="Open navigation"
          aria-expanded={open}
          aria-controls="sh-sidebar"
        >
          <Icon name="menu" size={20} />
        </button>
      )}
      {!wide && open && <div className="sh-backdrop" onClick={onClose} aria-hidden="true" />}

      <aside
        ref={aside}
        id="sh-sidebar"
        aria-label="Course navigation"
        className={`sh-sidebar ${rail ? "is-rail" : ""} ${open ? "is-open" : ""}`}
        inert={drawerHidden || undefined}
      >
        {rail ? <Rail {...props} /> : <Full {...props} wide={wide} />}
      </aside>
    </>
  );
}

function useWide(): boolean {
  const [wide, setWide] = useState(() => typeof window !== "undefined" && window.matchMedia(WIDE).matches);
  useEffect(() => {
    const list = window.matchMedia(WIDE);
    const update = () => setWide(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, []);
  return wide;
}

/* --------------------------------------------------------------------------
   The full sidebar.
   -------------------------------------------------------------------------- */

function Full({
  brand,
  searchLabel,
  onSearch,
  nav,
  pathLabel = "Your learning path",
  sections,
  storageKey,
  progress,
  utilities = [],
  status,
  theme,
  account,
  onClose,
  onToggleCollapsed,
  wide,
}: Props & { wide: boolean }) {
  const mod = modKey();
  const [openMap, setOpenMap] = useState<Record<string, boolean>>(() => {
    try {
      return JSON.parse(window.localStorage.getItem(`${storageKey}.sections`) ?? "{}") ?? {};
    } catch {
      return {};
    }
  });
  const currentSection = useMemo(
    () => sections.find((section) => section.lessons.some((lesson) => lesson.current))?.id ?? null,
    [sections],
  );

  // The section holding the current lesson opens whenever the reader moves
  // into it, even if they closed it earlier.
  useEffect(() => {
    if (currentSection) setOpenMap((map) => (map[currentSection] ? map : { ...map, [currentSection]: true }));
  }, [currentSection]);

  useEffect(() => {
    try {
      window.localStorage.setItem(`${storageKey}.sections`, JSON.stringify(openMap));
    } catch {
      // Without storage the sections reset on reload.
    }
  }, [openMap, storageKey]);

  // Bring the current lesson into view in a long tree.
  const tree = useRef<HTMLElement>(null);
  useEffect(() => {
    tree.current?.querySelector('[aria-current="page"]')?.scrollIntoView({ block: "nearest" });
  }, [currentSection]);

  const share = progress.total ? (progress.done / progress.total) * 100 : 0;

  return (
    <div className="sh-full">
      <div className="sh-top">
        <div className="sh-brand-row">
          <Link to="/" className="sh-brand" aria-label={brand.homeLabel ?? `${brand.name}, course overview`} onClick={onClose}>
            <span className="sh-brand-mark">{brand.mark}</span>
            <span className="sh-brand-text">
              <span className="sh-brand-name">{brand.name}</span>
              <small>{brand.kicker}</small>
            </span>
          </Link>
          {wide ? (
            <button
              type="button"
              className="sh-icon-button"
              onClick={onToggleCollapsed}
              aria-label="Collapse the sidebar"
              aria-expanded={true}
              aria-controls="sh-sidebar"
              data-tip={`Collapse (${mod} B)`}
            >
              <Icon name="panel" size={18} />
            </button>
          ) : (
            <button type="button" className="sh-icon-button" onClick={onClose} aria-label="Close navigation">
              <Icon name="close" size={18} />
            </button>
          )}
        </div>

        <button type="button" className="sh-search-trigger" onClick={onSearch}>
          <Icon name="search" size={16} />
          <span>{searchLabel}</span>
          <kbd>/</kbd>
        </button>

        <nav className="sh-primary-nav" aria-label="Explore">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onClose}
              className={item.active ? "is-active" : ""}
              aria-current={item.active ? "page" : undefined}
            >
              <Icon name={item.icon} size={17} />
              <span className="sh-nav-label">{item.label}</span>
              {item.count != null && <span className="sh-nav-count">{item.count}</span>}
            </NavLink>
          ))}
        </nav>
      </div>

      <nav ref={tree} className="sh-tree" aria-label="Lessons">
        <div className="sh-divider" />
        <div className="sh-label">{pathLabel}</div>
        {sections.map((section) => {
          const isOpen = openMap[section.id] ?? false;
          const done = section.lessons.filter((lesson) => lesson.done).length;
          const total = section.lessons.length;
          const listId = `sh-section-${section.id.replace(/[^a-z0-9]+/gi, "-")}`;
          return (
            <div key={section.id} className={`sh-module ${isOpen ? "is-open" : ""}`}>
              <button
                type="button"
                className="sh-module-title"
                aria-expanded={isOpen}
                aria-controls={listId}
                onClick={() => setOpenMap((map) => ({ ...map, [section.id]: !isOpen }))}
              >
                {section.number && <span className="sh-module-number">{section.number}</span>}
                <span className="sh-module-name">{section.title}</span>
                <span className={`sh-module-count ${total > 0 && done === total ? "is-done" : ""}`}>
                  {done}/{total}
                </span>
                <Icon name="chevronDown" size={13} className="sh-module-chevron" />
              </button>
              {isOpen && (
                <ul id={listId} className="sh-lessons">
                  {section.lessons.map((lesson) => (
                    <li key={lesson.id}>
                      <Link
                        to={lesson.to}
                        onClick={onClose}
                        className={`sh-lesson ${lesson.current ? "is-active" : ""} ${lesson.done ? "is-done" : ""}`}
                        aria-current={lesson.current ? "page" : undefined}
                      >
                        <span className="sh-lesson-mark">{lesson.mark ?? <DefaultMark done={lesson.done} />}</span>
                        {lesson.number && <span className="sh-lesson-number">{lesson.number}</span>}
                        <span className="sh-lesson-title">{lesson.title}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </nav>

      <div className="sh-bottom">
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
        {progress.caption && <p className="sh-progress-caption">{progress.caption}</p>}

        {(utilities.length > 0 || status) && (
          <div className="sh-utilities">
            {utilities.map((utility) => (
              <a
                key={utility.id}
                className="sh-utility"
                href={utility.href}
                download={utility.download || undefined}
                title={utility.detail ? `${utility.label} (${utility.detail})` : utility.label}
              >
                <Icon name={utility.icon} size={16} />
                <span className="sh-utility-text">
                  <span>{utility.label}</span>
                  {utility.detail && <small>{utility.detail}</small>}
                </span>
              </a>
            ))}
            {status && (
              <span className={`sh-status ${status.tone && status.tone !== "ok" ? `is-${status.tone}` : ""}`} title={status.title}>
                <span className="sh-status-dot" aria-hidden="true" />
                {status.label}
              </span>
            )}
          </div>
        )}

        <div className="sh-bottom-row">
          {account ? <AccountMenu account={account} /> : <span />}
          <ThemeSwitch mode={theme.mode} onChange={theme.onChange} />
        </div>
      </div>
    </div>
  );
}

function DefaultMark({ done }: { done: boolean }) {
  return <span className={`sh-dot ${done ? "is-done" : ""}`}>{done && <Icon name="check" size={9} />}</span>;
}

/* --------------------------------------------------------------------------
   The icon rail.
   -------------------------------------------------------------------------- */

function Rail({ brand, onSearch, nav, utilities = [], status, theme, account, onToggleCollapsed, progress }: Props) {
  const mod = modKey();
  const current = THEMES.find((item) => item.mode === theme.mode) ?? THEMES[0];
  const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
  const share = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <div className="sh-rail">
      <Link to="/" className="sh-rail-mark" aria-label={brand.homeLabel ?? `${brand.name}, course overview`} data-tip={brand.name}>
        {brand.mark}
      </Link>
      <button
        type="button"
        className="sh-icon-button"
        onClick={onToggleCollapsed}
        aria-label="Expand the sidebar"
        aria-expanded={false}
        aria-controls="sh-sidebar"
        data-tip={`Expand (${mod} B)`}
      >
        <Icon name="panel" size={18} />
      </button>
      <button type="button" className="sh-icon-button" onClick={onSearch} aria-label="Search" data-tip="Search (/)">
        <Icon name="search" size={18} />
      </button>
      <div className="sh-rail-divider" />
      {nav.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={`sh-icon-button ${item.active ? "is-active" : ""}`}
          aria-label={item.label}
          aria-current={item.active ? "page" : undefined}
          data-tip={item.count != null ? `${item.label} (${item.count})` : item.label}
        >
          <Icon name={item.icon} size={18} />
        </NavLink>
      ))}

      <div className="sh-rail-spacer" />

      <div
        className="sh-rail-progress"
        role="progressbar"
        aria-label={progress.label ?? "Your progress"}
        aria-valuemin={0}
        aria-valuemax={progress.total}
        aria-valuenow={progress.done}
        data-tip={`${progress.done} of ${progress.total} done`}
        tabIndex={0}
      >
        <span style={{ height: `${share}%` }} />
      </div>
      {status && (
        <span
          className={`sh-icon-button sh-rail-status ${status.tone && status.tone !== "ok" ? `is-${status.tone}` : ""}`}
          role="img"
          aria-label={status.label}
          data-tip={status.label}
          tabIndex={0}
        >
          <Icon name="gpu" size={18} />
          <span className="sh-status-dot" aria-hidden="true" />
        </span>
      )}
      {utilities.map((utility) => (
        <a
          key={utility.id}
          className="sh-icon-button"
          href={utility.href}
          download={utility.download || undefined}
          aria-label={utility.detail ? `${utility.label} (${utility.detail})` : utility.label}
          data-tip={utility.label}
        >
          <Icon name={utility.icon} size={18} />
        </a>
      ))}
      <button
        type="button"
        className="sh-icon-button"
        onClick={() => theme.onChange(next.mode)}
        aria-label={`Theme: ${current.label}. Switch to ${next.label.toLowerCase()}.`}
        data-tip={`Theme: ${current.label}`}
      >
        <Icon name={current.icon} size={18} />
      </button>
      {account && <AccountMenu account={account} compact />}
    </div>
  );
}

/* --------------------------------------------------------------------------
   Shared pieces.
   -------------------------------------------------------------------------- */

function ThemeSwitch({ mode, onChange }: { mode: ShellThemeMode; onChange: (mode: ShellThemeMode) => void }) {
  return (
    <div className="sh-theme-switch" role="radiogroup" aria-label="Theme">
      {THEMES.map((item) => (
        <button
          key={item.mode}
          type="button"
          role="radio"
          aria-checked={mode === item.mode}
          aria-label={item.label}
          data-tip={item.label}
          onClick={() => onChange(item.mode)}
        >
          <Icon name={item.icon} size={15} />
        </button>
      ))}
    </div>
  );
}

function AccountMenu({ account, compact = false }: { account: { username: string; signOutHref: string }; compact?: boolean }) {
  const popover = usePopover();
  return (
    <div ref={popover.rootRef} className={`sh-pop-root sh-account ${compact ? "is-compact" : ""}`}>
      <button
        {...popover.buttonProps}
        aria-haspopup="menu"
        className="sh-account-button"
        aria-label={compact ? `Account: ${account.username}` : undefined}
        data-tip={compact ? account.username : undefined}
      >
        <span className="sh-avatar" aria-hidden="true">
          {account.username.slice(0, 1)}
        </span>
        {!compact && <span className="sh-account-name">{account.username}</span>}
      </button>
      {popover.open && (
        <div {...popover.panelProps} role="menu" className="sh-popover sh-menu sh-account-menu">
          <p className="sh-menu-note">Signed in as {account.username}</p>
          <a href={account.signOutHref} role="menuitem" className="sh-menu-item">
            <Icon name="logout" size={15} />
            Sign out
          </a>
        </div>
      )}
    </div>
  );
}
