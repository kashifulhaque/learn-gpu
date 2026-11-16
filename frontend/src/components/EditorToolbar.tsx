import type { ReactNode } from "react";

import { usePopover } from "../lib/usePopover";
import EditorSettings from "./EditorSettings";
import { Icon, isMac, modKey, type IconName } from "./shellIcons";

export type ToolbarAction = {
  id: string;
  label: string;
  icon: IconName;
  onClick: () => void;
  /** Tooltip text. Defaults to the label. */
  tip?: string;
  /** For a toggle, such as a hint panel: whether it is on. */
  pressed?: boolean;
  /** The id of the panel the action opens, for aria-controls. */
  controls?: string;
  disabled?: boolean;
  /** "low" actions move into the overflow menu when the pane is narrow. */
  priority?: "high" | "low";
};

type Props = {
  title: string;
  /** The exercise brief, shown in a popover from the info button. */
  info?: ReactNode;
  /** Small chips after the title, such as the time limit. */
  meta?: ReactNode;
  /** Quiet status text before the actions, such as "Saved". Hidden when narrow. */
  status?: ReactNode;
  running: boolean;
  onRun: () => void;
  onCheck?: () => void;
  onStop?: () => void;
  runLabel?: string;
  /** Disables Run and Check while nothing can run them, such as with no runner configured. */
  runDisabled?: boolean;
  checkLabel?: string;
  actions?: ToolbarAction[];
  /** Folds the editor pane away. Without it, no collapse button shows. */
  onCollapse?: () => void;
  /** The shortcut that folds the pane, for the tooltip, such as "⌘ .". */
  collapseShortcut?: string;
  /** Extra controls before the settings button. */
  children?: ReactNode;
};

/**
 * The one-row exercise toolbar, about 40px tall:
 * collapse, title with an info popover, chips, then Run, Check, the icon
 * actions, and the editor settings. When the pane is narrow, low-priority
 * actions move into a "More actions" menu (a CSS container query decides).
 *
 * Portable: course-specific behaviour comes in through props.
 */
export default function EditorToolbar({
  title,
  info,
  meta,
  status,
  running,
  onRun,
  onCheck,
  onStop,
  runLabel = "Run",
  checkLabel = "Check",
  runDisabled = false,
  actions = [],
  onCollapse,
  collapseShortcut,
  children,
}: Props) {
  const mod = modKey();
  const infoPop = usePopover();
  const more = usePopover();
  const low = actions.filter((action) => action.priority === "low");

  return (
    <div className="sh-tb-wrap">
      <div className="sh-tb" role="toolbar" aria-label={`${title} toolbar`}>
        {onCollapse && (
          <button
            type="button"
            className="sh-tb-icon"
            onClick={onCollapse}
            aria-label="Hide the code editor"
            data-tip={`Hide the code editor${collapseShortcut ? ` (${collapseShortcut})` : ""}`}
          >
            <Icon name="panelRight" size={16} />
          </button>
        )}

        <div className="sh-tb-title-wrap" ref={infoPop.rootRef}>
          <h2 className="sh-tb-title" title={title}>
            {title}
          </h2>
          {info && (
            <>
              <button {...infoPop.buttonProps} className="sh-tb-icon sh-tb-info" aria-label="About this exercise">
                <Icon name="info" size={15} />
              </button>
              {infoPop.open && (
                <div {...infoPop.panelProps} role="dialog" aria-label="About this exercise" className="sh-popover sh-popover-start sh-tb-info-pop">
                  <p className="sh-settings-title">{title}</p>
                  <div className="sh-tb-info-body">{info}</div>
                </div>
              )}
            </>
          )}
          {meta && <span className="sh-tb-meta">{meta}</span>}
        </div>

        {status && <span className="sh-tb-status">{status}</span>}

        <div className="sh-tb-actions">
          {running && onStop ? (
            <button type="button" className="sh-tb-btn sh-tb-stop" onClick={onStop} data-tip="Stop the run">
              <Icon name="stop" size={12} />
              <span className="sh-tb-label">Stop</span>
            </button>
          ) : (
            <>
              <button
                type="button"
                className="sh-tb-btn"
                onClick={onRun}
                disabled={running || runDisabled}
                aria-label={runLabel}
                aria-keyshortcuts={isMac() ? "Meta+Enter" : "Control+Enter"}
                data-tip={`${runLabel} (${mod}↵)`}
              >
                <Icon name="play" size={12} />
                <span className="sh-tb-label">{runLabel}</span>
              </button>
              {onCheck && (
                <button
                  type="button"
                  className="sh-tb-btn sh-tb-primary"
                  onClick={onCheck}
                  disabled={running || runDisabled}
                  aria-label={checkLabel}
                  aria-keyshortcuts={isMac() ? "Shift+Meta+Enter" : "Shift+Control+Enter"}
                  data-tip={`${checkLabel} (⇧${mod}↵)`}
                >
                  <Icon name="check" size={14} />
                  <span className="sh-tb-label">{checkLabel}</span>
                </button>
              )}
            </>
          )}

          {actions.length > 0 && <span className="sh-tb-sep" aria-hidden="true" />}

          {actions.map((action) => (
            <button
              key={action.id}
              type="button"
              className={`sh-tb-icon ${action.priority === "low" ? "sh-tb-low" : ""}`}
              onClick={action.onClick}
              disabled={action.disabled}
              aria-label={action.label}
              aria-pressed={action.pressed}
              aria-controls={action.controls}
              data-tip={action.tip ?? action.label}
            >
              <Icon name={action.icon} size={16} />
            </button>
          ))}

          {low.length > 0 && (
            <div className="sh-pop-root sh-tb-more" ref={more.rootRef}>
              <button {...more.buttonProps} className="sh-tb-icon" aria-label="More actions" aria-haspopup="menu" data-tip="More actions">
                <Icon name="more" size={16} />
              </button>
              {more.open && (
                <div {...more.panelProps} role="menu" className="sh-popover sh-menu">
                  {low.map((action) => (
                    <button
                      key={action.id}
                      type="button"
                      role={action.pressed === undefined ? "menuitem" : "menuitemcheckbox"}
                      aria-checked={action.pressed}
                      disabled={action.disabled}
                      className="sh-menu-item"
                      onClick={() => {
                        more.close();
                        action.onClick();
                      }}
                    >
                      <Icon name={action.icon} size={15} />
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {children}
          <EditorSettings />
        </div>
      </div>
    </div>
  );
}

/**
 * A dismissible panel under the toolbar, for hints or the reference solution.
 * It shows only while open and scrolls on its own, so the editor keeps most
 * of the height.
 */
export function ToolbarPanel({
  id,
  title,
  onClose,
  children,
  footer,
}: {
  id?: string;
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <section id={id} className="sh-tb-panel" aria-label={title}>
      <header className="sh-tb-panel-head">
        <span className="sh-tb-panel-title">{title}</span>
        {footer}
        <button type="button" className="sh-tb-icon" onClick={onClose} aria-label={`Close ${title.toLowerCase()}`}>
          <Icon name="close" size={14} />
        </button>
      </header>
      <div className="sh-tb-panel-body">{children}</div>
    </section>
  );
}

/**
 * The slim vertical strip a folded editor leaves behind. Clicking it brings
 * the editor back.
 */
export function EditorRail({
  label = "Code editor",
  title,
  shortcut,
  onOpen,
}: {
  label?: string;
  title: string;
  shortcut?: string;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`Show the code editor: ${title}`}
      title={`Show the code editor${shortcut ? ` (${shortcut})` : ""}`}
      className="sh-editor-rail"
    >
      <Icon name="panelRight" size={16} />
      <span className="sh-editor-rail-label">{label}</span>
    </button>
  );
}
