import { useCallback, useEffect, useId, type ReactNode } from "react";

import { setEditorPrefs, useEditorPrefs } from "../lib/editorPrefs";
import { Icon } from "./shellIcons";

/**
 * The output panel's folded state, kept in the shared editor prefs.
 *
 * `runId` counts runs: pass a number that goes up each time a run or check
 * starts. The panel opens when it changes, so the reader sees the run. If the
 * reader folds it again during that run, it stays folded until the next one.
 */
export function useOutputCollapse(runId: number) {
  const [prefs] = useEditorPrefs();
  useEffect(() => {
    if (runId > 0) setEditorPrefs({ outputCollapsed: false });
  }, [runId]);
  const setCollapsed = useCallback((value: boolean) => setEditorPrefs({ outputCollapsed: value }), []);
  const toggle = useCallback(() => setEditorPrefs({ outputCollapsed: !prefs.outputCollapsed }), [prefs.outputCollapsed]);
  return { collapsed: prefs.outputCollapsed, setCollapsed, toggle };
}

type HeaderProps = {
  /** "Output" or "Checks". */
  label: string;
  /** The status summary, such as "3 of 4 passed" or "Finished in 1.2s". */
  status?: ReactNode;
  running?: boolean;
  /** Quiet detail after the label, such as where the run happened. */
  detail?: ReactNode;
  collapsed: boolean;
  onToggle: () => void;
  /** The id of the output body this header shows and hides. */
  controls?: string;
};

/**
 * The output panel's header strip: label, running spinner, status summary,
 * and a chevron. The whole strip is one button that folds the panel.
 */
export function OutputHeader({ label, status, running, detail, collapsed, onToggle, controls }: HeaderProps) {
  return (
    <button
      type="button"
      className="sh-out-head"
      onClick={onToggle}
      aria-expanded={!collapsed}
      aria-controls={controls}
      aria-label={`${collapsed ? "Show" : "Hide"} the ${label.toLowerCase()}`}
    >
      <span className="sh-out-label">{label}</span>
      {detail && <span className="sh-out-detail">{detail}</span>}
      {running && (
        <span className="sh-out-running">
          <Spinner />
          Running
        </span>
      )}
      <span className="sh-out-spacer" />
      {status && !running && <span className="sh-out-status">{status}</span>}
      <Icon name={collapsed ? "chevronUp" : "chevronDown"} size={14} className="sh-out-chevron" />
    </button>
  );
}

/**
 * A header strip and a body that folds under it. For panes without a
 * draggable split. A pane with a split can use `OutputHeader` as the split's
 * collapsed rail instead, as the ml course's ExercisePane does.
 */
export default function CollapsibleOutput({
  label,
  status,
  detail,
  running = false,
  runId,
  children,
  className = "",
}: Omit<HeaderProps, "collapsed" | "onToggle" | "controls"> & {
  runId: number;
  children: ReactNode;
  className?: string;
}) {
  const { collapsed, toggle } = useOutputCollapse(runId);
  const id = useId();
  return (
    <div className={`sh-out ${collapsed ? "is-collapsed" : ""} ${className}`}>
      <OutputHeader
        label={label}
        status={status}
        detail={detail}
        running={running}
        collapsed={collapsed}
        onToggle={toggle}
        controls={id}
      />
      <div id={id} className="sh-out-body" hidden={collapsed}>
        {children}
      </div>
    </div>
  );
}

export function Spinner() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" className="sh-spin" aria-hidden="true">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" fill="none" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  );
}
