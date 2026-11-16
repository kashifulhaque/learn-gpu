import {
  DEFAULT_PREFS,
  FONTS,
  FONT_SIZE_MAX,
  FONT_SIZE_MIN,
  loadEditorFont,
  useEditorPrefs,
  type EditorThemePref,
  type FontId,
} from "../lib/editorPrefs";
import { usePopover } from "../lib/usePopover";
import { Icon, modKey } from "./shellIcons";

const THEMES: { value: EditorThemePref; label: string }[] = [
  { value: "site", label: "Match site" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

/**
 * The editor settings button and its popover: theme, font size, font family,
 * and Vim bindings. Every change applies at once and is saved in the browser.
 *
 * Portable: reads and writes the shared editor prefs store only.
 */
export default function EditorSettings({ align = "end" }: { align?: "start" | "end" }) {
  const [prefs, setPrefs] = useEditorPrefs();
  const popover = usePopover();
  const mod = modKey();

  return (
    <div ref={popover.rootRef} className="sh-pop-root">
      <button
        {...popover.buttonProps}
        className="sh-tb-icon"
        aria-label="Editor settings"
        data-tip="Editor settings"
      >
        <Icon name="gear" size={16} />
      </button>

      {popover.open && (
        <div
          {...popover.panelProps}
          role="dialog"
          aria-label="Editor settings"
          className={`sh-popover sh-settings ${align === "start" ? "sh-popover-start" : ""}`}
        >
          <p className="sh-settings-title">Editor settings</p>

          <fieldset className="sh-field">
            <legend>Theme</legend>
            <div className="sh-segmented" role="radiogroup" aria-label="Editor theme">
              {THEMES.map((theme) => (
                <button
                  key={theme.value}
                  type="button"
                  role="radio"
                  aria-checked={prefs.theme === theme.value}
                  onClick={() => setPrefs({ theme: theme.value })}
                >
                  {theme.label}
                </button>
              ))}
            </div>
          </fieldset>

          <div className="sh-field">
            <span className="sh-field-label" id="sh-font-size-label">
              Font size
            </span>
            <div className="sh-stepper" role="group" aria-labelledby="sh-font-size-label">
              <button
                type="button"
                aria-label="Decrease the font size"
                disabled={prefs.fontSize <= FONT_SIZE_MIN}
                onClick={() => setPrefs({ fontSize: prefs.fontSize - 1 })}
              >
                <Icon name="minus" size={14} />
              </button>
              <output aria-live="polite">{prefs.fontSize}px</output>
              <button
                type="button"
                aria-label="Increase the font size"
                disabled={prefs.fontSize >= FONT_SIZE_MAX}
                onClick={() => setPrefs({ fontSize: prefs.fontSize + 1 })}
              >
                <Icon name="plus" size={14} />
              </button>
            </div>
          </div>

          <label className="sh-field">
            <span className="sh-field-label">Font</span>
            <select
              className="sh-select"
              value={prefs.fontFamily}
              onChange={(event) => {
                const id = event.target.value as FontId;
                void loadEditorFont(id);
                setPrefs({ fontFamily: id });
              }}
            >
              {FONTS.map((font) => (
                <option key={font.id} value={font.id}>
                  {font.label}
                  {font.id === DEFAULT_PREFS.fontFamily ? " (default)" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="sh-field sh-switch-row">
            <span>
              <span className="sh-field-label">Vim bindings</span>
              <span className="sh-field-hint">Modal editing with a status line under the editor.</span>
            </span>
            <input
              type="checkbox"
              role="switch"
              className="sh-switch"
              checked={prefs.vim}
              onChange={(event) => setPrefs({ vim: event.target.checked })}
            />
          </label>

          <p className="sh-settings-foot">
            <span>
              <kbd>{mod}</kbd> <kbd>=</kbd> and <kbd>{mod}</kbd> <kbd>-</kbd> resize in the editor.
            </span>
            <button
              type="button"
              className="sh-link-button"
              onClick={() => setPrefs({ ...DEFAULT_PREFS, outputCollapsed: prefs.outputCollapsed })}
            >
              Reset
            </button>
          </p>
        </div>
      )}
    </div>
  );
}
