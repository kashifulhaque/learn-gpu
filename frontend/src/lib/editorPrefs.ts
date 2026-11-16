/**
 * Code editor preferences: theme, font size, font family, Vim bindings, and
 * whether the output panel is folded. One store for the whole app, saved in
 * localStorage under `editor.prefs.v1`, so every editor and every settings
 * popover stays in step and a reload keeps the reader's choices.
 *
 * Portable: this file imports no course code. A course passes its own Monaco
 * instance and theme names to `useMonacoPrefs`.
 *
 * Fonts: every family except 0xProto is self-hosted through an `@fontsource`
 * package and loaded on demand with a dynamic `import()` of its CSS. 0xProto
 * has no fontsource or Google Fonts package, so its `@font-face` rules are
 * injected on demand and point at the official 0xType/0xProto release files
 * on jsDelivr, after a `local()` lookup for a copy the reader has installed.
 */

import { useEffect, useRef, useSyncExternalStore } from "react";
import type * as Monaco from "monaco-editor";

export type EditorThemePref = "site" | "light" | "dark";

/** The parts of the `monaco` namespace these helpers call. */
export type MonacoLike = {
  editor: Pick<typeof Monaco.editor, "defineTheme" | "setTheme" | "remeasureFonts">;
};

export type EditorPrefs = {
  theme: EditorThemePref;
  fontSize: number;
  fontFamily: FontId;
  vim: boolean;
  outputCollapsed: boolean;
};

export type FontId =
  | "0xproto"
  | "jetbrains-mono"
  | "fira-code"
  | "ibm-plex-mono"
  | "geist-mono"
  | "source-code-pro"
  | "red-hat-mono"
  | "martian-mono";

export type FontOption = {
  id: FontId;
  label: string;
  /** The CSS family name the font's @font-face rules declare. */
  family: string;
  /** Whether the font has programming ligatures worth turning on. */
  ligatures: boolean;
  /** Loads the font's @font-face rules. */
  load: () => Promise<unknown>;
};

export const FONT_SIZE_MIN = 11;
export const FONT_SIZE_MAX = 24;

const FALLBACK_STACK = '"SFMono-Regular", ui-monospace, Consolas, "Liberation Mono", monospace';

export const FONTS: FontOption[] = [
  { id: "0xproto", label: "0xProto", family: "0xProto", ligatures: false, load: load0xProto },
  {
    id: "jetbrains-mono",
    label: "JetBrains Mono",
    family: "JetBrains Mono Variable",
    ligatures: true,
    load: () => import("@fontsource-variable/jetbrains-mono"),
  },
  {
    id: "fira-code",
    label: "Fira Code",
    family: "Fira Code Variable",
    ligatures: true,
    load: () => import("@fontsource-variable/fira-code"),
  },
  {
    id: "ibm-plex-mono",
    label: "IBM Plex Mono",
    family: "IBM Plex Mono",
    ligatures: false,
    load: () => Promise.all([import("@fontsource/ibm-plex-mono/400.css"), import("@fontsource/ibm-plex-mono/400-italic.css")]),
  },
  {
    id: "geist-mono",
    label: "Geist Mono",
    family: "Geist Mono Variable",
    ligatures: false,
    load: () => import("@fontsource-variable/geist-mono"),
  },
  {
    id: "source-code-pro",
    label: "Source Code Pro",
    family: "Source Code Pro Variable",
    ligatures: false,
    load: () => import("@fontsource-variable/source-code-pro"),
  },
  {
    id: "red-hat-mono",
    label: "Red Hat Mono",
    family: "Red Hat Mono Variable",
    ligatures: false,
    load: () => import("@fontsource-variable/red-hat-mono"),
  },
  {
    id: "martian-mono",
    label: "Martian Mono",
    family: "Martian Mono Variable",
    ligatures: false,
    load: () => import("@fontsource-variable/martian-mono"),
  },
];

export const DEFAULT_PREFS: EditorPrefs = {
  theme: "site",
  fontSize: 14,
  fontFamily: "0xproto",
  vim: false,
  outputCollapsed: false,
};

const KEY = "editor.prefs.v1";

function fontOption(id: FontId): FontOption {
  return FONTS.find((font) => font.id === id) ?? FONTS[0];
}

/** The CSS font-family value for a font, with the system mono stack behind it. */
export function fontStack(id: FontId): string {
  return `"${fontOption(id).family}", ${FALLBACK_STACK}`;
}

export function fontHasLigatures(id: FontId): boolean {
  return fontOption(id).ligatures;
}

function clampSize(size: number): number {
  if (!Number.isFinite(size)) return DEFAULT_PREFS.fontSize;
  return Math.min(FONT_SIZE_MAX, Math.max(FONT_SIZE_MIN, Math.round(size)));
}

function sanitize(raw: unknown): EditorPrefs {
  const value = (raw && typeof raw === "object" ? raw : {}) as Partial<EditorPrefs>;
  return {
    theme: value.theme === "light" || value.theme === "dark" ? value.theme : "site",
    fontSize: clampSize(Number(value.fontSize ?? DEFAULT_PREFS.fontSize)),
    fontFamily: FONTS.some((font) => font.id === value.fontFamily)
      ? (value.fontFamily as FontId)
      : DEFAULT_PREFS.fontFamily,
    vim: value.vim === true,
    outputCollapsed: value.outputCollapsed === true,
  };
}

function read(): EditorPrefs {
  try {
    const raw = window.localStorage.getItem(KEY);
    return sanitize(raw ? JSON.parse(raw) : null);
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

let current: EditorPrefs = typeof window === "undefined" ? { ...DEFAULT_PREFS } : read();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  // Another tab changing the prefs updates this one too.
  const onStorage = (event: StorageEvent) => {
    if (event.key !== KEY) return;
    current = read();
    listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function getEditorPrefs(): EditorPrefs {
  return current;
}

export function setEditorPrefs(patch: Partial<EditorPrefs>) {
  current = sanitize({ ...current, ...patch });
  try {
    window.localStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    // Without storage the choices last until the page reloads.
  }
  listeners.forEach((listener) => listener());
}

/** The current prefs and a setter that merges a partial update. */
export function useEditorPrefs(): [EditorPrefs, (patch: Partial<EditorPrefs>) => void] {
  const prefs = useSyncExternalStore(subscribe, getEditorPrefs, getEditorPrefs);
  return [prefs, setEditorPrefs];
}

/* --------------------------------------------------------------------------
   Fonts.
   -------------------------------------------------------------------------- */

const loaded = new Map<FontId, Promise<void>>();

/**
 * Loads a font family's CSS and waits until the browser has the face, so
 * Monaco can measure it. Resolves even when loading fails: the editor then
 * keeps the system mono stack.
 */
export function loadEditorFont(id: FontId): Promise<void> {
  const existing = loaded.get(id);
  if (existing) return existing;
  const font = fontOption(id);
  const promise = font
    .load()
    .then(() => (document.fonts ? document.fonts.load(`14px "${font.family}"`) : undefined))
    .then(() => undefined)
    .catch(() => undefined);
  loaded.set(id, promise);
  return promise;
}

// Self-hosted copy of 0xProto 2.502 (SIL OFL 1.1); see public/fonts/0xproto/LICENSE.
const OXPROTO_BASE = "/fonts/0xproto";

function load0xProto(): Promise<void> {
  if (document.getElementById("font-0xproto")) return Promise.resolve();
  const style = document.createElement("style");
  style.id = "font-0xproto";
  const face = (file: string, local: string, fontStyle: string) =>
    `@font-face{font-family:"0xProto";font-style:${fontStyle};font-weight:400;font-display:swap;` +
    `src:local("${local}"),url("${OXPROTO_BASE}/${file}") format("woff2");}`;
  style.textContent =
    face("0xProto-Regular.woff2", "0xProto Regular", "normal") +
    face("0xProto-Italic.woff2", "0xProto Italic", "italic");
  document.head.appendChild(style);
  return Promise.resolve();
}

/* --------------------------------------------------------------------------
   Monaco themes.
   -------------------------------------------------------------------------- */

/** The light "paper" editor theme, from the kit's `.code-light` palette in tokens.css. */
export const PAPER_THEME = "shell-paper";

let paperDefined = false;

export function definePaperTheme(monaco: MonacoLike) {
  if (paperDefined) return;
  paperDefined = true;
  const c = {
    bg: "#f4f3ec",
    active: "#eae9df",
    ink: "#263e33",
    line: "#d9ddd1",
    muted: "#5a675c",
    selection: "#dfe6d6",
    keyword: "#97512b",
    string: "#4f6d26",
    number: "#9c5424",
    title: "#1d4939",
    type: "#386790",
    attr: "#82508f",
    error: "#a33832",
  };
  const strip = (hex: string) => hex.slice(1);
  monaco.editor.defineTheme(PAPER_THEME, {
    base: "vs",
    inherit: true,
    rules: [
      { token: "", foreground: strip(c.ink) },
      { token: "keyword", foreground: strip(c.keyword) },
      { token: "string", foreground: strip(c.string) },
      { token: "string.escape", foreground: strip(c.number) },
      { token: "comment", foreground: strip(c.muted), fontStyle: "italic" },
      { token: "number", foreground: strip(c.number) },
      { token: "type", foreground: strip(c.type) },
      { token: "type.identifier", foreground: strip(c.type) },
      { token: "delimiter", foreground: strip(c.ink) },
      { token: "identifier", foreground: strip(c.ink) },
      { token: "predefined", foreground: strip(c.title) },
    ],
    colors: {
      "editor.background": c.bg,
      "editor.foreground": c.ink,
      "editorCursor.foreground": c.keyword,
      "editor.lineHighlightBackground": c.active,
      "editor.lineHighlightBorder": "#00000000",
      "editor.selectionBackground": "#c9d8c0",
      "editor.inactiveSelectionBackground": "#dfe6d6",
      "editorLineNumber.foreground": "#8a968b",
      "editorLineNumber.activeForeground": c.ink,
      "editorIndentGuide.background1": c.line,
      "editorGutter.background": c.bg,
      "editorWidget.background": "#fffef9",
      "editorWidget.border": c.line,
      "editorSuggestWidget.background": "#fffef9",
      "editorSuggestWidget.border": c.line,
      "editorSuggestWidget.selectedBackground": c.selection,
      "editorBracketHighlight.foreground1": c.keyword,
      "editorBracketHighlight.foreground2": c.title,
      "editorBracketHighlight.foreground3": c.type,
      "editorBracketHighlight.unexpectedBracket.foreground": c.error,
      "scrollbarSlider.background": "#8a968b44",
      "scrollbarSlider.hoverBackground": "#8a968b77",
    },
  });
}

/**
 * Which look the editor pane takes: the course's code slab in the site's
 * light or dark shade, or the light paper panel.
 *
 * "Match site" keeps the course design: the slab, in the shade that goes with
 * the page theme. "Light" is the paper panel and "Dark" is the dark slab,
 * whatever the page theme is.
 */
export type EditorLook = "slab-light" | "slab-dark" | "paper";

export function editorLook(pref: EditorThemePref, site: "light" | "dark"): EditorLook {
  if (pref === "light") return "paper";
  if (pref === "dark") return "slab-dark";
  return site === "dark" ? "slab-dark" : "slab-light";
}

/** The Monaco theme name for a look, given the course's slab theme names. */
export function editorThemeName(look: EditorLook, themes: { light: string; dark: string }): string {
  return look === "paper" ? PAPER_THEME : look === "slab-dark" ? themes.dark : themes.light;
}

/** The current editor look, for components that wrap the editor. */
export function useEditorLook(site: "light" | "dark"): EditorLook {
  const [prefs] = useEditorPrefs();
  return editorLook(prefs.theme, site);
}

/* --------------------------------------------------------------------------
   Applying the prefs to a Monaco editor.
   -------------------------------------------------------------------------- */

type MonacoPrefsOptions = {
  editor: Monaco.editor.IStandaloneCodeEditor | null;
  monaco: MonacoLike | null;
  /** The resolved page theme. */
  site: "light" | "dark";
  /** The course's Monaco theme names for its light and dark slab. */
  themes: { light: string; dark: string };
  /** Where Vim mode writes its status line. Required for Vim mode to start. */
  vimStatus: HTMLElement | null;
  /** Keyboard actions. They work in every Vim mode too. */
  onRun?: () => void;
  onCheck?: () => void;
};

/**
 * Keeps a mounted Monaco editor in step with the prefs: theme, font size,
 * family and ligatures, and Vim mode. It also owns the keyboard shortcuts
 * Cmd or Ctrl with Return (run), with Shift and Return (check), and with
 * `=`, `-`, or `0` (font size), handled before Monaco and Vim see them.
 */
export function useMonacoPrefs({ editor, monaco, site, themes, vimStatus, onRun, onCheck }: MonacoPrefsOptions) {
  const [prefs] = useEditorPrefs();
  const run = useRef(onRun);
  const check = useRef(onCheck);
  run.current = onRun;
  check.current = onCheck;

  // Theme.
  useEffect(() => {
    if (!monaco) return;
    definePaperTheme(monaco);
    monaco.editor.setTheme(editorThemeName(editorLook(prefs.theme, site), themes));
  }, [monaco, prefs.theme, site, themes.light, themes.dark]);

  // Font size, family, and ligatures. The font is loaded first, then Monaco
  // re-measures its glyphs so the cursor lands on the right column.
  useEffect(() => {
    if (!editor || !monaco) return;
    let live = true;
    editor.updateOptions({
      fontSize: prefs.fontSize,
      lineHeight: Math.round(prefs.fontSize * 1.55),
      fontFamily: fontStack(prefs.fontFamily),
      fontLigatures: fontHasLigatures(prefs.fontFamily),
    });
    void loadEditorFont(prefs.fontFamily).then(() => {
      if (live) monaco.editor.remeasureFonts();
    });
    return () => {
      live = false;
    };
  }, [editor, monaco, prefs.fontSize, prefs.fontFamily]);

  // Vim mode, loaded only when the reader turns it on.
  useEffect(() => {
    if (!editor || !prefs.vim || !vimStatus) return;
    let disposed = false;
    let mode: { dispose: () => void } | null = null;
    void import("monaco-vim").then(({ initVimMode }) => {
      if (disposed) return;
      mode = initVimMode(editor, vimStatus);
    });
    return () => {
      disposed = true;
      mode?.dispose();
      vimStatus.textContent = "";
    };
  }, [editor, prefs.vim, vimStatus]);

  // Shortcuts, in the capture phase on the editor's own node so that they
  // run before Monaco's keybindings and before Vim's normal-mode handler.
  useEffect(() => {
    if (!editor) return;
    const node = editor.getContainerDomNode();
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      let handled = true;
      if (event.key === "Enter" && event.shiftKey) check.current?.();
      else if (event.key === "Enter") run.current?.();
      else if (event.key === "=" || event.key === "+") setEditorPrefs({ fontSize: getEditorPrefs().fontSize + 1 });
      else if (event.key === "-" || event.key === "_") setEditorPrefs({ fontSize: getEditorPrefs().fontSize - 1 });
      else if (event.key === "0") setEditorPrefs({ fontSize: DEFAULT_PREFS.fontSize });
      else handled = false;
      if (handled) {
        event.preventDefault();
        event.stopPropagation();
      }
    }
    node.addEventListener("keydown", onKey, true);
    return () => node.removeEventListener("keydown", onKey, true);
  }, [editor]);
}
