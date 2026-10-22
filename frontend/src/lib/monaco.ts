/**
 * Monaco, bundled with the app instead of fetched from a CDN at runtime.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance, which made the first paint of every lab wait on a third-party
 * download. This module builds a lean instance with the features and the one
 * language the labs need, gives it the app's colours, and registers it with the
 * loader. Import it from the editor component only, so the bundle stays in its
 * own chunk and pages without a lab never download it.
 */
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

// Editor features. The full set adds a diff editor, a colour picker, code
// lenses, and other machinery that has no provider here.
import "monaco-editor/features/codicon/register";
import "monaco-editor/features/anchorSelect/register";
import "monaco-editor/features/bracketMatching/register";
import "monaco-editor/features/caretOperations/register";
import "monaco-editor/features/clipboard/register";
import "monaco-editor/features/comment/register";
import "monaco-editor/features/contextmenu/register";
import "monaco-editor/features/cursorUndo/register";
import "monaco-editor/features/dnd/register";
import "monaco-editor/features/find/register";
import "monaco-editor/features/folding/register";
import "monaco-editor/features/fontZoom/register";
import "monaco-editor/features/indentation/register";
import "monaco-editor/features/lineSelection/register";
import "monaco-editor/features/linesOperations/register";
import "monaco-editor/features/links/register";
import "monaco-editor/features/longLinesHelper/register";
import "monaco-editor/features/multicursor/register";
import "monaco-editor/features/placeholderText/register";
import "monaco-editor/features/quickCommand/register";
import "monaco-editor/features/readOnlyMessage/register";
import "monaco-editor/features/smartSelect/register";
import "monaco-editor/features/snippet/register";
import "monaco-editor/features/suggest/register";
import "monaco-editor/features/tokenization/register";
import "monaco-editor/features/unicodeHighlighter/register";
import "monaco-editor/features/wordHighlighter/register";
import "monaco-editor/features/wordOperations/register";
import "monaco-editor/features/wordPartOperations/register";

// The one language the labs are written in.
import "monaco-editor/languages/definitions/python/register";

self.MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

/**
 * The editor themes, one per app theme. Monaco takes literal colours rather
 * than CSS variables, so each palette is a copy of the matching block in
 * index.css: `:root` for light, `:root[data-theme="dark"]` for dark. Change a
 * value there and change it here too.
 */
export const LIGHT_THEME = "paper";
export const DARK_THEME = "graphite";

type Palette = {
  paper: string;
  card: string;
  tint: string;
  line: string;
  lineStrong: string;
  fg: string;
  fgSubtle: string;
  fgFaint: string;
  accent: string;
  keyword: string;
  string: string;
  number: string;
  comment: string;
  fn: string;
  type: string;
  punct: string;
};

const LIGHT: Palette = {
  paper: "#fcfbf8",
  card: "#ffffff",
  tint: "#ebe8e1",
  line: "#e4e0d7",
  lineStrong: "#cdc7ba",
  fg: "#1d1b18",
  fgSubtle: "#706a60",
  fgFaint: "#a09a8f",
  accent: "#c2410c",
  keyword: "#a13d12",
  string: "#3b7738",
  number: "#8a5a00",
  comment: "#8d877c",
  fn: "#2c5ea8",
  type: "#7446b0",
  punct: "#6b655b",
};

const DARK: Palette = {
  paper: "#171614",
  card: "#1f1d1b",
  tint: "#282623",
  line: "#2c2a26",
  lineStrong: "#3d3a35",
  fg: "#ece8e1",
  fgSubtle: "#968f83",
  fgFaint: "#6c665c",
  accent: "#f0864a",
  keyword: "#f0955f",
  string: "#a3cf8c",
  number: "#e6b566",
  comment: "#7f786d",
  fn: "#8db1ee",
  type: "#c2a8f0",
  punct: "#a39c90",
};

// Token rules want a bare hex value; the colour map wants the leading "#".
const bare = (hex: string) => hex.slice(1);

function defineTheme(name: string, base: "vs" | "vs-dark", p: Palette) {
  monaco.editor.defineTheme(name, {
    base,
    inherit: true,
    rules: [
      { token: "", foreground: bare(p.fg) },
      { token: "keyword", foreground: bare(p.keyword) },
      { token: "string", foreground: bare(p.string) },
      { token: "string.escape", foreground: bare(p.number) },
      { token: "comment", foreground: bare(p.comment), fontStyle: "italic" },
      { token: "number", foreground: bare(p.number) },
      { token: "tag", foreground: bare(p.type) },
      { token: "type", foreground: bare(p.type) },
      { token: "function", foreground: bare(p.fn) },
      { token: "delimiter", foreground: bare(p.punct) },
      { token: "delimiter.parenthesis", foreground: bare(p.punct) },
      { token: "identifier", foreground: bare(p.fg) },
    ],
    colors: {
      // The editor sits straight on the reading surface, so it has no box.
      "editor.background": p.paper,
      "editor.foreground": p.fg,
      "editorCursor.foreground": p.accent,
      "editor.lineHighlightBackground": `${p.tint}80`,
      "editor.lineHighlightBorder": `${p.tint}00`,
      "editor.selectionBackground": `${p.accent}2e`,
      "editor.inactiveSelectionBackground": `${p.accent}1a`,
      "editor.selectionHighlightBackground": `${p.accent}14`,
      "editor.wordHighlightBackground": `${p.fn}1a`,
      "editor.wordHighlightStrongBackground": `${p.fn}29`,
      "editor.findMatchBackground": `${p.number}55`,
      "editor.findMatchHighlightBackground": `${p.number}26`,
      "editorLineNumber.foreground": p.fgFaint,
      "editorLineNumber.activeForeground": p.fgSubtle,
      "editorIndentGuide.background1": p.line,
      "editorIndentGuide.activeBackground1": p.lineStrong,
      "editorWhitespace.foreground": p.lineStrong,
      "editorBracketMatch.background": `${p.accent}14`,
      "editorBracketMatch.border": `${p.accent}66`,
      "editorBracketHighlight.foreground1": p.punct,
      "editorBracketHighlight.foreground2": p.fn,
      "editorBracketHighlight.foreground3": p.type,
      "editorGutter.background": p.paper,
      "editorWidget.background": p.card,
      "editorWidget.border": p.line,
      "editorSuggestWidget.background": p.card,
      "editorSuggestWidget.border": p.line,
      "editorSuggestWidget.foreground": p.fg,
      "editorSuggestWidget.selectedBackground": p.tint,
      "editorSuggestWidget.highlightForeground": p.accent,
      "editorHoverWidget.background": p.card,
      "editorHoverWidget.border": p.line,
      "input.background": p.card,
      "input.border": p.line,
      "input.foreground": p.fg,
      "focusBorder": p.accent,
      "list.hoverBackground": p.tint,
      "list.focusBackground": p.tint,
      "list.highlightForeground": p.accent,
      "menu.background": p.card,
      "menu.foreground": p.fg,
      "menu.selectionBackground": p.tint,
      "menu.selectionForeground": p.fg,
      "menu.separatorBackground": p.line,
      "scrollbar.shadow": "#00000000",
      "scrollbarSlider.background": `${p.lineStrong}80`,
      "scrollbarSlider.hoverBackground": `${p.lineStrong}b3`,
      "scrollbarSlider.activeBackground": p.lineStrong,
      "minimap.background": p.paper,
      "editorOverviewRuler.border": "#00000000",
    },
  });
}

defineTheme(LIGHT_THEME, "vs", LIGHT);
defineTheme(DARK_THEME, "vs-dark", DARK);

loader.config({ monaco });

export { monaco };
