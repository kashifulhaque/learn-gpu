// Generated from weights-and-wires/sites ui/ at b3af704 by scripts/sync-ui.sh. Do not edit here.

/**
 * Monaco themes that match the dark-green code slab in components.css.
 *
 * The editor is a slab in both page themes, as arm64 draws its code blocks,
 * so the light and dark themes differ only in the background shade. The hex
 * values mirror the --code-* and --hl-* tokens in tokens.css. Change both
 * files together.
 *
 * Usage:
 *
 *   import * as monaco from "monaco-editor/editor/editor.api";
 *   import { defineSlabThemes } from "../ui/monaco-theme";
 *   const THEMES = defineSlabThemes(monaco.editor);
 *   // THEMES.light and THEMES.dark are the theme names to pass to Monaco.
 */

/** The subset of monaco.editor that this module calls. */
export interface MonacoThemeHost {
  defineTheme(
    name: string,
    data: {
      base: "vs" | "vs-dark" | "hc-black" | "hc-light";
      inherit: boolean;
      rules: { token: string; foreground?: string; fontStyle?: string }[];
      colors: Record<string, string>;
    },
  ): void;
}

export const slab = {
  light: {
    background: "#18392e",
    activeLine: "#1e4134",
  },
  dark: {
    background: "#10281e",
    activeLine: "#163527",
  },
  ink: "#e8eddf",
  line: "#355645",
  muted: "#b2c7b3",
  gutter: "#6f8d78",
  selection: "#2f5a45",
  cursor: "#efb48b",
  keyword: "#efb48b",
  string: "#d7e4b9",
  number: "#e5a16e",
  comment: "#a8bea6",
  title: "#b6d3b2",
  type: "#9fd0bd",
  punct: "#c3d1c0",
  error: "#f0a49a",
  // Bracket pair colours, cycled by nesting depth. They reuse the syntax
  // hues so brackets sit in the same palette as the code around them.
  brackets: ["#efb48b", "#b6d3b2", "#9fd0bd", "#e7c9a0", "#d7e4b9", "#c9a3e0"],
} as const;

const strip = (hex: string) => hex.replace("#", "");

function rules() {
  return [
    { token: "", foreground: strip(slab.ink) },
    { token: "keyword", foreground: strip(slab.keyword) },
    { token: "string", foreground: strip(slab.string) },
    { token: "string.escape", foreground: strip(slab.number) },
    { token: "comment", foreground: strip(slab.comment), fontStyle: "italic" },
    { token: "number", foreground: strip(slab.number) },
    { token: "type", foreground: strip(slab.type) },
    { token: "type.identifier", foreground: strip(slab.type) },
    { token: "delimiter", foreground: strip(slab.punct) },
    { token: "identifier", foreground: strip(slab.ink) },
    { token: "predefined", foreground: strip(slab.title) },
  ];
}

function colors(shade: { background: string; activeLine: string }) {
  return {
    "editor.background": shade.background,
    "editor.foreground": slab.ink,
    "editorCursor.foreground": slab.cursor,
    "editor.lineHighlightBackground": shade.activeLine,
    "editor.lineHighlightBorder": "#00000000",
    "editor.selectionBackground": slab.selection,
    "editor.inactiveSelectionBackground": `${slab.selection}99`,
    "editor.selectionHighlightBackground": `${slab.selection}80`,
    "editor.wordHighlightBackground": `${slab.selection}66`,
    "editor.findMatchBackground": "#e5a16e66",
    "editor.findMatchHighlightBackground": "#e5a16e33",
    "editorLineNumber.foreground": slab.gutter,
    "editorLineNumber.activeForeground": slab.muted,
    "editorIndentGuide.background1": `${slab.line}`,
    "editorIndentGuide.activeBackground1": slab.gutter,
    "editorWhitespace.foreground": slab.line,
    "editorBracketMatch.background": `${slab.selection}80`,
    "editorBracketMatch.border": slab.muted,
    "editorBracketHighlight.foreground1": slab.brackets[0],
    "editorBracketHighlight.foreground2": slab.brackets[1],
    "editorBracketHighlight.foreground3": slab.brackets[2],
    "editorBracketHighlight.foreground4": slab.brackets[3],
    "editorBracketHighlight.foreground5": slab.brackets[4],
    "editorBracketHighlight.foreground6": slab.brackets[5],
    "editorBracketHighlight.unexpectedBracket.foreground": slab.error,
    "editorBracketPairGuide.background1": `${slab.brackets[0]}55`,
    "editorBracketPairGuide.background2": `${slab.brackets[1]}55`,
    "editorBracketPairGuide.background3": `${slab.brackets[2]}55`,
    "editorBracketPairGuide.activeBackground1": slab.brackets[0],
    "editorBracketPairGuide.activeBackground2": slab.brackets[1],
    "editorBracketPairGuide.activeBackground3": slab.brackets[2],
    "editorGutter.background": shade.background,
    "editorWidget.background": shade.activeLine,
    "editorWidget.border": slab.line,
    "editorSuggestWidget.background": shade.activeLine,
    "editorSuggestWidget.border": slab.line,
    "editorSuggestWidget.selectedBackground": slab.selection,
    "scrollbarSlider.background": `${slab.gutter}55`,
    "scrollbarSlider.hoverBackground": `${slab.gutter}88`,
    "scrollbarSlider.activeBackground": `${slab.gutter}aa`,
  };
}

/**
 * Registers the two slab themes and returns their names.
 *
 * @param editor `monaco.editor`, or any object with a compatible defineTheme.
 * @param names Theme names to register. Defaults to "slab-light" and "slab-dark".
 */
export function defineSlabThemes(
  editor: MonacoThemeHost,
  names: { light: string; dark: string } = { light: "slab-light", dark: "slab-dark" },
) {
  editor.defineTheme(names.light, {
    base: "vs-dark",
    inherit: true,
    rules: rules(),
    colors: colors(slab.light),
  });
  editor.defineTheme(names.dark, {
    base: "vs-dark",
    inherit: true,
    rules: rules(),
    colors: colors(slab.dark),
  });
  return names;
}
