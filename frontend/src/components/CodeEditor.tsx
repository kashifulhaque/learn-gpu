import { useState } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import {
  definePaperTheme,
  editorLook,
  editorThemeName,
  fontHasLigatures,
  fontStack,
  useEditorPrefs,
  useMonacoPrefs,
} from "../lib/editorPrefs";
import { DARK_THEME, LIGHT_THEME, monaco as bundledMonaco } from "../lib/monaco";
import { useTheme } from "../lib/theme";

type Props = {
  value: string;
  onChange: (value: string) => void;
  /** Called on Cmd or Ctrl with Return inside the editor, in every Vim mode too. */
  onRun?: () => void;
  /** Path shown to Monaco, so it picks the language and keeps undo history per lab. */
  path: string;
};

const PADDING = 18;
const THEMES = { light: LIGHT_THEME, dark: DARK_THEME };

// This module is the lazy editor chunk, so the paper theme is defined here
// and Monaco stays out of the main bundle.
definePaperTheme(bundledMonaco);

type MountedEditor = Parameters<OnMount>[0];
type MountedMonaco = Parameters<OnMount>[1];

/**
 * The lab editor. It fills the pane it is given. The theme, font, size, and
 * Vim mode come from the shared editor prefs (src/lib/editorPrefs.ts).
 */
export default function CodeEditor({ value, onChange, onRun, path }: Props) {
  const site = useTheme();
  const [prefs] = useEditorPrefs();
  const theme = editorThemeName(editorLook(prefs.theme, site), THEMES);

  const [editor, setEditor] = useState<MountedEditor | null>(null);
  const [monaco, setMonaco] = useState<MountedMonaco | null>(null);
  const [vimStatus, setVimStatus] = useState<HTMLDivElement | null>(null);

  useMonacoPrefs({ editor, monaco, site, themes: THEMES, vimStatus, onRun });

  const handleMount: OnMount = (mounted, instance) => {
    setEditor(mounted);
    setMonaco(instance);
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-code">
      <div className="min-h-0 flex-1">
        <Editor
          height="100%"
          path={path}
          defaultLanguage="python"
          theme={theme}
          value={value}
          onChange={(next) => onChange(next ?? "")}
          onMount={handleMount}
          loading={
            <div className="flex h-full w-full items-center justify-center bg-code text-[12.5px] text-code-muted">
              Loading the editor…
            </div>
          }
          options={{
            fontSize: prefs.fontSize,
            lineHeight: Math.round(prefs.fontSize * 1.55),
            fontFamily: fontStack(prefs.fontFamily),
            fontLigatures: fontHasLigatures(prefs.fontFamily),
            padding: { top: PADDING, bottom: PADDING },
            tabSize: 4,
            insertSpaces: true,
            detectIndentation: false,
            wordWrap: "on",
            wrappingIndent: "indent",
            minimap: { enabled: false },
            lineNumbersMinChars: 3,
            lineDecorationsWidth: 14,
            glyphMargin: false,
            folding: true,
            showFoldingControls: "mouseover",
            renderLineHighlight: "line",
            renderLineHighlightOnlyWhenFocus: true,
            renderWhitespace: "selection",
            guides: { indentation: true, bracketPairs: false },
            // The slab theme colours brackets as punctuation, not in rainbow pairs.
            bracketPairColorization: { enabled: false },
            matchBrackets: "always",
            rulers: [],
            overviewRulerLanes: 0,
            overviewRulerBorder: false,
            hideCursorInOverviewRuler: true,
            scrollbar: {
              vertical: "auto",
              horizontal: "hidden",
              verticalScrollbarSize: 10,
              useShadows: false,
              alwaysConsumeMouseWheel: false,
            },
            scrollBeyondLastLine: false,
            smoothScrolling: true,
            cursorBlinking: "smooth",
            cursorSmoothCaretAnimation: "on",
            stickyScroll: { enabled: false },
            // There is no language server behind this editor, so the only
            // completions are words from the file. Offer them on demand
            // (Ctrl+Space) rather than after every keystroke.
            quickSuggestions: false,
            suggestOnTriggerCharacters: false,
            wordBasedSuggestions: "currentDocument",
            parameterHints: { enabled: false },
            acceptSuggestionOnEnter: "off",
            // The starters use em dashes in docstrings. Without this Monaco
            // boxes every one of them as an ambiguous character.
            unicodeHighlight: { ambiguousCharacters: false, invisibleCharacters: false },
            fixedOverflowWidgets: true,
            automaticLayout: true,
          }}
        />
      </div>
      {prefs.vim && (
        <div ref={setVimStatus} className="sh-vim-status" aria-live="polite" aria-label="Vim status" />
      )}
    </div>
  );
}
