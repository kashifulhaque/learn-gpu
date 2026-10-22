import { useEffect, useRef, useState } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import { DARK_THEME, LIGHT_THEME, monaco } from "../lib/monaco";
import { useTheme } from "../lib/theme";

type Props = {
  value: string;
  onChange: (value: string) => void;
  /** Called on Ctrl+Enter or Cmd+Enter inside the editor. */
  onRun?: () => void;
  /** Path shown to Monaco, so it picks the language and keeps undo history per lab. */
  path: string;
  /**
   * Fill the parent instead of growing with the file. The split view gives the
   * editor a pane of its own, so there the height is the pane's to decide.
   */
  fill?: boolean;
};

const LINE_HEIGHT = 22;
const PADDING = 18;
const MIN_HEIGHT = 18 * LINE_HEIGHT + PADDING * 2;
const MAX_HEIGHT = 34 * LINE_HEIGHT + PADDING * 2;

function clampHeight(contentHeight: number): number {
  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, contentHeight));
}

/**
 * The lab editor. In a pane it fills what it is given; laid out in a page it
 * grows with the file up to a cap, so short starters do not sit in an empty box
 * and long ones do not trap the page scroll.
 */
export default function CodeEditor({ value, onChange, onRun, path, fill = false }: Props) {
  const [height, setHeight] = useState(MIN_HEIGHT);
  const theme = useTheme();
  const runRef = useRef(onRun);
  runRef.current = onRun;

  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);

  useEffect(() => {
    return () => {
      editorRef.current = null;
    };
  }, []);

  const handleMount: OnMount = (editor) => {
    editorRef.current = editor;
    if (!fill) {
      setHeight(clampHeight(editor.getContentHeight()));
      editor.onDidContentSizeChange((event) => {
        setHeight(clampHeight(event.contentHeight));
      });
    }
    // Monaco measures glyphs once, at mount. If IBM Plex Mono arrives after
    // that, the cursor drifts from the text until the metrics are taken again.
    document.fonts?.ready.then(() => monaco.editor.remeasureFonts());
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      runRef.current?.();
    });
  };

  return (
    <div
      className={fill ? "h-full min-h-0 bg-paper" : "bg-paper"}
      style={fill ? undefined : { height }}
    >
      <Editor
        height="100%"
        path={path}
        defaultLanguage="python"
        theme={theme === "dark" ? DARK_THEME : LIGHT_THEME}
        value={value}
        onChange={(next) => onChange(next ?? "")}
        onMount={handleMount}
        loading={
          <div className="flex h-full w-full items-center justify-center bg-paper text-[12.5px] text-fg-faint">
            Loading the editor…
          </div>
        }
        options={{
          fontSize: 13,
          lineHeight: LINE_HEIGHT,
          fontFamily: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace',
          fontLigatures: false,
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
          bracketPairColorization: { enabled: true },
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
  );
}
