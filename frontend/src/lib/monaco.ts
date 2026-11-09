/**
 * Monaco, bundled with the app instead of fetched from a CDN at runtime.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance, which made the first paint of every lab wait on a third-party
 * download. This module builds a lean instance with the features and the one
 * language the labs need, gives it the code-slab colours, and registers it with the
 * loader. Import it from the editor component only, so the bundle stays in its
 * own chunk and pages without a lab never download it.
 */
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import { defineSlabThemes } from "../ui/monaco-theme";

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
 * The editor themes, one per app theme. The editor is the design system's
 * dark-green code slab in both themes, so the two differ only in the shade of
 * the background. The colours live in src/ui/monaco-theme.ts, which mirrors
 * the --code-* and --hl-* tokens in src/ui/tokens.css.
 */
export const LIGHT_THEME = "paper";
export const DARK_THEME = "graphite";

defineSlabThemes(monaco.editor, { light: LIGHT_THEME, dark: DARK_THEME });

loader.config({ monaco });

export { monaco };
