# App shell porting guide

This guide describes the shared app shell: the sidebar with its icon rail, the
search dialog, the overview page, and the editor chrome (toolbar, settings,
and collapsible output). The ml course is the reference implementation. Use
this guide to port the shell to rl and speech (Family A: `LessonPage`,
`ExercisePane`, and `RunConsole`) and to learn-gpu and learn-inference
(Family B: `ChapterPage` and `LabPane`).

## Portable files

Copy the following files verbatim. None of them imports course code. Each
course passes its own data in through props.

| File | What it is |
|---|---|
| `src/ui/shell.css` | Every shell style. Classes start with `sh-`. Reads only kit tokens from `tokens.css` and `type.css`. |
| `src/components/shellIcons.tsx` | `Icon` (24px line icons), `modKey()`, and `isMac()`. |
| `src/lib/usePopover.ts` | Button-and-popover state: outside click and Escape close it, and focus returns to the button. |
| `src/lib/editorPrefs.ts` | Editor prefs store, hooks, font loading, the light paper Monaco theme, and `useMonacoPrefs`. |
| `src/components/EditorSettings.tsx` | The settings button and popover. No props other than an optional `align`. |
| `src/components/EditorToolbar.tsx` | `EditorToolbar`, `ToolbarPanel`, and `EditorRail`. |
| `src/components/CollapsibleOutput.tsx` | `useOutputCollapse`, `OutputHeader`, `CollapsibleOutput`, and `Spinner`. |
| `src/components/AppSidebar.tsx` | The sidebar, icon rail, mobile drawer, and floating menu button. |
| `src/components/SearchDialog.tsx` | The tpu-style search dialog. |
| `src/pages/Overview.tsx` | `Overview`, `OverviewBlock`, and `OverviewList`. |

`shell.css` lives in `src/ui/`, but `scripts/sync-ui.sh` doesn't write it, so
a kit sync never overwrites it. Import it after the kit in `src/index.css`:

```css
@import "./ui/components.css";
@import "./ui/shell.css";
```

Shared components don't use Tailwind colour utilities, because the two
families name their Tailwind colours differently. Keep it that way.

## Props and config

### `AppSidebar`

- `brand`: `{ name, kicker, mark, homeLabel? }`. `mark` is the course's mark
  component, about 30px.
- `searchLabel`: for example, "Find a lesson" or "Find a chapter".
- `onSearch`: opens the search dialog.
- `nav`: `{ to, label, icon, count?, active }[]`. The first item is "Course
  overview" (`/`, icon `grid`). Add Projects, Slides, Compute, Demo gallery,
  or Lectures here with a count badge where it helps.
- `sections`: `{ id, number?, title, lessons }[]`, where each lesson is
  `{ id, to, title, number?, done, current, mark? }`. Pass the course's own
  status glyph as `mark`; without it, the sidebar draws a tick or an empty
  circle from `done`.
- `storageKey`: prefix for the open-sections key, for example `rl.sidebar`.
- `progress`: `{ done, total, label?, caption? }`.
- `utilities`: `{ id, label, detail?, icon, href, download? }[]`. Use it for
  the book download: icon `download`, `download: true`.
- `status`: `{ label, title? }` or `null`. Use it for the GPU or compute chip.
- `theme`: `{ mode, onChange }` with modes `system`, `light`, and `dark`.
- `account`: `{ username, signOutHref }` or `null`.
- `open`, `onOpen`, `onClose`: the narrow-screen drawer.
- `collapsed`, `onToggleCollapsed`: the wide-screen icon rail.

The sidebar renders its own floating menu button and backdrop below 1024px.
It keeps the section that holds the current lesson open, and it remembers the
other sections under `${storageKey}.sections`.

### `SearchDialog`

- `open`, `onClose`.
- `entries`: `{ id, href, title, kicker, summary, keywords?, headings? }[]`,
  where `headings` is `{ text, href }[]` and `href` jumps to the section.
- `noun`: "lesson" or "chapter". The hint reads "Search all N lessons ·
  Escape to close".
- `placeholder`: optional. Name topics from your own course.

Build `entries` in a course file. ml does it in `src/lib/shellData.ts`. If a
course indexes headings, include them. If it doesn't, add heading search from
the data it already has.

### `Overview`

- `eyebrow`, `pill?`, `title` (a node, so `<em>` works), and `intro`.
- `primary?`: `{ href, label }`, for "Start the course" or "Continue: TITLE".
- `secondary?`: `{ href, label }[]`.
- `next?`: `{ href, kicker, title, summary?, action }`, the "Up next" card.
- `progress`: `{ done, total, label?, detail? }`.
- `facts?`: `{ icon, value, label }[]`.
- `path`: `{ eyebrow, title, aside? }`, the heading over the section cards.
- `sections`: `{ id, number, title, description?, meta?, lessons }[]`, where
  each lesson is `{ id, href, title, number?, done, tag? }`.
- `previewCount?`: lessons a card lists before **Show all**. Default 5.
- `children`: extra blocks. Wrap each in `OverviewBlock` (`eyebrow`, `title`,
  `aside?`, `id?`) and use `OverviewList` for rows of links.

Fold everything the old dashboard showed into `children`, such as projects,
recent runs, GPU notes, or lecture links. Derive every string from course
data. Don't invent course facts.

### `EditorToolbar`

- `title`, `info?` (the brief, shown in a popover), `meta?` (the time-limit
  chip), and `status?` (for example, "Saved").
- `running`, `onRun`, `onCheck?`, `onStop?`, `runLabel?`, `checkLabel?`.
- `actions?`: `{ id, label, icon, onClick, tip?, pressed?, controls?,
  disabled?, priority? }[]`. Actions with `priority: "low"` move into the
  **More actions** menu when the pane is narrower than 560px.
- `onCollapse?` and `collapseShortcut?`: show the collapse button.
- `children?`: extra controls before the settings button.

`ToolbarPanel` (`id?`, `title`, `onClose`, `children`) is the dismissible
panel for hints or the solution, placed directly after the toolbar.
`EditorRail` (`title`, `shortcut?`, `onOpen`, `label?`) is the strip a folded
editor leaves behind.

### `editorPrefs.ts`

- Prefs live in localStorage under `editor.prefs.v1` as `{ theme, fontSize,
  fontFamily, vim, outputCollapsed }`. All five courses share the key, so a
  reader's choices follow them across courses on the same origin only.
- `useEditorPrefs()` returns `[prefs, setPrefs]`.
- `useMonacoPrefs({ editor, monaco, site, themes, vimStatus, onRun, onCheck })`
  applies the theme, font, size, ligatures, and Vim mode to a mounted editor.
  It also handles Cmd or Ctrl with Return, Shift and Return, `=`, `-`, and
  `0` in the capture phase, so the shortcuts work in Vim normal mode. Remove
  the course's own `addCommand` calls for Run and Check.
- `editorLook(pref, site)` and `useEditorLook(site)` return `slab-light`,
  `slab-dark`, or `paper`. `editorThemeName(look, themes)` maps a look to a
  Monaco theme name. Call `definePaperTheme(monaco)` once before the editor
  mounts.
- Also pass `fontSize`, `lineHeight`, `fontFamily: fontStack(id)`, and
  `fontLigatures: fontHasLigatures(id)` in the `<Editor options>` prop.
  `@monaco-editor/react` reapplies `options` on every render, so leaving them
  out resets the font.

### `CollapsibleOutput`

- `useOutputCollapse(runId)`: pass a counter that goes up when a run starts.
  The panel opens on each new run and stays folded if the reader folds it
  during that run.
- `OutputHeader`: `{ label, status?, running?, detail?, collapsed, onToggle,
  controls? }`.
- `CollapsibleOutput`: a header and a body in one, for panes without a
  draggable split. Family B labs can use this directly.

## Packages

Add these packages to `frontend/package.json`:

```bash
npm install monaco-vim@0.4.4 \
  @fontsource-variable/jetbrains-mono @fontsource-variable/fira-code \
  @fontsource/ibm-plex-mono @fontsource-variable/geist-mono \
  @fontsource-variable/source-code-pro @fontsource-variable/red-hat-mono \
  @fontsource-variable/martian-mono
```

0xProto has no fontsource or Google Fonts package. Copy
`frontend/public/fonts/0xproto/` (the two `.woff2` files and the OFL
`LICENSE`) into the course. `editorPrefs.ts` injects its `@font-face` rules on
demand, with a `local()` lookup first and then the self-hosted files under
`/fonts/0xproto`, so no third-party request is made.

Add these aliases to `vite.config.ts`. Both are required for Vim mode:

```ts
import { fileURLToPath } from "node:url";

resolve: {
  alias: [
    { find: /^monaco-editor\/esm\/vs\/(.*)$/, replacement: "monaco-editor/$1" },
    {
      find: /^monaco-vim$/,
      replacement: fileURLToPath(new URL("./node_modules/monaco-vim/dist/index.mjs", import.meta.url)),
    },
  ],
},
```

The first alias maps monaco-vim's pre-0.50 import path onto Monaco 0.56's
export map, so the app keeps one Monaco instance. The second skips
monaco-vim's `browser` export, a UMD build that bundles Monaco 0.33 code.

## What ml adapted

- `App.tsx`: the `<header>` and `AccountMenu` are gone. The layout is
  `<div class="sh-app">`, then `AppSidebar`, then `<main class="sh-main">`.
  Cmd or Ctrl with B toggles the rail on a wide screen and the drawer on a
  narrow one. `/` and Cmd or Ctrl with K open search.
- `components/Sidebar.tsx`: deleted. `AppSidebar` replaces it.
- `components/DownloadBook.tsx`: now only the `useBook(enabled)` hook. App
  turns the result into a sidebar utility.
- `lib/shellData.ts` (course-specific): search entries with `##` and `###`
  heading anchors, `nextLesson`, `solvedCount`, and `partNumber`.
- `pages/Dashboard.tsx`: an adapter that fills `Overview` and adds projects
  and recent runs as `OverviewBlock`s.
- `pages/LessonPage.tsx`: uses the shared `EditorRail`. The Cmd or Ctrl with
  `.` shortcut and the persisted `ml.editor.hidden` flag are unchanged.
- `components/ExercisePane.tsx`: the tall header is replaced by
  `EditorToolbar`. Hint is a high-priority action. Reset, Solution, and the GPU
  toggle are low priority. Hints and the solution open in `ToolbarPanel`s
  under the toolbar, and the editor stays visible. A `runId` counter drives
  `useOutputCollapse`. The pane root adds `code-light sh-slab-paper` beside
  `slab` when the look is `paper`.
- `components/RunConsole.tsx`: its header is `ConsoleHeader`, a wrapper
  around `OutputHeader`. `ExercisePane` passes `ConsoleHeader` with
  `collapsed` set as the `SplitPane`'s `rail`, so the console body stays
  mounted while folded.
- `components/CodeEditor.tsx`: reads prefs, calls `useMonacoPrefs`, and
  renders the Vim status line (`.sh-vim-status`) under the editor while Vim
  mode is on.

## Port to rl and speech

The steps match ml. Diff their `App.tsx`, `ExercisePane.tsx`, `RunConsole.tsx`,
`CodeEditor.tsx`, and `Dashboard.tsx` against ml's versions before this
change, then apply the same edits. Map their extra pages into `nav`: rl has
Projects and Slides, and speech has the Demo gallery. Use each course's own
storage prefix, brand, and mark.

## Port to learn-gpu and learn-inference

- Use `AppSidebar`, `SearchDialog` (`noun="chapter"`, `searchLabel="Find a
  chapter"`), and `Overview` the same way. Put Compute, Slides, and Lectures
  in `nav`. Their compute badge becomes `status`.
- Their `useTheme()` returns a different shape. Pass the resolved `"light"` or
  `"dark"` string as `site`, and adapt `theme.mode` and `theme.onChange`.
- In `LabPane`, replace the header with `EditorToolbar` and the output with
  `CollapsibleOutput`, or use `OutputHeader` as a split rail as ml does.
- Their editor is lazy-loaded. Call `definePaperTheme` and `useMonacoPrefs`
  inside the lazy `CodeEditor`, so Monaco stays out of the main bundle.
- The editor styles read `--paper`, `--surface`, `--ink`, `--muted`,
  `--forest`, `--line`, `--strong-line`, `--error`, and `--error-soft`. If the
  lab pane draws a code slab, remap those tokens on the pane root, as ml's
  `.slab` class does in `index.css`.

## What learn-gpu adapted

learn-gpu is the first Family B port. It changed four shared files in
backward-compatible ways. Copy them from learn-gpu, not ml, to learn-inference,
and backport them to ml, rl, and speech so the copies stay identical:

- `AppSidebar.tsx`: `status` takes an optional `tone` of `ok` (the default),
  `alert`, or `neutral`. learn-gpu passes `alert` while something runs on a GPU
  provider, `neutral` while the count is unknown, and `ok` when idle.
  `shell.css` has the matching `.is-alert` and `.is-neutral` rules.
- `EditorToolbar.tsx`: `runDisabled` disables Run and Check while nothing can
  run them, such as when no GPU provider is configured.
- `pages/Overview.tsx`: `noun` names a card's items in **Show all N
  chapters**. The default is `lesson`.
- `shellIcons.tsx`: a `slides` icon for the Slides nav item.

Course-specific changes:

- `App.tsx`: the old `Sidebar` and the narrow-screen header are gone. `nav` is
  Course overview, Slides, and Compute. Compute shows a count badge while
  something is running. The compute poll from `useActiveCompute` becomes
  `status`. The book download is a utility. Sign out goes to the same portal as
  the other courses. The slide routes (`/slides/:slug`, `/present`, and
  `/print`) stay outside the shell.
- `lib/shellData.ts`: `useSearchEntries(chapters, open)` loads every chapter
  body once, the first time search opens, and indexes its `##` headings as
  `/c/SLUG#s-N`, which `ChapterPage` already scrolls to.
- `pages/Dashboard.tsx`: an adapter that fills `Overview` from the chapter
  index and adds recent lab runs as an `OverviewBlock`. The part cards show a
  chapter count, not a number range, because a part's chapter numbers aren't
  contiguous.
- `components/LabPane.tsx`: `EditorToolbar` replaces the two header rows. The
  provider select goes in the toolbar's `children`. Hints and the worked
  solution open in `ToolbarPanel`s. The Console and Checks tabs become one
  output body, with checks and metrics first and the log after them.
  `OutputHeader` is the `SplitPane` rail, as in ml.
- `components/SplitPane.tsx`: gains ml's `collapsed` and `rail` props.
  `ChapterPage` uses them to fold the lab with Cmd or Ctrl and `.`, and
  remembers the choice under `lg.lab.hidden`.
- `index.css`: `.slab` also remaps `--surface`, `--surface-muted`, `--muted`,
  `--strong-line`, `--error`, and `--error-soft` for the editor chrome.
- Storage keys keep the existing `lg.` prefix, so readers keep their theme,
  splits, and sidebar state.
- The backend's SPA route serves any file in `frontend/dist`, so
  `/fonts/0xproto/*` needs no server change.

## Decisions to keep

- Classes keep the `sh-` prefix. The kit's `components.css` already defines
  `.hero`, `.module-card`, and `.search-result`.
- **Match site** keeps the course design: the dark code slab, in the shade for
  the page theme. **Light** is the paper panel and **Dark** is the dark slab,
  whatever the page theme is.
- Don't change routes, backend APIs, lesson content, or the book and slide
  builds.
