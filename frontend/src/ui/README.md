<!-- Generated from weights-and-wires/sites ui/ at b3af704 by scripts/sync-ui.sh. Do not edit here. -->

# Course design system

This directory is the single source of truth for the look of every ifkash.dev
course site: papers.ifkash.dev, courses.ifkash.dev, and the five course apps
(ml, rl, speech, learn-gpu, and learn-inference). It captures the arm64
Fieldnotes design: warm paper, green-tinted ink, serif headings, mono kicker
labels, hairline borders, and a dark-green code slab.

## Files

- `tokens.css`: the palette for light and dark themes, plot series, code-slab
  and syntax colours, and compatibility aliases for the two course-app families.
- `type.css`: system font stacks, 16px reading prose with a 740px measure,
  serif headings, and the `.kicker` and `.meta` labels.
- `components.css`: buttons, chips, the hero, module cards, lesson anatomy,
  callouts, code blocks, sidebar rows, progress tracks, lesson navigation, the
  table of contents, and the `/` search dialog.
- `theme.js`: a blocking theme boot script and a toggle that stores the choice.
- `monaco-theme.ts`: Monaco editor themes that match the code slab.

## Tokens

The palette uses arm64's names. Components read only these variables, so dark
mode is one re-declaration of the same names.

| Token | Role |
|---|---|
| `--paper`, `--surface`, `--surface-muted` | Page, raised panel, and sunken surfaces |
| `--ink`, `--muted` | Body text and secondary text |
| `--forest`, `--forest-soft` | Links, active states, progress, and their tint |
| `--accent`, `--accent-ink`, `--accent-soft` | Callout bar, kicker text, and warm tint |
| `--line`, `--strong-line` | Hairline borders |
| `--error`, `--error-soft` | Failed checks and warnings |
| `--code-*`, `--hl-*` | The code slab and its syntax colours, the same in both themes |
| `--code-ok`, `--code-error` | Pass and fail text on the slab (`#b5d49b` and `#f0a49a`) |
| `--slab-ok`, `--slab-bad` | Compatibility names for `--code-ok` and `--code-error`. Apps can delete their own definitions |
| `--hl-c` | The third maths colour. It is `--series-2`, the figures' terracotta, in both themes. Don't redefine it |
| `--series-1` to `--series-6`, `--demo` | Plot and figure colours |

With no `data-theme` attribute on `<html>`, the page follows the system
setting. `data-theme="light"` or `data-theme="dark"` pins a theme.

The aliases block maps the course apps' own names onto the palette. Family A
(ml, rl, and speech) uses `--canvas`, `--chrome`, `--sunk`, `--raise`, `--rule`,
`--ink-2` to `--ink-4`, `--good`, and `--bad`. Family B (learn-gpu and
learn-inference) uses `--well`, `--card`, `--tint`, `--fg` to `--fg-faint`,
`--ok`, and `--syn-*`. Neither family's `--accent` is aliased, because both apps
use it for interactive colour. Map the app's Tailwind accent colour to
`--forest` in its `index.css`.

## Light code slab for print

The code slab is dark in both themes. To draw it as a light panel with
matching syntax colours, add `class="code-light"` or `data-code="light"` to
any container, such as the `.book` root. Under `@media print` the kit applies
the same palette to the whole page, so a print stylesheet needs no `--code-*`
or `--hl-*` overrides of its own. The panel stays light even when the page
theme is dark.

## Table of contents label

The "On this page" label is `.toc-label`. It was `.toc-title` in earlier
versions. Apps use `.toc-title` for their own purposes, so the kit no longer
defines it. Rename any use of the kit's label when you sync.

## Layers

`type.css` and `components.css` put their rules in the `base` and
`components` cascade layers. In a Tailwind 4 app, a utility class on the same
element still wins. On a plain page, declare the layer order first:

```css
@layer base, components;
```

## Sync the design system into a course repo

The course apps vendor a copy of this directory. To change the design, edit the
files here, then copy them into each course repo:

1. In a worktree of the course repo, run the sync script from this repo:

   ```bash
   SITES_DIR/scripts/sync-ui.sh COURSE_DIR
   ```

   Replace the following:

   - `SITES_DIR`: your checkout of `weights-and-wires/sites`.
   - `COURSE_DIR`: the course repo worktree.

   The script writes the files to `frontend/src/ui/` with a "generated, do not
   edit" header and prints a diff stat.

1. Run `npm run build` in the course repo's `frontend/` directory, and then
   open a pull request.

Don't edit the copies in `frontend/src/ui/`. The next sync overwrites them.
