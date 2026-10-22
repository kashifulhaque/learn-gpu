# How to write a chapter

This is the style guide for **GPU programming for ML**, the course in this
repository: from the streaming multiprocessor to the kernels that train and
serve models. The course map, with every chapter's slug, part, and lab, is
[docs/CURRICULUM.md](../docs/CURRICULUM.md).

A chapter teaches one idea well enough that the reader can build it in the lab.
The reader is a capable engineer who writes Python and PyTorch and knows what a
transformer is, but has never written a kernel. They read on a screen, often
beside the lab editor, and they have no C++ background beyond reading it. They
skim first, then read closely. Write for both passes: someone who reads only
the callouts and the highlighted phrases leaves with the argument, and someone
who reads every line gets the full derivation.

The prose follows the
[Google developer documentation style guide](https://developers.google.com/style).
This file covers what that guide doesn't: the reading aids the app renders, and
how a chapter is shaped. Each chapter also has a lab and a slide deck. For
those, see [docs/LABS.md](../docs/LABS.md) and [docs/SLIDES.md](../docs/SLIDES.md).

## Front matter

Every chapter starts with YAML front matter. `scripts/check_chapters.py` checks
it.

```yaml
---
title: Memory and bandwidth
slug: 03-memory-and-bandwidth
part: "Part 1 — The machine"
summary: One sentence that says what the reader learns, for the contents page.
minutes: 90
gpu: true
objectives:
  - "Quote each objective, because an unquoted one that holds ': ' is a mapping."
lab: 03-bandwidth-probe
---
```

- `slug` equals the file name without `.md`, and the file name starts with the
  two-digit chapter number. [docs/CURRICULUM.md](../docs/CURRICULUM.md) fixes
  every slug, part, and lab id: copy them exactly.
- `gpu: true` marks a chapter whose lab runs on the GPU.
- `lab` names a directory in `content/labs/`. A chapter with no lab, such as
  chapter 23, leaves `lab:` out and omits the `## Lab` section. Everything else
  in the shape below still applies.

## The shape of a chapter

Every chapter follows the same outline, so a reader always knows where they are:

1. The `#` title. The page header shows it, and the body hides this copy, but
   keep it so the file reads on its own.
2. A `[!TLDR]` callout of three to five bullets: the whole chapter in one
   minute.
3. A short opening, two or three paragraphs at most, that says why the topic
   matters when you train or serve a model.
4. `## Before you start`, if the chapter needs background. Give each refresher a
   bold run-in term and one to three sentences. Move anything longer into a
   `[!DEEPDIVE]`.
5. The teaching sections, each a `##` heading. The app numbers them and tracks
   which one the reader is in, so each one should answer one question.
6. `## What goes wrong`, for the mistakes the lab is likely to provoke.
7. A `[!RECAP]` callout, as the last thing before the questions.
8. `## Check your understanding`, one `[!QUESTION]` per question.
9. `## Lab`, which opens with a `[!TRY]` callout. A chapter with no lab leaves
   this section out.
10. `## Further reading`.

## Write for a screen

- Keep paragraphs to four sentences or fewer. One idea per paragraph.
- Open each `##` section with one sentence that says what the section answers.
- Put a number, a comparison, or a sequence in a list or a table instead of a
  paragraph. A table needs three or more columns to earn its place.
- Lead with the conclusion, then justify it. "A vector add is bandwidth bound:
  it does 1 FLOP per 12 bytes against a ridge point of 161" beats building up to
  it.
- Name the symbol and the thing together the first time: "the bandwidth $\beta$,"
  not "$\beta$."
- Use contractions, the second person, the present tense, and the active voice.

## Maths

Show the maths where it carries the argument, and walk through it one step at a
time. The pattern for every display equation is: say in words what you are
about to write, write it, then say what it means.

- **Before the equation**, one sentence of intent: "Divide the arithmetic rate
  by the memory rate:"
- **After the equation**, a plain-language reading, either one sentence or, for a
  dense result, an `[!INTUITION]` callout.
- **Box the result** a derivation arrives at with `\boxed{}`, so the eye lands on
  it.
- **Colour the parts** of an equation that has several named pieces, and use the
  same colour when the prose refers to them. Four macros exist:
  `\hla{}` (green), `\hlb{}` (blue), `\hlc{}` (amber), and `\hld{}` (rose). For
  example, write `$\hla{C}$` in the prose that explains the $C$ in
  `$$I^{\ast} = \frac{\hla{C}}{\hlb{\beta}}$$`.
  Keep the colour of a symbol the same across the chapter, and colour at most
  three things in one equation.
- **Plug in real numbers** in an `[!EXAMPLE]` callout once a formula is general:
  the reader believes 1935 GB/s more than $\beta$.
- **Keep long derivations**, but when a chain runs past about four display
  equations, keep the setup and the result in the body and put the steps in a
  `[!DEEPDIVE]`. The conclusion must never live only inside a deep dive.

The site renders maths with KaTeX, which supports a subset of LaTeX. Check every
expression before you commit:

```bash
node scripts/check_math.mjs content/chapters/CHAPTER.md
```

## Reading aids

### Callouts

A callout is a blockquote whose first line is a marker. Text after the marker, on
the same line, is the callout's title. Every following line keeps the `>`
prefix, including blank lines, display maths, and code blocks:

```markdown
> [!KEY] Addends vanish at a ratio of 256
> Once the running sum is 256 times the next term, bfloat16 rounds the
> addition away entirely:
>
> $$
> \frac{S}{a} \ge 2^{8} \implies S + a = S
> $$
```

| Marker | Renders as | Use it for |
|---|---|---|
| `[!TLDR]` | In one minute | The chapter summary, straight after the title. One per chapter. |
| `[!KEY]` | Key idea, highlighted | The one thing to carry out of a section. At most one per section. |
| `[!INTUITION]` | In plain words | A plain-language reading of dense maths. |
| `[!EXAMPLE]` | Worked example | A formula with real hardware numbers plugged in. |
| `[!NOTE]` | Note | An aside that helps but that the argument doesn't need. |
| `[!TIP]` | Tip | Practical advice for the lab or for debugging. |
| `[!WARNING]` | Watch out | A mistake with a silent or expensive failure. |
| `[!DEEPDIVE]` | Deep dive, collapsed | A full derivation or a tangent. Always give it a title. |
| `[!QUESTION]` | Check yourself, collapsed | A question as the title, with the answer as the body. |
| `[!RECAP]` | What to remember | Four to six bullets before the questions. One per chapter. |
| `[!TRY]` | Try it | What the lab asks you to build, and what passing looks like. |

A callout draws the eye, so a page full of them draws it nowhere. Aim for no more
than one callout every screen or so of prose, don't put two of the same kind
next to each other, except `[!QUESTION]`, and never put the only statement of a
fact inside a collapsed callout.

To check a chapter's callouts and outline, run the following command:

```bash
python3 scripts/check_chapters.py content/chapters/CHAPTER.md
```

### Highlights

Wrap the single phrase in a section that a skimming reader must not miss in
`==double equals signs==`. It renders as a highlighter mark. Use at most two per
section, and highlight a claim, not a whole sentence:

```markdown
The error is ==a bias, not noise==, so averaging over a million terms doesn't
cancel it.
```

### Figures

A figure is a small interactive drawing: a slider over block size, a toggle
between data types, or an algorithm you can step through. To place one, write a
`viz` fence that holds only the figure's name:

````markdown
```viz
03-coalescing
```
````

Put the fence at the top level, straight after the paragraph that introduces
the idea, and use at most two figures in a chapter. A figure illustrates the
argument; it never carries a fact the prose doesn't also state. A slide deck can
use the same fence, so a figure you write for a chapter also works on a slide.

#### Where figures live

Every chapter keeps its figures in its own folder,
`frontend/src/components/viz/<NN>/`, where `<NN>` is the chapter's two-digit
number. The folder holds the figures' source files and an `index.ts` that
registers them:

```text
frontend/src/components/viz/
  kit.tsx            shared parts: Figure, Slider, color, useWidth... (do not edit)
  registry.ts        collects every <NN>/index.ts with import.meta.glob (do not edit)
  00/index.ts        an empty example: `const figures: VizRegistry = {}`
  03/
    index.ts         { "03-coalescing": Coalescing, "03-sectors": Sectors }
    coalescing.tsx   default-exports a component with no props
    sectors.tsx
```

Writers of different chapters never edit the same file, so parallel work can't
conflict: `registry.ts` finds each `index.ts` with
`import.meta.glob("./[0-9][0-9]/index.ts", { eager: true })`, and the book build,
the site, and the `viz.html` gallery all read the registry. To add a figure:

1. Create `viz/<NN>/<name>.tsx` with a default-exported component that takes no
   props.
2. In `viz/<NN>/index.ts`, import it and add a key to the map. **The key starts
   with the chapter number and a hyphen**, for example `03-coalescing`. The
   registry logs an error for a key whose number differs from its folder's, and
   for a name registered twice.
3. Write the fence in the chapter, and run `scripts/check_chapters.py`, which
   reads the keys and reports a fence that names no registered figure.

Put anything specific to your chapter, such as a helper or a constants file, in
your own folder. If a helper would help every chapter, copy it into your folder
instead of editing `kit.tsx`, which every chapter's figures share.

#### How to build one

Build a figure from the parts in `frontend/src/components/viz/kit.tsx`, and
follow these rules:

- Take every number from the chapter, `docs/FACTS.md`, or a cited source. Keep
  the constants at the top of the file and say where each one comes from.
- Draw at the width that `useWidth` measures, so labels stay legible on a phone.
- Colour marks with the tokens in `color`, so the figure follows the theme. To
  tie a mark to a coloured term in an equation, use the same maths colour.
- Don't animate on load. Use `useStepper` and a **Play** button instead.
- Choose a meaningful initial state. The PDF book prints each figure in that
  state, without its controls, so show every value a control sets in the
  drawing or the readout.

To see every figure on one page, without the backend or a sign-in, run
`npm run dev` in `frontend/` and open http://localhost:5173/viz.html. To show
only one chapter's figures, add a prefix, for example `?only=03-`.

### Everything else

- Code blocks get a language label and a copy button. Tag every fence with its
  language.
- Link to another chapter with its route, for example
  `[the roofline chapter](/c/04-roofline-and-latency)`, so it opens in place.
- Link to a chapter's slide deck with `/slides/<slug>`.
- A `*term*` in italics marks a term on first use. Don't italicize for emphasis;
  use a highlight or restructure the sentence instead.

## Accuracy

Every number in a chapter has one of three origins, and the chapter says which:

1. **Arithmetic from [docs/FACTS.md](../docs/FACTS.md).** That file, written by a
   research agent alongside the chapters, holds the hardware figures and the
   facts the course relies on, each with its source. Derive everything else from
   it with arithmetic you show or can reproduce: a ridge point, a time floor, an
   occupancy, a byte count.
2. **A cited source.** Link the source next to the number, and name the
   hardware it was measured on: "1.5x on an H100 SXM5, per the FlashAttention-3
   paper." Never quote a result without its hardware, and never compare figures
   from different cards as if they were one.
3. **Measured on the lab A100.** The lab prints its measurements, and a chapter
   may quote them once someone has run the lab.

**Chapter authors can't run the GPU.** So a value that would come from a
measurement, such as "this kernel reaches 1100 GB/s," is not yours to write.
Instead, state it as a **prediction derived from the specs**: "The roofline
predicts at least 0.5 ms; the lab measures the real figure." Or quote it from a
cited source with its hardware named. Never invent a number, never round a
prediction into something that reads like a result, and never write "typically"
about a figure you haven't derived or cited.

The one exception is figures that learn-inference measured on this same lab
card, such as the 1275 GB/s device-to-device copy ceiling, against a rating of
1935. Quote those as "measured in learn-inference, chapter 10."
`content/labs/lab_common.py` holds the same constants in its `A100` object, so a
lab and its chapter agree.

When you revise a chapter, keep every number, measurement, code sample, lab
requirement, and link intact, and recompute any new arithmetic you add. If a
measurement contradicts the tidy prediction, say so: the chapters exist partly
to show where the textbook byte count is wrong.

Bandwidth is decimal in this course: 1935 GB/s is $1.935 \times 10^{12}$ bytes
per second. Sizes of memory and caches are binary where the hardware's are:
a 40 MB L2 is 40 MiB. State the unit every time.
