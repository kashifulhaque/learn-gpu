// Syntax tree plugins that give a chapter its reading aids. Authors write plain
// Markdown; content/STYLE.md documents the syntax these plugins recognize.
//
// - remarkCallouts turns `> [!KEY] Title` blockquotes into styled callouts, and
//   the collapsible kinds into <details>.
// - remarkMark turns `==text==` into <mark>.
// - remarkViz turns a ```viz fence into a placeholder that Markdown.tsx swaps
//   for the interactive figure it names.
// - rehypeSectionize wraps each `##` section in a <section>, so the reader can
//   track, number, and focus one section at a time.

// The trees are walked structurally, so a loose node type is enough and keeps
// these plugins free of the unified type packages.
type Node = {
  type: string;
  value?: string;
  lang?: string | null;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: Node[];
  data?: Record<string, unknown>;
};

type CalloutSpec = { label: string; collapsible?: boolean; hint?: string };

/** Every callout kind, keyed by the marker authors write. */
export const CALLOUTS: Record<string, CalloutSpec> = {
  TLDR: { label: "In one minute" },
  KEY: { label: "Key idea" },
  INTUITION: { label: "In plain words" },
  EXAMPLE: { label: "Worked example" },
  NOTE: { label: "Note" },
  TIP: { label: "Tip" },
  WARNING: { label: "Watch out" },
  RECAP: { label: "What to remember" },
  TRY: { label: "Try it" },
  DEEPDIVE: { label: "Deep dive", collapsible: true, hint: "Expand" },
  QUESTION: { label: "Check yourself", collapsible: true, hint: "Show answer" },
};

// GitHub's alert names, so a chapter written for GitHub still renders.
const ALIASES: Record<string, string> = {
  IMPORTANT: "KEY",
  CAUTION: "WARNING",
};

const MARKER = /^\[!([A-Za-z]+)\][ \t]*/;

function element(hName: string, className: string, children: Node[]): Node {
  // `paragraph` is only a carrier: `data.hName` decides the element it becomes.
  return { type: "paragraph", data: { hName, hProperties: { className: [className] } }, children };
}

function isBlank(node: Node): boolean {
  return node.type === "text" && !node.value?.trim();
}

/** Removes the first line of a paragraph's inline content and returns it. */
function takeFirstLine(paragraph: Node): Node[] {
  const line: Node[] = [];
  const inline = paragraph.children ?? [];
  while (inline.length > 0) {
    const node = inline[0];
    if (node.type === "break") {
      inline.shift();
      break;
    }
    if (node.type === "text" && node.value !== undefined) {
      const newline = node.value.indexOf("\n");
      if (newline >= 0) {
        const head = node.value.slice(0, newline);
        node.value = node.value.slice(newline + 1);
        if (head) line.push({ type: "text", value: head });
        break;
      }
    }
    line.push(inline.shift()!);
  }
  const last = line[line.length - 1];
  if (last?.type === "text" && last.value) last.value = last.value.trimEnd();
  return line.filter((node) => !isBlank(node));
}

export type CalloutOptions = {
  /** Replaces a kind's label, for example QUESTION on a slide. */
  labels?: Record<string, string>;
  /** Renders the collapsible kinds expanded, as the printed layouts need. */
  open?: boolean;
};

function toCallout(quote: Node, options: CalloutOptions): void {
  const first = quote.children?.[0];
  const text = first?.type === "paragraph" ? first.children?.[0] : undefined;
  if (!first || text?.type !== "text" || text.value === undefined) return;

  const match = MARKER.exec(text.value);
  if (!match) return;
  const kind = ALIASES[match[1].toUpperCase()] ?? match[1].toUpperCase();
  const spec = CALLOUTS[kind];
  if (!spec) return;

  text.value = text.value.slice(match[0].length);
  const title = takeFirstLine(first);
  const body = quote.children!;
  if ((first.children ?? []).every(isBlank)) body.shift();

  const label = options.labels?.[kind] ?? spec.label;
  const head: Node[] = [element("span", "callout-label", [{ type: "text", value: label }])];
  if (title.length > 0) head.push(element("span", "callout-title", title));
  if (spec.hint) head.push(element("span", "callout-hint", [{ type: "text", value: spec.hint }]));

  const children: Node[] = [element(spec.collapsible ? "summary" : "div", "callout-head", head)];
  if (body.length > 0) {
    children.push({
      type: "blockquote",
      data: { hName: "div", hProperties: { className: ["callout-body"] }, calloutBody: true },
      children: body,
    });
  }

  quote.children = children;
  quote.data = {
    hName: spec.collapsible ? "details" : "aside",
    hProperties: {
      className: ["callout", `callout-${kind.toLowerCase()}`],
      ...(spec.collapsible && options.open ? { open: true } : {}),
    },
  };
}

export function remarkCallouts(options: CalloutOptions = {}) {
  const walk = (node: Node) => {
    for (const child of node.children ?? []) {
      if (child.type === "blockquote" && !child.data?.calloutBody) toCallout(child, options);
      walk(child);
    }
  };
  return (tree: Node) => walk(tree);
}

// A highlight can wrap a line and hold inline code or maths, so it is matched
// across a parent's inline children rather than inside one text node. Each
// marker must hug its text, so `a == b` in prose stays prose.
const PHRASING_PARENTS = new Set([
  "paragraph", "heading", "tableCell", "emphasis", "strong", "delete", "link",
]);

type Piece = { node: Node } | { marker: true; opens: boolean; closes: boolean };

function markInline(parent: Node): void {
  const pieces: Piece[] = [];
  const children = parent.children ?? [];
  children.forEach((child, index) => {
    if (child.type !== "text" || !child.value?.includes("==")) {
      pieces.push({ node: child });
      return;
    }
    const value = child.value;
    let last = 0;
    for (let at = value.indexOf("=="); at >= 0; at = value.indexOf("==", last)) {
      if (at > last) pieces.push({ node: { type: "text", value: value.slice(last, at) } });
      const before = at > 0 ? value[at - 1] : index > 0 ? "x" : " ";
      const after = at + 2 < value.length ? value[at + 2] : index < children.length - 1 ? "x" : " ";
      pieces.push({ marker: true, opens: /\S/.test(after), closes: /\S/.test(before) });
      last = at + 2;
    }
    if (last < value.length) pieces.push({ node: { type: "text", value: value.slice(last) } });
  });
  if (!pieces.some((piece) => "marker" in piece)) return;

  const out: Node[] = [];
  let open: { at: number; inner: Node[] } | null = null;
  const literal = (): Node => ({ type: "text", value: "==" });
  for (const piece of pieces) {
    if (!("marker" in piece)) {
      (open ? open.inner : out).push(piece.node);
    } else if (open && piece.closes && open.inner.length > 0) {
      out.push({ type: "emphasis", data: { hName: "mark" }, children: open.inner });
      open = null;
    } else if (!open && piece.opens) {
      open = { at: out.length, inner: [] };
    } else {
      (open ? open.inner : out).push(literal());
    }
  }
  // An opener that never closes was prose after all.
  if (open) out.push(literal(), ...open.inner);
  parent.children = out;
}

export function remarkMark() {
  const walk = (node: Node) => {
    if (PHRASING_PARENTS.has(node.type)) markInline(node);
    for (const child of node.children ?? []) walk(child);
  };
  return (tree: Node) => walk(tree);
}

/**
 * Replaces each ```viz fence with an empty <div class="viz-embed">. The fence
 * holds only the figure's name, for example `10-roofline-explorer`, and the
 * registry in components/viz decides what renders.
 */
export function remarkViz() {
  const walk = (node: Node) => {
    const children = node.children ?? [];
    children.forEach((child, index) => {
      if (child.type === "code" && child.lang === "viz") {
        const name = (child.value ?? "").trim().split(/\s+/)[0] ?? "";
        children[index] = {
          type: "paragraph",
          data: { hName: "div", hProperties: { className: ["viz-embed"], dataViz: name } },
          children: [],
        };
      } else {
        walk(child);
      }
    });
  };
  return (tree: Node) => walk(tree);
}

/** Wraps the chapter's intro and each `##` section in its own <section>. */
export function rehypeSectionize() {
  return (tree: Node) => {
    const sections: Node[] = [];
    let index = 0;
    let current: Node = {
      type: "element",
      tagName: "section",
      properties: { className: ["chapter-section", "chapter-intro"], dataSection: "0" },
      children: [],
    };
    for (const child of tree.children ?? []) {
      if (child.type === "element" && child.tagName === "h2") {
        if (current.children!.some((node) => !isBlank(node))) sections.push(current);
        index += 1;
        child.properties = { ...child.properties, id: `s-${index}` };
        current = {
          type: "element",
          tagName: "section",
          properties: { className: ["chapter-section"], dataSection: String(index) },
          children: [],
        };
      }
      current.children!.push(child);
    }
    if (current.children!.some((node) => !isBlank(node))) sections.push(current);
    tree.children = sections;
  };
}
