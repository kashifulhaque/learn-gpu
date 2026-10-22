import { useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import macros from "../lib/katexMacros.json";
import { rehypeSectionize, remarkCallouts, remarkMark, remarkViz } from "../lib/markdownPlugins";
import MermaidBlock from "./MermaidBlock";
import Viz from "./viz/Viz";

/** A link that stays inside the app, as opposed to one that leaves it. */
function isInternal(href: string): boolean {
  return href.startsWith("/") && !href.startsWith("//");
}

/** The language rehype-highlight tagged a fenced block with, if any. */
function languageOf(children: ReactNode): string {
  const child = Array.isArray(children) ? children[0] : children;
  const className =
    child && typeof child === "object" && "props" in child
      ? String((child.props as { className?: string }).className ?? "")
      : "";
  const match = /language-([\w+-]+)/.exec(className);
  return match ? match[1] : "";
}

function CodeBlock({ children }: { children: ReactNode }) {
  const pre = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const language = languageOf(children);

  function copy() {
    const text = pre.current?.innerText ?? "";
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => undefined);
  }

  return (
    <div className="code-block">
      <div className="code-block-bar">
        <span>{language || "text"}</span>
        <button type="button" onClick={copy} aria-label="Copy the code">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre ref={pre}>{children}</pre>
    </div>
  );
}

/** The text inside a hast node, however deeply the highlighter nested it. */
function nodeText(node: unknown): string {
  if (!node || typeof node !== "object") return "";
  const candidate = node as { type?: string; value?: string; children?: unknown[] };
  if (candidate.type === "text") return candidate.value ?? "";
  if (Array.isArray(candidate.children)) return candidate.children.map(nodeText).join("");
  return "";
}

/**
 * - `chapter`: the reading layout, split into sections.
 * - `slide`: a slide body. Questions read "You might ask" and start closed, so
 *   a presenter can pose one and then click to answer it.
 * - `print`: a slide body with every answer showing, for the PDF.
 * - `inline`: one line of text, such as a slide's title, with no block wrapper.
 */
export type MarkdownVariant = "chapter" | "slide" | "print" | "inline";

const SLIDE_LABELS = { QUESTION: "You might ask" };

export default function Markdown({
  children,
  variant = "chapter",
}: {
  children: string;
  variant?: MarkdownVariant;
}) {
  const slide = variant !== "chapter";
  const inline = variant === "inline";
  const Wrapper = inline ? "span" : "div";
  return (
    <Wrapper className={inline ? "inline-md" : "prose-chapter"}>
      <ReactMarkdown
        remarkPlugins={[
          remarkGfm,
          remarkMath,
          remarkViz,
          [remarkCallouts, slide ? { labels: SLIDE_LABELS, open: variant === "print" } : {}],
          remarkMark,
        ]}
        rehypePlugins={[
          // Sections first, so each one wraps the source headings before KaTeX
          // and the highlighter add their own markup. A slide has no sections.
          ...(slide ? [] : [rehypeSectionize]),
          // KaTeX mutates the macro table it is given, so hand it a copy. The
          // colour macros expand to \htmlClass, the one HTML extension allowed,
          // so the theme decides their colour. scripts/check_math.mjs matches.
          [
            rehypeKatex,
            {
              macros: { ...macros },
              strict: (code: string) => (code === "htmlExtension" ? "ignore" : "warn"),
              trust: (context: { command: string }) => context.command === "\\htmlClass",
            },
          ],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
        components={{
          ...(inline ? { p: ({ children }) => <>{children}</> } : {}),
          a: ({ href, children }) =>
            // A cross-reference to another chapter routes in place. Sending it
            // to a new tab would lose the reader's scroll position and their
            // lab draft.
            href && isInternal(href) ? (
              <Link to={href}>{children}</Link>
            ) : (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            ),
          // remarkViz leaves a placeholder where a ```viz fence was.
          div: ({ node: _node, ...props }) => {
            const name = (props as Record<string, unknown>)["data-viz"];
            return typeof name === "string" && String(props.className).includes("viz-embed") ? (
              <Viz name={name} />
            ) : (
              <div {...props} />
            );
          },
          pre: ({ node, children }) => {
            const code = (node as { children?: { tagName?: string; properties?: { className?: unknown } }[] } | undefined)
              ?.children?.[0];
            const names = code?.properties?.className;
            const list = Array.isArray(names) ? names.map(String) : [String(names ?? "")];
            if (code?.tagName === "code" && list.includes("language-mermaid")) {
              return <MermaidBlock source={nodeText(code).replace(/\n$/, "")} />;
            }
            return <CodeBlock>{children}</CodeBlock>;
          },
          table: ({ children }) => (
            <div className="table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </Wrapper>
  );
}
