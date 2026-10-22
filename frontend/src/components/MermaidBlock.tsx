import { useEffect, useId, useRef, useState } from "react";

/**
 * One Mermaid diagram, from a fenced ```mermaid block.
 *
 * Mermaid is large, so it loads on first use rather than with the site. The
 * diagram takes its colours from the site's theme tokens, read where the
 * block sits and draws again when the
 * reader switches between light and dark. A block that doesn't parse shows
 * the error as a visible note, the way a missing demo does.
 */

type MermaidApi = typeof import("mermaid").default;

let loading: Promise<MermaidApi> | null = null;

function loadMermaid(): Promise<MermaidApi> {
  loading ??= import("mermaid").then((module) => module.default);
  return loading;
}

/** A `#rrggbb` colour mixed toward another, `amount` of the way. Anything else comes back unchanged. */
function mix(from: string, to: string, amount: number): string {
  const parse = (hex: string) => {
    const match = /^#([0-9a-f]{6})$/i.exec(hex.trim());
    if (!match) return null;
    const n = parseInt(match[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const a = parse(from);
  const b = parse(to);
  if (!a || !b) return from;
  return `#${a.map((v, i) => Math.round(v + (b[i] - v) * amount).toString(16).padStart(2, "0")).join("")}`;
}

/** Mermaid's theme variables, from the CSS tokens in force at `node`. */
function themeVariables(node: Element): { dark: boolean; vars: Record<string, unknown> } {
  const style = getComputedStyle(node);
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  const dark = document.documentElement.dataset.theme === "dark";
  const ink = token("--fg", "#1d1b18");
  const ink3 = token("--fg-subtle", "#706a60");
  const raise = token("--card", "#ffffff");
  const chrome = token("--well", "#f4f2ed");
  const sunk = token("--tint", "#ebe8e1");
  const rule = token("--line-strong", "#cdc7ba");
  const tier = token("--accent", "#c2410c");
  const accent = token("--info", "#2c5ea8");
  const good = token("--ok", "#2d7a4c");
  const warn = token("--warn", "#9a5b06");
  const bad = token("--bad", "#b42318");
  const purple = token("--violet", "#6d4fb0");
  const teal = token("--hl-b", "#2c5ea8");
  const font = token("--font-sans", "ui-sans-serif, system-ui, sans-serif");
  const series = [tier, warn, good, purple, bad, teal, accent, ink3];
  const tint = (colour: string, amount = 0.82) => mix(colour, raise, amount);

  const vars: Record<string, unknown> = {
    darkMode: dark,
    fontFamily: font,
    fontSize: "19px",
    background: raise,
    textColor: ink,
    primaryColor: tint(tier),
    primaryTextColor: ink,
    primaryBorderColor: tier,
    secondaryColor: tint(warn),
    secondaryTextColor: ink,
    secondaryBorderColor: warn,
    tertiaryColor: chrome,
    tertiaryTextColor: ink,
    tertiaryBorderColor: rule,
    lineColor: ink3,
    mainBkg: tint(tier),
    nodeBorder: tier,
    nodeTextColor: ink,
    clusterBkg: sunk,
    clusterBorder: rule,
    titleColor: ink,
    edgeLabelBackground: raise,
    noteBkgColor: chrome,
    noteTextColor: ink,
    noteBorderColor: rule,
    // Pie slices, strong colours with the page behind the labels.
    pieTitleTextColor: ink,
    pieSectionTextColor: raise,
    pieLegendTextColor: ink,
    pieStrokeColor: raise,
    pieOuterStrokeColor: rule,
    pieOpacity: "1",
    // Quadrants, lightest at the top right.
    quadrant1Fill: tint(tier, 0.88),
    quadrant2Fill: tint(good, 0.88),
    quadrant3Fill: tint(ink3, 0.9),
    quadrant4Fill: tint(warn, 0.88),
    quadrant1TextFill: ink,
    quadrant2TextFill: ink,
    quadrant3TextFill: ink,
    quadrant4TextFill: ink,
    quadrantPointFill: tier,
    quadrantPointTextFill: ink,
    quadrantXAxisTextFill: ink,
    quadrantYAxisTextFill: ink,
    quadrantInternalBorderStrokeFill: rule,
    quadrantExternalBorderStrokeFill: rule,
    quadrantTitleFill: ink,
    xyChart: {
      backgroundColor: raise,
      titleColor: ink,
      xAxisLabelColor: ink3,
      xAxisTitleColor: ink,
      xAxisTickColor: ink3,
      xAxisLineColor: rule,
      yAxisLabelColor: ink3,
      yAxisTitleColor: ink,
      yAxisTickColor: ink3,
      yAxisLineColor: rule,
      plotColorPalette: series.join(", "),
    },
  };
  series.forEach((colour, i) => {
    vars[`pie${i + 1}`] = colour;
  });
  // Timeline sections and other scaled colours: light tints with dark text.
  for (let i = 0; i < 12; i += 1) {
    const colour = series[i % series.length];
    vars[`cScale${i}`] = tint(colour, 0.78);
    vars[`cScaleLabel${i}`] = ink;
    vars[`cScalePeer${i}`] = colour;
  }
  return { dark, vars };
}

/** Re-renders when <html> changes its theme. */
function useThemeStamp(): string {
  const read = () => document.documentElement.dataset.theme ?? "";
  const [stamp, setStamp] = useState(read);
  useEffect(() => {
    const observer = new MutationObserver(() => setStamp(read()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return stamp;
}

/** Matches `CONTENT_CHANGE` in SlideFrame, kept apart so Markdown needn't import the slide code. */
const CONTENT_CHANGE = "lg:contentchange";

export default function MermaidBlock({ source }: { source: string }) {
  const id = `mermaid-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const box = useRef<HTMLDivElement>(null);
  const stamp = useThemeStamp();
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const node = box.current;
    if (!node) return;
    loadMermaid()
      .then(async (mermaid) => {
        const { vars } = themeVariables(node);
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          themeVariables: vars,
          fontFamily: String(vars.fontFamily),
          // A chart sits in a short, wide slot on a 16:9 slide, so draw it
          // wide and with type large enough to read from the back of a room.
          xyChart: {
            width: 760,
            height: 380,
            titleFontSize: 22,
            xAxis: { labelFontSize: 18, titleFontSize: 18 },
            yAxis: { labelFontSize: 18, titleFontSize: 18 },
          },
          flowchart: { useMaxWidth: true, nodeSpacing: 28, rankSpacing: 36, padding: 10, wrappingWidth: 180 },
        });
        // Parse first: a failed render can leave Mermaid's own error drawing
        // in the page, and parse fails cleanly.
        await mermaid.parse(source);
        const result = await mermaid.render(`${id}-${Date.now().toString(36)}`, source);
        if (!cancelled) {
          setSvg(result.svg);
          setError(null);
        }
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        const message = caught instanceof Error ? caught.message : String(caught);
        setError(message.trim() || "Mermaid couldn't read this diagram.");
        setSvg(null);
      });
    return () => {
      cancelled = true;
    };
  }, [id, source, stamp]);

  // Tell a slide around the diagram that its content changed size.
  useEffect(() => {
    box.current?.dispatchEvent(new Event(CONTENT_CHANGE, { bubbles: true }));
  }, [svg, error]);

  return (
    <div ref={box} className="mermaid-wrap">
      {error ? (
        <div className="viz-missing" role="note">
          <p>This diagram has a syntax error, so it can't be drawn.</p>
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-[12px]">{error}</pre>
        </div>
      ) : (
        <div
          className="mermaid-block"
          role="img"
          aria-label="Diagram"
          aria-busy={svg === null}
          // Mermaid sanitises the SVG itself under `securityLevel: "strict"`.
          dangerouslySetInnerHTML={svg === null ? undefined : { __html: svg }}
        />
      )}
    </div>
  );
}
