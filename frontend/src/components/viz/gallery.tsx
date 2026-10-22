// Every registered figure on one page, at the chapter column's width, for
// checking figures without the backend or a login. Development only: see
// viz.html. `?only=10-` shows the figures whose names start with 10-.

import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import "../../index.css";
import { VIZ } from "./registry";
import Markdown from "../Markdown";

function Gallery() {
  const only = new URLSearchParams(window.location.search).get("only") ?? "";
  const names = Object.keys(VIZ)
    .filter((name) => name.startsWith(only))
    .sort();
  const [narrow, setNarrow] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  function pick(next: "light" | "dark") {
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
  }

  return (
    <div className="min-h-full bg-paper">
      <div className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-line bg-well px-5 py-2 text-[13px]">
        <strong className="text-fg">{names.length} figures</strong>
        <button type="button" className="viz-button" onClick={() => pick(theme === "light" ? "dark" : "light")}>
          {theme === "light" ? "Dark" : "Light"}
        </button>
        <button type="button" className="viz-button" onClick={() => setNarrow(!narrow)} aria-pressed={narrow}>
          Phone width
        </button>
        <span className="text-fg-subtle">{names.join(" · ")}</span>
      </div>
      <div className={`mx-auto px-5 py-8 ${narrow ? "max-w-[360px]" : "max-w-[46rem]"}`}>
        {/* Through the chapter's own Markdown, so the fence plugin is exercised too. */}
        <Markdown>{names.map((name) => `\`${name}\`\n\n\`\`\`viz\n${name}\n\`\`\``).join("\n\n")}</Markdown>
      </div>
    </div>
  );
}

document.documentElement.setAttribute("data-theme", "light");
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Gallery />
  </StrictMode>,
);
