import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import AppSidebar, { type SidebarSection, type SidebarUtility } from "./components/AppSidebar";
import { Mark } from "./components/Brand";
import { computeLabel, computeTitle, useActiveCompute } from "./components/ComputeBadge";
import { useBook } from "./components/DownloadBook";
import Login from "./components/Login";
import SearchDialog from "./components/SearchDialog";
import { Loading } from "./components/ui";
import { api, type ChapterMeta } from "./lib/api";
import { chapterNumbers, groupByPart, splitPart } from "./lib/chapters";
import { doneCount, partNumber, useSearchEntries } from "./lib/shellData";
import { setThemeChoice, useThemeChoice } from "./lib/theme";
import ChapterPage from "./pages/ChapterPage";
import Compute from "./pages/Compute";
import Dashboard from "./pages/Dashboard";
import Presenter from "./pages/Presenter";
import SlidesIndex from "./pages/SlidesIndex";
import SlidesPrint from "./pages/SlidesPrint";

const COLLAPSED_KEY = "lg.sidebar.collapsed";

function wasCollapsed(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

/** Pages that flow down the screen scroll inside the shell, not with it. */
function Scroller({ children }: { children: ReactNode }) {
  return <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>;
}

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [chapters, setChapters] = useState<ChapterMeta[] | null>(null);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [gpu, setGpu] = useState("");
  // menuOpen drives the drawer on a narrow screen; collapsed folds the
  // sidebar to its icon rail on a wide one.
  const [menuOpen, setMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(wasCollapsed);
  const [searchOpen, setSearchOpen] = useState(false);
  const location = useLocation();
  const themeMode = useThemeChoice();
  const active = useActiveCompute(Boolean(name));
  const book = useBook(Boolean(name));

  const list = useMemo(() => chapters ?? [], [chapters]);
  const entries = useSearchEntries(list, searchOpen);

  useEffect(() => {
    api
      .me()
      .then((me) => {
        setName(me.name);
        setGpu(me.gpu);
      })
      .finally(() => setReady(true));
  }, []);

  useEffect(() => {
    if (!name) return;
    api.chapters().then((result) => {
      setChapters(result.chapters);
      setProgress(result.progress);
    });
  }, [name]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // The sidebar still collapses, it just forgets between visits.
    }
  }, [collapsed]);

  // Cmd or Ctrl with B folds the sidebar to its rail on a wide screen and
  // opens or closes the drawer on a narrow one, the way an editor hides its
  // file tree.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key !== "b" && event.key !== "B") return;
      event.preventDefault();
      if (window.matchMedia("(min-width: 1024px)").matches) setCollapsed((v) => !v);
      else setMenuOpen((v) => !v);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // "/" opens the search when the keyboard is not busy in a field or the
  // editor. Cmd or Ctrl with K opens it from anywhere.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
          !!target.closest(".monaco-editor"));
      const slash = event.key === "/" && !event.metaKey && !event.ctrlKey && !event.altKey && !typing;
      const commandK = (event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k";
      if (!slash && !commandK) return;
      event.preventDefault();
      setMenuOpen(false);
      setSearchOpen(true);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const changeProgress = useCallback(
    (slug: string, status: "in_progress" | "done") => {
      setProgress((current) => ({ ...current, [slug]: status }));
      api.setProgress(slug, status).catch(() => undefined);
    },
    [],
  );

  const currentSlug = location.pathname.startsWith("/c/") ? location.pathname.slice("/c/".length) : null;

  const sections = useMemo<SidebarSection[]>(() => {
    const numbers = chapterNumbers(list);
    return groupByPart(list).map(([part, items], index) => ({
      id: part,
      number: partNumber(part, index),
      title: splitPart(part).name,
      lessons: items.map((chapter) => {
        return {
          id: chapter.slug,
          to: `/c/${chapter.slug}`,
          title: chapter.title,
          number: numbers.get(chapter.slug),
          done: progress[chapter.slug] === "done",
          current: chapter.slug === currentSlug,
        };
      }),
    }));
  }, [list, progress, currentSlug]);

  const utilities = useMemo<SidebarUtility[]>(
    () =>
      book
        ? [
            {
              id: "book",
              label: "Download book",
              detail: `PDF · ${book.pages} pages · ${(book.bytes / 1e6).toFixed(1)} MB`,
              icon: "download",
              href: book.url,
              download: true,
            },
          ]
        : [],
    [book],
  );

  if (!ready) {
    return <Loading />;
  }
  if (!name) {
    return <Login />;
  }

  const done = doneCount(list, progress);
  const running = (active ?? 0) > 0;

  const shell = (
    <div className="sh-app">
      <AppSidebar
        brand={{ name: "learn/gpu", kicker: "Field notes", mark: <Mark className="size-[30px]" /> }}
        searchLabel="Find a chapter"
        onSearch={() => {
          setMenuOpen(false);
          setSearchOpen(true);
        }}
        nav={[
          { to: "/", label: "Course overview", icon: "grid", active: location.pathname === "/" },
          { to: "/slides", label: "Slides", icon: "slides", active: location.pathname === "/slides" },
          {
            to: "/compute",
            label: "Compute",
            icon: "gpu",
            count: running ? (active ?? undefined) : undefined,
            active: location.pathname === "/compute",
          },
        ]}
        sections={sections}
        storageKey="lg.sidebar"
        progress={{
          done,
          total: list.length,
          label: "Chapters done",
          caption: done === list.length && list.length > 0 ? "Every chapter done." : `Saved for ${name}`,
        }}
        utilities={utilities}
        status={{
          label: `GPU ${computeLabel(active).toLowerCase()}`,
          title: computeTitle(active),
          tone: active === null ? "neutral" : running ? "alert" : "ok",
        }}
        theme={{ mode: themeMode, onChange: setThemeChoice }}
        account={{ username: name, signOutHref: "https://auth.ifkash.dev/logout" }}
        open={menuOpen}
        onOpen={() => setMenuOpen(true)}
        onClose={() => setMenuOpen(false)}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((v) => !v)}
      />

      <main className="sh-main" id="main">
        <Routes>
          <Route
            path="/"
            element={<Dashboard chapters={chapters} progress={progress} gpu={gpu} />}
          />
          <Route
            path="/c/:slug"
            element={
              <div className="min-h-0 flex-1">
                <ChapterPage chapters={list} progress={progress} onProgress={changeProgress} />
              </div>
            }
          />
          <Route
            path="/slides"
            element={
              <Scroller>
                <SlidesIndex />
              </Scroller>
            }
          />
          <Route
            path="/compute"
            element={
              <Scroller>
                <Compute />
              </Scroller>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <SearchDialog
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        entries={entries}
        noun="chapter"
        placeholder="Try “coalescing”, “tensor cores”, or “flash attention”"
      />
    </div>
  );

  // The slide views take the whole window, so they sit outside the shell.
  return (
    <Routes>
      <Route path="/slides/:slug" element={<Presenter mode="presenter" />} />
      <Route path="/slides/:slug/present" element={<Presenter mode="present" />} />
      <Route path="/slides/:slug/print" element={<SlidesPrint />} />
      <Route path="*" element={shell} />
    </Routes>
  );
}
