import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { api, type ChapterMeta } from "./lib/api";
import { Mark } from "./components/Brand";
import ComputeBadge, { useActiveCompute } from "./components/ComputeBadge";
import { MenuIcon } from "./components/icons";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import { IconButton, Loading } from "./components/ui";
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
  return <div className="h-full overflow-y-auto">{children}</div>;
}

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [chapters, setChapters] = useState<ChapterMeta[] | null>(null);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [gpu, setGpu] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(wasCollapsed);
  const active = useActiveCompute(Boolean(name));

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

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // The sidebar still collapses, it just forgets between visits.
    }
  }, [collapsed]);

  // Cmd/Ctrl+B gives the chapter and its lab the whole screen, the way an
  // editor hides its file tree. Escape closes the narrow-screen drawer.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "b") {
        event.preventDefault();
        setCollapsed((current) => !current);
      } else if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };
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

  async function signOut() {
    await api.logout();
    setName(null);
    setChapters(null);
  }

  if (!ready) {
    return <Loading />;
  }
  if (!name) {
    return <Login onSignedIn={setName} />;
  }

  const list = chapters ?? [];

  const shell = (
    <div className="flex h-full overflow-hidden bg-paper">
      <Sidebar
        chapters={list}
        progress={progress}
        open={menuOpen}
        collapsed={collapsed}
        onNavigate={() => setMenuOpen(false)}
        onClose={() => setMenuOpen(false)}
        onToggleCollapsed={() => setCollapsed((current) => !current)}
        name={name}
        active={active}
        onSignOut={signOut}
      />

      {menuOpen && (
        <button
          aria-label="Close the menu"
          onClick={() => setMenuOpen(false)}
          className="fixed inset-0 z-20 bg-fg/25 lg:hidden"
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        {/* A wide screen has the sidebar for all of this. A narrow one needs a
            way to open it, and keeps the GPU status in view. */}
        <header className="flex h-12 shrink-0 items-center gap-1.5 border-b border-line bg-well px-2 lg:hidden">
          <IconButton onClick={() => setMenuOpen(true)} aria-label="Open the course list">
            <MenuIcon />
          </IconButton>
          <Link to="/" className="flex min-w-0 items-center gap-2 rounded-md px-1">
            <Mark className="size-5" />
            <span className="truncate text-[14px] font-semibold tracking-tight text-fg">
              learn-gpu
            </span>
          </Link>
          <div className="ml-auto">
            <ComputeBadge active={active} />
          </div>
        </header>

        <main className="min-h-0 min-w-0 flex-1 overflow-hidden">
          <Routes>
            <Route
              path="/"
              element={
                <Scroller>
                  <Dashboard
                    chapters={chapters}
                    progress={progress}
                    gpu={gpu}
                  />
                </Scroller>
              }
            />
            <Route
              path="/c/:slug"
              element={
                <ChapterPage
                  chapters={list}
                  progress={progress}
                  onProgress={changeProgress}
                />
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
      </div>
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
