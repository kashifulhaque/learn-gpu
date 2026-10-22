import { useSyncExternalStore } from "react";

/**
 * The colour theme. "system" follows the operating system; the other two pin a
 * theme. The choice is stored per browser, and index.html applies it before the
 * first paint so a reload never flashes the wrong theme.
 */
export type ThemeChoice = "system" | "light" | "dark";
export type Theme = "light" | "dark";

const KEY = "lg.theme";
const DARK = "(prefers-color-scheme: dark)";

function readChoice(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

let choice: ThemeChoice = typeof window === "undefined" ? "system" : readChoice();
const listeners = new Set<() => void>();

function resolve(value: ThemeChoice): Theme {
  if (value !== "system") return value;
  return window.matchMedia(DARK).matches ? "dark" : "light";
}

function emit() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const media = window.matchMedia(DARK);
  media.addEventListener("change", listener);
  return () => {
    listeners.delete(listener);
    media.removeEventListener("change", listener);
  };
}

export function setThemeChoice(next: ThemeChoice) {
  choice = next;
  const root = document.documentElement;
  if (next === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", next);
  try {
    if (next === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, next);
  } catch {
    // The theme still changes; it just isn't remembered.
  }
  emit();
}

/** The reader's choice, which may be "system". */
export function useThemeChoice(): ThemeChoice {
  return useSyncExternalStore(subscribe, () => choice);
}

/** The theme actually on screen. */
export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, () => resolve(choice));
}
