import { useEffect, useState } from "react";

/**
 * Tracks a CSS media query from React, for the few places where a layout has to
 * branch in JavaScript rather than in a Tailwind breakpoint — a split view that
 * becomes a pair of tabs on a narrow screen cannot simply hide one half.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === "undefined" ? false : window.matchMedia(query).matches,
  );

  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);

  return matches;
}

/** True from Tailwind's `lg` breakpoint up, where the split view is worth it. */
export function useIsWide(): boolean {
  return useMediaQuery("(min-width: 1024px)");
}
