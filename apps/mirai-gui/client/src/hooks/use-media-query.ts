import { useState, useEffect } from "react";

// Absent in some test environments (jsdom).
const canMatchMedia = () => typeof window !== "undefined" && typeof window.matchMedia === "function";

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (canMatchMedia() ? window.matchMedia(query).matches : false));

  useEffect(() => {
    if (!canMatchMedia()) return;
    const media = window.matchMedia(query);
    setMatches(media.matches);

    const listener = (event: MediaQueryListEvent) => setMatches(event.matches);
    media.addEventListener("change", listener);

    return () => media.removeEventListener("change", listener);
  }, [query]);

  return matches;
}

export function useIsMobile(): boolean {
  return useMediaQuery("(max-width: 767px)");
}
