import { useEffect } from "react";

// A change of width this large in one step is the window jumping (into or out of full screen, or
// maximized), not an edge being dragged, which moves a few points at a time.
const JUMP_PX = 120;
const MORPH_MS = 450;

/**
 * Glides the layout that follows the window's size (the sidebar's width, the chat column, the
 * sidebar button) when the window jumps to a new size, instead of snapping to it in one frame.
 * Marks the page with `data-window-morph` for a moment; index.css turns the transitions on for
 * `.follows-window` while it's there. Dragging an edge stays instant.
 */
export function useWindowJumpMorph() {
  useEffect(() => {
    const root = document.documentElement;
    let lastWidth = window.innerWidth;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onResize = () => {
      const width = window.innerWidth;
      const jumped = Math.abs(width - lastWidth) >= JUMP_PX;
      lastWidth = width;
      if (!jumped) return;
      // Set in the same frame as the new size, so the change itself is what transitions.
      root.setAttribute("data-window-morph", "");
      clearTimeout(timer);
      timer = setTimeout(() => root.removeAttribute("data-window-morph"), MORPH_MS);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      clearTimeout(timer);
      root.removeAttribute("data-window-morph");
    };
  }, []);
}
