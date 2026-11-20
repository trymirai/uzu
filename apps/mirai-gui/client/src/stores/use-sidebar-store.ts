import { useSyncExternalStore } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";

export const SIDEBAR_DEFAULT_WIDTH = 248;
/** The automatic width's ceiling, reached on large monitors (a 2560px-wide window). */
export const SIDEBAR_AUTO_MAX_WIDTH = 340;
export const SIDEBAR_MIN_WIDTH = 224;
export const SIDEBAR_MAX_WIDTH = 400;

type SidebarState = {
  isOpen: boolean;
  /**
   * The width the sidebar opens at, set by dragging its edge and kept while it's hidden. null until
   * then: the sidebar follows the window instead (see useSidebarWidth).
   */
  width: number | null;
  isMobile: boolean;
  chatVisibility: { chatId: string; ratio: number } | null;
  setChatVisibility: (visibility: SidebarState["chatVisibility"]) => void;
  toggle: () => void;
  setOpen: (isOpen: boolean) => void;
  /** A width from dragging or the keyboard; null goes back to following the window. */
  setWidth: (width: number | null) => void;
  setMobile: (isMobile: boolean) => void;
  closeOnMobile: () => void;
};

export const useSidebarStore = create<SidebarState>()(
  persist(
    (set, get) => ({
      isOpen: true,
      width: null,
      isMobile: false,
      chatVisibility: null,
      setChatVisibility: (chatVisibility) => set({ chatVisibility }),
      toggle: () => set((state) => ({ isOpen: !state.isOpen })),
      setOpen: (isOpen) => set({ isOpen }),
      setWidth: (width) =>
        set({
          width: width === null ? null : Math.round(Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, width))),
        }),
      setMobile: (isMobile: boolean) => {
        const currentState = get();
        const wasMobile = currentState.isMobile;

        set({ isMobile });

        if (isMobile && currentState.isOpen) {
          set({ isOpen: false });
        } else if (!isMobile && wasMobile && !currentState.isOpen) {
          set({ isOpen: true });
        }
      },
      closeOnMobile: () => {
        const state = get();
        if (state.isMobile) {
          set({ isOpen: false });
        }
      },
    }),
    {
      name: "mirai-sidebar-store",
      // v1: the width can be automatic. A saved 248 was the old fixed default, not a choice, so it
      // becomes automatic too.
      version: 1,
      migrate: (persisted, version) => {
        const state = (persisted ?? {}) as { isOpen?: boolean; isMobile?: boolean; width?: number | null };
        const width = version < 1 && state.width === SIDEBAR_DEFAULT_WIDTH ? null : (state.width ?? null);
        return { isOpen: state.isOpen ?? true, isMobile: state.isMobile ?? false, width };
      },
      partialize: (state: SidebarState) => ({
        isOpen: state.isOpen,
        isMobile: state.isMobile,
        width: state.width,
      }),
    },
  ),
);

// The automatic width grows across this range of window widths: 248px up to a laptop's full width,
// 340px from a 27" display's full width (2560).
const AUTO_GROW_FROM = 1440;
const AUTO_GROW_TO = 2560;

/** The automatic width for a window: 248px on a laptop, widening steadily to 340px on a big display. */
export const autoSidebarWidth = (windowWidth: number): number => {
  const progress = Math.min(1, Math.max(0, (windowWidth - AUTO_GROW_FROM) / (AUTO_GROW_TO - AUTO_GROW_FROM)));
  return Math.round(SIDEBAR_DEFAULT_WIDTH + progress * (SIDEBAR_AUTO_MAX_WIDTH - SIDEBAR_DEFAULT_WIDTH));
};

const subscribeWindowWidth = (onChange: () => void) => {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
};

/** The sidebar's width in effect: the one set by dragging, or else the automatic one. */
export function useSidebarWidth(): number {
  const width = useSidebarStore((s) => s.width);
  const windowWidth = useSyncExternalStore(
    subscribeWindowWidth,
    () => window.innerWidth,
    () => 1440,
  );
  return width ?? autoSidebarWidth(windowWidth);
}
