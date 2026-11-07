import { create } from "zustand";
import { persist } from "zustand/middleware";

export const SIDEBAR_DEFAULT_WIDTH = 248;
export const SIDEBAR_MIN_WIDTH = 224;
export const SIDEBAR_MAX_WIDTH = 400;

type SidebarState = {
  isOpen: boolean;
  /** The width the sidebar opens at, set by dragging its edge; kept while it's hidden. */
  width: number;
  isMobile: boolean;
  chatVisibility: { chatId: string; ratio: number } | null;
  setChatVisibility: (visibility: SidebarState["chatVisibility"]) => void;
  toggle: () => void;
  setOpen: (isOpen: boolean) => void;
  setWidth: (width: number) => void;
  setMobile: (isMobile: boolean) => void;
  closeOnMobile: () => void;
};

export const useSidebarStore = create<SidebarState>()(
  persist(
    (set, get) => ({
      isOpen: true,
      width: SIDEBAR_DEFAULT_WIDTH,
      isMobile: false,
      chatVisibility: null,
      setChatVisibility: (chatVisibility) => set({ chatVisibility }),
      toggle: () => set((state) => ({ isOpen: !state.isOpen })),
      setOpen: (isOpen) => set({ isOpen }),
      setWidth: (width) => set({ width: Math.round(Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, width))) }),
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
      partialize: (state: SidebarState) => ({
        isOpen: state.isOpen,
        isMobile: state.isMobile,
        width: state.width,
      }),
    },
  ),
);
