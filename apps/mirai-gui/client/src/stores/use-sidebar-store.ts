import { create } from "zustand";
import { persist } from "zustand/middleware";

type SidebarState = {
  isOpen: boolean;
  isMobile: boolean;
  toggle: () => void;
  setMobile: (isMobile: boolean) => void;
  closeOnMobile: () => void;
};

export const useSidebarStore = create<SidebarState>()(
  persist(
    (set, get) => ({
      isOpen: true,
      isMobile: false,
      toggle: () => set((state) => ({ isOpen: !state.isOpen })),
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
      }),
    },
  ),
);
