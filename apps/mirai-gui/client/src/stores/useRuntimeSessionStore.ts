import type { RuntimeSessionRef } from "@/types/session";
import { create } from "zustand";

export type RuntimeSessionState = {
  residentSession: RuntimeSessionRef | null;
  loadingSession: RuntimeSessionRef | null;
  ejectingSession: RuntimeSessionRef | null;
  setResidentSession: (session: RuntimeSessionRef | null) => void;
  startLoadingSession: (session: RuntimeSessionRef) => void;
  endLoadingSession: () => void;
  setEjectingSession: (session: RuntimeSessionRef | null) => void;
};

export const useRuntimeSessionStore = create<RuntimeSessionState>((set) => ({
  residentSession: null,
  loadingSession: null,
  ejectingSession: null,

  setResidentSession: (session) =>
    set((state) => ({
      residentSession: session,
      loadingSession: state.loadingSession?.repoId === session?.repoId ? null : state.loadingSession,
      ejectingSession: state.ejectingSession?.repoId === session?.repoId ? null : state.ejectingSession,
    })),

  startLoadingSession: (session) => set({ loadingSession: session }),

  endLoadingSession: () => set({ loadingSession: null }),

  setEjectingSession: (session) => set({ ejectingSession: session }),
}));
