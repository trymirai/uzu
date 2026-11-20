import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import type { WebSearchModes } from "@/platform/services/settings";
import type { WebSearchMode } from "@/types/web-search";

type WebSearchState = WebSearchModes & {
  fresh: Record<string, true>;
  markNew: (chatId: string) => void;
  load: () => Promise<void>;
  setMode: (chatId: string, mode: WebSearchMode) => void;
  setChatMode: (chatId: string, mode: WebSearchMode) => void;
  forgetChat: (chatId: string) => void;
  forgetAll: () => void;
};

const save = (modes: WebSearchModes): void => {
  getPlatform()
    .settings.setWebSearchModes(modes)
    .catch((error: unknown) => console.error("[settings] saving the web search modes failed", error));
};

// Only a new chat starts from the latest choice; a chat from before the modes were kept stays off.
export const webSearchModeOf = (state: WebSearchModes & Pick<WebSearchState, "fresh">, chatId: string): WebSearchMode =>
  state.chats[chatId] ?? (state.fresh[chatId] ? state.last : "off");

export const useWebSearchStore = create<WebSearchState>((set, get) => ({
  chats: {},
  last: "off",
  fresh: {},

  markNew: (chatId) => set({ fresh: { ...get().fresh, [chatId]: true } }),

  load: async () => {
    const stored = await getPlatform()
      .settings.getWebSearchModes()
      .catch((): WebSearchModes => ({ chats: {}, last: "off" }));
    set(stored);
  },

  setMode: (chatId, mode) => {
    const modes = { chats: { ...get().chats, [chatId]: mode }, last: mode };
    set(modes);
    save(modes);
  },

  setChatMode: (chatId, mode) => {
    const modes = { chats: { ...get().chats, [chatId]: mode }, last: get().last };
    set(modes);
    save(modes);
  },

  forgetChat: (chatId) => {
    const { [chatId]: forgotten, ...chats } = get().chats;
    if (forgotten === undefined) return;
    set({ chats });
    save({ chats, last: get().last });
  },

  forgetAll: () => {
    set({ chats: {} });
    save({ chats: {}, last: get().last });
  },
}));
