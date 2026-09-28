import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import type { StorageCleanupPreview } from "@/platform/services/storage";

type CleanupCategory = "dialogs" | "models" | "logs";

type CleanupResult = {
  executed: string[];
  modelsSkipped: string[];
};

type CleanupStoreState = {
  preview: StorageCleanupPreview | null;
  previewLoading: boolean;
  result: CleanupResult | null;
  executing: boolean;
  fetchPreview: (skipModelIdentifiers?: string[]) => Promise<void>;
  execute: (categories: CleanupCategory[], skipModelIdentifiers?: string[]) => Promise<void>;
  reset: () => void;
};

export const useCleanupStore = create<CleanupStoreState>()((set) => ({
  preview: null,
  previewLoading: false,
  result: null,
  executing: false,

  fetchPreview: async (skipModelIdentifiers) => {
    set({ previewLoading: true });
    const { storage } = getPlatform();
    try {
      const preview = await storage.previewCleanup(skipModelIdentifiers);
      set({ preview });
    } catch {
      set({ preview: null });
    } finally {
      set({ previewLoading: false });
    }
  },

  execute: async (categories, skipModelIdentifiers) => {
    set({ executing: true });
    const { storage } = getPlatform();
    try {
      const result = await storage.executeCleanup(categories, skipModelIdentifiers);
      set({ result });
    } catch {
      set({ result: { executed: [], modelsSkipped: [] } });
    } finally {
      set({ executing: false });
    }
  },

  reset: () => set({ preview: null, previewLoading: false, result: null, executing: false }),
}));
