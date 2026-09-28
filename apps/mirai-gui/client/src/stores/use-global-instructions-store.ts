import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";

type GlobalInstructionsState = {
  instructions: string;
  isLoading: boolean;
  loadInstructions: () => Promise<void>;
  saveInstructions: (instructions: string) => Promise<void>;
};

export const useGlobalInstructionsStore = create<GlobalInstructionsState>((set) => ({
  instructions: "",
  isLoading: false,

  loadInstructions: async () => {
    set({ isLoading: true });
    const { storage } = getPlatform();
    try {
      const instructions = await storage.loadGlobalInstructions();
      set({ instructions: instructions || "" });
    } catch (error) {
      console.error("Failed to load global instructions:", error);
    } finally {
      set({ isLoading: false });
    }
  },

  saveInstructions: async (instructions: string) => {
    set({ isLoading: true });
    const { storage } = getPlatform();
    try {
      await storage.saveGlobalInstructions(instructions);
      set({ instructions });
    } catch (error) {
      console.error("Failed to save global instructions:", error);
    } finally {
      set({ isLoading: false });
    }
  },
}));
