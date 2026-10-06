import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import { isReasoningEffort, normalizeSampling, type ModelParams, type ReasoningEffort } from "@/types/sampling";

const DEFAULT_PARAMS: ModelParams = { sampling: { type: "Default" } };

export const STOCHASTIC_SEED = { temperature: 0.7, topK: 40, topP: 0.95, minP: 0.05 } as const;

export const defaultReasoningEffort = (globalReasoningEnabled: boolean): ReasoningEffort =>
  globalReasoningEnabled ? "default" : "disabled";

export const isCustomParams = (params: ModelParams | undefined, globalReasoningEnabled: boolean): boolean => {
  if (!params) return false;
  const samplingChanged = params.sampling.type !== "Default";
  const reasoningOverridden =
    params.reasoningEffort !== undefined && params.reasoningEffort !== defaultReasoningEffort(globalReasoningEnabled);
  return samplingChanged || reasoningOverridden;
};

// Settings written before reasoning levels stored a boolean.
const migrateParams = (raw: unknown): ModelParams | null => {
  if (!raw || typeof raw !== "object") return null;
  const { sampling, reasoningEffort, reasoningEnabled } = raw as {
    sampling?: ModelParams["sampling"];
    reasoningEffort?: unknown;
    reasoningEnabled?: unknown;
  };
  if (!sampling) return null;
  const effort: ReasoningEffort | undefined = isReasoningEffort(reasoningEffort)
    ? reasoningEffort
    : typeof reasoningEnabled === "boolean"
      ? defaultReasoningEffort(reasoningEnabled)
      : undefined;
  return { sampling: normalizeSampling(sampling), ...(effort !== undefined ? { reasoningEffort: effort } : {}) };
};

type ModelParamsState = {
  paramsByRepoId: Record<string, ModelParams>;
  globalReasoningEnabled: boolean;
  loaded: boolean;
  load: () => Promise<void>;
  getParams: (repoId: string) => ModelParams;
  setParams: (repoId: string, params: ModelParams) => void;
  resetParams: (repoId: string) => void;
  setGlobalReasoningEnabled: (enabled: boolean) => void;
};

const persistTimers = new Map<string, ReturnType<typeof setTimeout>>();

const persist = (repoId: string, params: ModelParams | null): void => {
  const existing = persistTimers.get(repoId);
  if (existing) clearTimeout(existing);
  persistTimers.set(
    repoId,
    setTimeout(() => {
      persistTimers.delete(repoId);
      void getPlatform()
        .settings.setModelParams(repoId, params)
        .catch((e) => console.warn("setModelParams failed", e));
    }, 250),
  );
};

export const useModelParamsStore = create<ModelParamsState>((set, get) => ({
  paramsByRepoId: {},
  globalReasoningEnabled: true,
  loaded: false,

  load: async () => {
    try {
      const [stored, globalReasoningEnabled] = await Promise.all([
        getPlatform().settings.getModelParams(),
        getPlatform().settings.getEnableThinking(),
      ]);
      const paramsByRepoId: Record<string, ModelParams> = {};
      for (const [repoId, raw] of Object.entries(stored ?? {})) {
        const migrated = migrateParams(raw);
        if (migrated) paramsByRepoId[repoId] = migrated;
      }
      set({
        paramsByRepoId,
        globalReasoningEnabled: typeof globalReasoningEnabled === "boolean" ? globalReasoningEnabled : true,
        loaded: true,
      });
    } catch (e) {
      console.warn("useModelParamsStore.load failed", e);
      set({ loaded: true });
    }
  },

  getParams: (repoId) => get().paramsByRepoId[repoId] ?? DEFAULT_PARAMS,

  setParams: (repoId, params) => {
    set((state) => ({ paramsByRepoId: { ...state.paramsByRepoId, [repoId]: params } }));
    persist(repoId, params);
  },

  resetParams: (repoId) => {
    set((state) => {
      const next = { ...state.paramsByRepoId };
      delete next[repoId];
      return { paramsByRepoId: next };
    });
    persist(repoId, null);
  },

  setGlobalReasoningEnabled: (enabled) => set({ globalReasoningEnabled: enabled }),
}));
