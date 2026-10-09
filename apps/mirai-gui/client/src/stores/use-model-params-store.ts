import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import { isReasoningEffort, normalizeSampling, type ModelParams, type ReasoningEffort } from "@/types/sampling";

const DEFAULT_PARAMS: ModelParams = { sampling: { type: "Default" } };

export const resolveModelTools = (
  params: ModelParams | undefined,
  globalModelChatNamingEnabled: boolean,
  paramSize?: number,
) => {
  const enabledByDefault = paramSize === undefined || paramSize >= 2_000_000_000;
  return {
    modelChatNamingEnabled: globalModelChatNamingEnabled && (params?.modelChatNamingEnabled ?? enabledByDefault),
    dateTimeToolEnabled: params?.dateTimeToolEnabled ?? enabledByDefault,
    chartToolEnabled: params?.chartToolEnabled ?? enabledByDefault,
  };
};

export const isCustomParams = (
  params: ModelParams | undefined,
  globalModelChatNamingEnabled: boolean,
  paramSize?: number,
): boolean => {
  if (!params) return false;
  const samplingChanged = params.sampling.type !== "Default";
  const tools = resolveModelTools(params, globalModelChatNamingEnabled, paramSize);
  const defaults = resolveModelTools(undefined, globalModelChatNamingEnabled, paramSize);
  return (
    samplingChanged ||
    tools.modelChatNamingEnabled !== defaults.modelChatNamingEnabled ||
    tools.dateTimeToolEnabled !== defaults.dateTimeToolEnabled ||
    tools.chartToolEnabled !== defaults.chartToolEnabled
  );
};

// Settings written before reasoning levels stored a boolean.
const migrateParams = (raw: unknown): ModelParams | null => {
  if (!raw || typeof raw !== "object") return null;
  const { sampling, reasoningEffort, reasoningEnabled, modelChatNamingEnabled, dateTimeToolEnabled, chartToolEnabled } =
    raw as {
      sampling?: ModelParams["sampling"] | { type: "Argmax" };
      reasoningEffort?: unknown;
      reasoningEnabled?: unknown;
      modelChatNamingEnabled?: unknown;
      dateTimeToolEnabled?: unknown;
      chartToolEnabled?: unknown;
    };
  if (!sampling) return null;
  const effort: ReasoningEffort | undefined = isReasoningEffort(reasoningEffort)
    ? reasoningEffort
    : typeof reasoningEnabled === "boolean"
      ? reasoningEnabled
        ? "default"
        : "disabled"
      : undefined;
  return {
    sampling: normalizeSampling(sampling),
    ...(effort !== undefined ? { reasoningEffort: effort } : {}),
    ...(typeof modelChatNamingEnabled === "boolean" ? { modelChatNamingEnabled } : {}),
    ...(typeof dateTimeToolEnabled === "boolean" ? { dateTimeToolEnabled } : {}),
    ...(typeof chartToolEnabled === "boolean" ? { chartToolEnabled } : {}),
  };
};

type ModelParamsState = {
  paramsByRepoId: Record<string, ModelParams>;
  globalModelChatNamingEnabled: boolean;
  loaded: boolean;
  load: () => Promise<void>;
  getParams: (repoId: string) => ModelParams;
  setParams: (repoId: string, params: ModelParams) => void;
  resetParams: (repoId: string) => void;
  setGlobalModelChatNamingEnabled: (enabled: boolean) => void;
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
  globalModelChatNamingEnabled: true,
  loaded: false,

  load: async () => {
    try {
      const [stored, globalModelChatNamingEnabled] = await Promise.all([
        getPlatform().settings.getModelParams(),
        getPlatform().settings.getModelChatNamingEnabled(),
      ]);
      const paramsByRepoId: Record<string, ModelParams> = {};
      for (const [repoId, raw] of Object.entries(stored ?? {})) {
        const migrated = migrateParams(raw);
        if (migrated) paramsByRepoId[repoId] = migrated;
      }
      set({
        paramsByRepoId,
        globalModelChatNamingEnabled: globalModelChatNamingEnabled !== false,
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

  setGlobalModelChatNamingEnabled: (enabled) => set({ globalModelChatNamingEnabled: enabled }),
}));
