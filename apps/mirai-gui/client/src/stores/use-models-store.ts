import type { DownloadEvent } from "@/platform/services/models";
import { ModelKind } from "@/types/models";
import { downloadStatePatch, preserveTotalKbytes } from "./model-download-state";
import {
  modelDownloadPhases,
  type ModelDownloadPhase,
  type ModelDownloadState,
  type VendorIcons,
} from "@/types/model-manager";
import { extractFamily } from "@/utils/extract-family";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { PlatformModel } from "@/types/models";
import { useModelParamsStore } from "./use-model-params-store";
import { getPlatform } from "@/platform/platform-singleton";

type ModelsState = {
  models: PlatformModel[];
  loading: boolean;
  /** First fetch settled, success or failure; gates initial spinners. */
  initialized: boolean;
  /** At least one fetch succeeded; gates logic that must trust `models`. */
  hasLoadedModels: boolean;
  catalogComplete: boolean;
  catalogRefreshing: boolean;
  error: string | null;
  modelStatesById: Record<string, ModelDownloadState>;
  modelPhasesById: Record<string, ModelDownloadPhase>;
  installedAtById: Record<string, number>;
  vendorIconsByName: Record<string, VendorIcons>;
  fetchModels: () => Promise<void>;
  refreshModels: () => Promise<void>;
  deleteLocalModel: (repoId: string) => Promise<void>;
  getModelState: (repoId: string) => ModelDownloadState | undefined;
  updateModelState: (repoId: string, patch: Partial<ModelDownloadState>) => void;
  applyDownloadEvent: (event: DownloadEvent) => void;
};

let fetchInFlight: Promise<void> | null = null;
let fetchAgain = false;

export const useModelsStore = create<ModelsState>()(
  persist(
    (set, get) => ({
      models: [],
      loading: false,
      initialized: false,
      hasLoadedModels: false,
      catalogComplete: false,
      catalogRefreshing: false,
      error: null,
      modelStatesById: {},
      modelPhasesById: {},
      installedAtById: {},
      vendorIconsByName: {},

      fetchModels: async () => {
        if (fetchInFlight) {
          fetchAgain = true;
          return fetchInFlight;
        }
        fetchInFlight = (async () => {
          const firstLoad = get().initialized === false;
          set(firstLoad ? { loading: true, error: null } : { error: null });
          do {
            fetchAgain = false;
            try {
              const { models: rawModels, complete, refreshing } = await getPlatform().models.getModels();

              const prevById = get().modelStatesById;
              const prevInstalledAt = get().installedAtById;

              const entries = rawModels.map((m) => {
                const repoId = m.repoId ?? m.identifier;
                const prev = prevById[repoId];
                const mergedState = prev && prev.seq > m.state.seq ? prev : preserveTotalKbytes(m.state, prev);
                return { repoId, mergedState };
              });

              const byId: Record<string, ModelDownloadState> = Object.fromEntries(
                entries.map(({ repoId, mergedState }) => [repoId, mergedState]),
              );
              const phaseById: Record<string, ModelDownloadPhase> = Object.fromEntries(
                entries.map(({ repoId, mergedState }) => [repoId, mergedState.phase]),
              );

              const now = Date.now();
              const newlyInstalledEntries = entries.flatMap(({ repoId, mergedState }) =>
                mergedState.phase === modelDownloadPhases.downloaded && prevInstalledAt[repoId] === undefined
                  ? [[repoId, now] as const]
                  : [],
              );
              const installedAt: Record<string, number> =
                newlyInstalledEntries.length > 0
                  ? { ...prevInstalledAt, ...Object.fromEntries(newlyInstalledEntries) }
                  : prevInstalledAt;

              const mapped: PlatformModel[] = rawModels.map((m, sourceIndex) => {
                const repoId = m.repoId ?? m.identifier;
                const mergedState = byId[repoId];
                const totalBytes = (mergedState?.totalKbytes ?? 0) * 1024;
                const familyName = m.familyName ?? extractFamily(m.name);
                return {
                  vendor: m.vendor,
                  name: m.name,
                  quantization: m.quantization ?? null,
                  ...(typeof m.quantizationBits === "number" ? { quantizationBits: m.quantizationBits } : {}),
                  repoId,
                  kind: ModelKind.Text,
                  reasoning: m.reasoning,
                  supportsTools: m.supportsTools,
                  size: totalBytes,
                  ...(typeof m.paramSize === "number" ? { paramSize: m.paramSize } : {}),
                  family: familyName,
                  ...(m.familyIdentifier ? { familyIdentifier: m.familyIdentifier } : {}),
                  sourceIndex,
                };
              });

              const vendorIconsByName: Record<string, VendorIcons> = Object.fromEntries(
                rawModels.flatMap((m) => (m.vendorIcons ? [[m.vendor, m.vendorIcons] as const] : [])),
              );

              set({
                models: mapped,
                modelStatesById: byId,
                modelPhasesById: phaseById,
                installedAtById: installedAt,
                vendorIconsByName,
                loading: false,
                initialized: true,
                hasLoadedModels: true,
                catalogComplete: complete,
                catalogRefreshing: refreshing,
                error: null,
              });
            } catch (error) {
              set({
                loading: false,
                initialized: true,
                error: error instanceof Error ? error.message : "Failed to fetch models",
              });
            }
          } while (fetchAgain);
        })().finally(() => {
          fetchInFlight = null;
        });
        return fetchInFlight;
      },

      refreshModels: async () => {
        try {
          await getPlatform().models.refreshModels();
          await get().fetchModels();
        } catch (error) {
          set({ error: error instanceof Error ? error.message : "Failed to refresh models" });
        }
      },

      deleteLocalModel: async (repoId: string) => {
        await getPlatform().models.deleteModel(repoId);
        useModelParamsStore.getState().resetParams(repoId);
      },

      getModelState: (repoId: string) => get().modelStatesById[repoId],

      applyDownloadEvent: (event) => {
        const current = get().modelStatesById[event.identifier];
        if (current && event.seq <= current.seq) return;
        // seq advances even for an event without a state change, so a snapshot
        // taken before it cannot come back on top later.
        const patch = downloadStatePatch(current, event) ?? {};
        get().updateModelState(event.identifier, { ...patch, seq: event.seq });
      },

      updateModelState: (repoId: string, patch: Partial<ModelDownloadState>) => {
        const prev = get().modelStatesById;
        const next: ModelDownloadState = {
          totalKbytes: 0,
          downloadedKbytes: 0,
          phase: modelDownloadPhases.notDownloaded,
          seq: 0,
          ...prev[repoId],
          ...patch,
        };
        const prevPhase = prev[repoId]?.phase;
        const nextPhase = next.phase;
        const prevInstalledAt = get().installedAtById;
        const isFreshInstall =
          prevPhase !== modelDownloadPhases.downloaded &&
          nextPhase === modelDownloadPhases.downloaded &&
          prevInstalledAt[repoId] === undefined;
        const justRemoved = nextPhase === modelDownloadPhases.notDownloaded && repoId in prevInstalledAt;
        const installedAtPatch = isFreshInstall
          ? { ...prevInstalledAt, [repoId]: Date.now() }
          : justRemoved
            ? Object.fromEntries(Object.entries(prevInstalledAt).filter(([id]) => id !== repoId))
            : undefined;
        set({
          modelStatesById: { ...prev, [repoId]: next },
          ...(prevPhase !== nextPhase
            ? {
                modelPhasesById: {
                  ...get().modelPhasesById,
                  [repoId]: nextPhase,
                },
              }
            : {}),
          ...(installedAtPatch ? { installedAtById: installedAtPatch } : {}),
        });
      },
    }),
    {
      name: "mirai-models-store",
      partialize: (state) => ({ installedAtById: state.installedAtById }),
    },
  ),
);
