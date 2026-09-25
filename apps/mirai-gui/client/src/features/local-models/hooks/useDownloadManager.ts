import { useToast } from "@/ui-kit";
import type { PlatformModel } from "@/types/models";
import { useCallback } from "react";
import { useModelsStore } from "@/stores/useModelsStore";
import { modelDownloadPhases } from "@/types/modelManager";
import { getPlatform } from "@/platform/platformSingleton";

const failureCopy = {
  start: (name: string) => `Failed to start download for ${name}`,
  resume: (name: string) => `Failed to resume ${name}`,
} as const;

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export const useDownloadManager = () => {
  const toast = useToast();

  const begin = useCallback(
    async (model: PlatformModel, action: keyof typeof failureCopy) => {
      const { getModelState, updateModelState } = useModelsStore.getState();
      // A retry starts from a clean row instead of showing the previous failure until the first tick.
      if (getModelState(model.repoId)?.phase === modelDownloadPhases.error) {
        updateModelState(model.repoId, {
          phase: modelDownloadPhases.downloading,
          downloadedKbytes: 0,
          totalKbytes: 0,
          error: "",
        });
      }
      const { models } = getPlatform();
      try {
        await (action === "start" ? models.startDownload(model.repoId) : models.resumeDownload(model.repoId));
      } catch (e) {
        const error = errorText(e);
        updateModelState(model.repoId, { phase: modelDownloadPhases.error, error });
        toast.error(`${failureCopy[action](model.name)}: ${error}`);
      }
    },
    [toast],
  );

  const handleDownload = useCallback((model: PlatformModel) => begin(model, "start"), [begin]);
  const handleResume = useCallback((model: PlatformModel) => begin(model, "resume"), [begin]);

  const handlePause = useCallback(
    async (model: PlatformModel) => {
      try {
        await getPlatform().models.pauseDownload(model.repoId);
      } catch (e) {
        toast.error(`Failed to pause ${model.name}: ${errorText(e)}`);
      }
    },
    [toast],
  );

  const handleCancel = useCallback(
    async (model: PlatformModel) => {
      try {
        await getPlatform().models.deleteModel(model.repoId);
      } catch (e) {
        const error = errorText(e);
        useModelsStore.getState().updateModelState(model.repoId, { phase: modelDownloadPhases.error, error });
        toast.error(`Failed to cancel ${model.name}: ${error}`);
      }
    },
    [toast],
  );

  return { handleDownload, handlePause, handleResume, handleCancel };
};
