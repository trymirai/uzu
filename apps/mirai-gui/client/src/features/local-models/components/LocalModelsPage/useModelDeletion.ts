import { useToast } from "@/ui-kit";
import { ejectRuntimeSessionAndWait } from "@/features/runtime/ejectRuntimeSession";
import { useModelsStore } from "@/stores/useModelsStore";
import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import type { PlatformModel } from "@/types/models";
import { useCallback, useState, type MouseEvent } from "react";

type ModelDeletion = {
  pendingDeleteModel: PlatformModel | null;
  isDeletingModel: boolean;
  requestDeleteModel: (model: PlatformModel, event: MouseEvent<HTMLElement>) => void;
  confirmDeleteModel: () => Promise<void>;
  closeDeleteModal: () => void;
};

export function useModelDeletion(fetchModels: () => Promise<void>): ModelDeletion {
  const toast = useToast();
  const [pendingDeleteModel, setPendingDeleteModel] = useState<PlatformModel | null>(null);
  const [isDeletingModel, setIsDeletingModel] = useState(false);

  const deleteModel = useCallback(
    async (model: PlatformModel) => {
      try {
        const residentSession = useRuntimeSessionStore.getState().residentSession;
        if (residentSession?.repoId === model.repoId) {
          await ejectRuntimeSessionAndWait({ target: residentSession });
        }

        await useModelsStore.getState().deleteLocalModel(model.repoId);
        toast.success(`Uninstalled ${model.name}`);
      } catch (error) {
        toast.error(`Failed to uninstall ${model.name}`);
        console.error(error);
      } finally {
        await fetchModels();
      }
    },
    [fetchModels, toast],
  );

  const requestDeleteModel = useCallback(
    (model: PlatformModel, event: MouseEvent<HTMLElement>) => {
      if (event.shiftKey) {
        void deleteModel(model);
        return;
      }
      setPendingDeleteModel(model);
    },
    [deleteModel],
  );

  const confirmDeleteModel = useCallback(async () => {
    if (!pendingDeleteModel || isDeletingModel) return;
    setIsDeletingModel(true);
    try {
      await deleteModel(pendingDeleteModel);
    } finally {
      setIsDeletingModel(false);
      setPendingDeleteModel(null);
    }
  }, [deleteModel, pendingDeleteModel, isDeletingModel]);

  const closeDeleteModal = useCallback(() => {
    setPendingDeleteModel(null);
  }, []);

  return { pendingDeleteModel, isDeletingModel, requestDeleteModel, confirmDeleteModel, closeDeleteModal };
}
