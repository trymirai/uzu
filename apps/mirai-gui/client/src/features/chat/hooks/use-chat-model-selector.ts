import { useEffect, useMemo } from "react";

import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { useModelsStore } from "@/stores/use-models-store";
import { modelDownloadPhases } from "@/types/model-manager";

type UseChatModelSelectorProps = {
  chatId: string;
  searchModel?: string;
  searchModelName?: string;
};

type UseChatModelSelectorResult = {
  selectedChatModel: {
    modelId: string;
    modelName: string;
  };
  hasAvailableModels: boolean;
  currentModelId: string | undefined;
  autoSelectSuppressed: boolean;
};

export function useChatModelSelector(props: UseChatModelSelectorProps): UseChatModelSelectorResult {
  const { chatId, searchModel, searchModelName } = props;

  const models = useModelsStore((s) => s.models);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);
  const localsLoaded = useModelsStore((s) => s.hasLoadedModels);
  const catalogComplete = useModelsStore((s) => s.catalogComplete);

  const setChatModel = useChatStore((s) => s.setChatModel);
  const selectedChatModel = useChatStore(
    (s) =>
      s.chatModels[chatId] || {
        modelId: "",
        modelName: "",
      },
  );
  const autoSelectSuppressed = useChatStore((s) => s.autoSelectSuppressed?.[chatId] === true);
  const lastUsedModel = useChatStore((s) => s.lastUsedModel);

  const residentSession = useRuntimeSessionStore((s) => s.residentSession);

  const allAvailableModels = useMemo(
    () => models.filter((model) => modelPhasesById[model.repoId] === modelDownloadPhases.downloaded),
    [models, modelPhasesById],
  );

  useEffect(() => {
    if (!localsLoaded) return;
    const selectedId = selectedChatModel.modelId;
    if (!selectedId) return;
    if (searchModel && selectedId === searchModel) return;

    const exists = allAvailableModels.some((model) => model.repoId === selectedId);
    if (exists) return;
    if (modelPhasesById[selectedId] === modelDownloadPhases.initializing || !catalogComplete) return;

    setChatModel(chatId, "", "");
  }, [
    selectedChatModel.modelId,
    searchModel,
    allAvailableModels,
    chatId,
    setChatModel,
    localsLoaded,
    catalogComplete,
    modelPhasesById,
  ]);

  useEffect(() => {
    if (autoSelectSuppressed) return;

    if (searchModel && searchModelName && !selectedChatModel.modelId) {
      setChatModel(chatId, searchModel, searchModelName);
      return;
    }

    if (selectedChatModel.modelId) return;

    if (residentSession?.repoId) {
      const residentName =
        allAvailableModels.find((model) => model.repoId === residentSession.repoId)?.name || residentSession.repoId;
      setChatModel(chatId, residentSession.repoId, residentName);
      return;
    }

    if (lastUsedModel?.modelId) {
      const isAvailable = allAvailableModels.some((model) => model.repoId === lastUsedModel.modelId);
      if (
        !localsLoaded ||
        !catalogComplete ||
        modelPhasesById[lastUsedModel.modelId] === modelDownloadPhases.initializing ||
        isAvailable
      ) {
        setChatModel(chatId, lastUsedModel.modelId, lastUsedModel.modelName);
      }
    }
  }, [
    autoSelectSuppressed,
    searchModel,
    searchModelName,
    selectedChatModel.modelId,
    chatId,
    residentSession?.repoId,
    allAvailableModels,
    setChatModel,
    lastUsedModel?.modelId,
    lastUsedModel?.modelName,
    localsLoaded,
    catalogComplete,
    modelPhasesById,
  ]);

  const currentModelId = selectedChatModel.modelId || searchModel;

  return {
    selectedChatModel,
    hasAvailableModels: allAvailableModels.length > 0,
    currentModelId,
    autoSelectSuppressed,
  };
}
