import { useModelsStore } from "@/stores/use-models-store";
import { modelDownloadPhases } from "@/types/model-manager";
import { ModelVendorIcon } from "@/components/model-vendor-icon";
import type { ChatInputModel } from "../components/composer/chat-input/types";
import { useMemo } from "react";

export const useInstalledPickerModels = (): ChatInputModel[] => {
  const models = useModelsStore((s) => s.models);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);

  return useMemo(
    () =>
      models
        .filter((m) => modelPhasesById[m.repoId] === modelDownloadPhases.downloaded)
        .map((m) => ({
          id: m.repoId,
          name: m.name,
          logo: <ModelVendorIcon vendor={m.vendor} />,
        })),
    [models, modelPhasesById],
  );
};
