import { useModelsStore } from "@/stores/useModelsStore";
import { modelDownloadPhases } from "@/types/modelManager";
import { ModelVendorIcon } from "@/components/models/ModelVendorIcon";
import type { ChatInputModel } from "@/ui-kit";
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
