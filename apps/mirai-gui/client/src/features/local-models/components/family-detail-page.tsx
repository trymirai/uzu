import { useIsMobile } from "@/hooks/use-media-query";
import { useModelsStore } from "@/stores/use-models-store";
import type { PlatformModel } from "@/types/models";
import { formatModelSize } from "@/utils/format";
import { ModelVendorIcon } from "@/components/model-vendor-icon";
import { ModelCard } from "./model-card";
import { ModelTable } from "./model-table";
import { useShiftHeld } from "@/hooks/use-shift-held";
import { useMemo, type MouseEvent } from "react";
import { buildFamilyDetailView, formatQuantization, toModelCardState, type FamilyEntry } from "../lib/view";

export type FamilyDetailPageProps = {
  family: FamilyEntry;
  searchQuery: string;
  sortModels: (list: PlatformModel[]) => PlatformModel[];
  onDownload: (model: PlatformModel) => Promise<void> | void;
  onPause: (model: PlatformModel) => void;
  onResume: (model: PlatformModel) => void;
  onCancel: (model: PlatformModel) => Promise<void> | void;
  onDelete: (model: PlatformModel, event: MouseEvent<HTMLElement>) => void;
  onOpen: (model: PlatformModel) => void;
};

export function FamilyDetailPage({
  family,
  searchQuery,
  sortModels,
  onDownload,
  onPause,
  onResume,
  onCancel,
  onDelete,
  onOpen,
}: FamilyDetailPageProps) {
  const isMobile = useIsMobile();
  const shiftHeld = useShiftHeld();
  const models = useModelsStore((s) => s.models);
  const modelStatesById = useModelsStore((s) => s.modelStatesById);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);

  const detail = useMemo(
    () =>
      buildFamilyDetailView({
        models,
        modelPhasesById,
        familyIdentifier: family.familyIdentifier,
        searchLower: searchQuery.trim().toLowerCase(),
        sortModels,
      }),
    [models, modelPhasesById, family.familyIdentifier, searchQuery, sortModels],
  );

  const renderRow = (model: PlatformModel) => {
    const state = modelStatesById[model.repoId];
    const cardState = toModelCardState(state?.phase, state?.downloadedKbytes, state?.totalKbytes);
    const isPaused = cardState.status === "paused";

    const card = (
      <ModelCard
        compact={isMobile}
        name={model.name}
        logo={<ModelVendorIcon vendor={model.vendor} size={16} className="h-4 w-4" />}
        size={formatModelSize(model.size)}
        parameters={formatQuantization(model)}
        state={cardState}
        onDownload={() => void onDownload(model)}
        onPause={() => (isPaused ? onResume(model) : onPause(model))}
        onCancel={() => void onCancel(model)}
        onRetry={() => void onDownload(model)}
        onDelete={(event) => onDelete(model, event)}
        quickDelete={shiftHeld}
        onOpen={() => onOpen(model)}
      />
    );

    return <div key={model.repoId}>{card}</div>;
  };

  if (detail.installed.length === 0 && detail.available.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center">
        <span className="text-[13px] text-text-muted">
          {searchQuery ? `No models matching "${searchQuery}"` : "No models in this family"}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {detail.installed.length > 0 && (
        <ModelTable title="Installed" compact={isMobile}>
          {detail.installed.map((m) => renderRow(m))}
        </ModelTable>
      )}
      {detail.available.length > 0 && (
        <ModelTable title="Available" compact={isMobile}>
          {detail.available.map((m) => renderRow(m))}
        </ModelTable>
      )}
    </div>
  );
}
