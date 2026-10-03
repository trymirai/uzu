import { Loader } from "@/components/loader";
import { useIsMobile } from "@/hooks/use-media-query";
import { useChatStore } from "@/stores/use-chat-store";
import { useDownloadManager } from "../../hooks/use-download-manager";
import { useModelsStore } from "@/stores/use-models-store";
import type { PlatformModel } from "@/types/models";
import { ModelVendorIcon } from "@/components/model-vendor-icon";
import { createNewestTextModelComparator, createSizeAccessor } from "../../lib/model-sort";
import { useNavigate } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { FamilyCard, type FamilyCardBadge } from "../family-card";
import { HoverPopoverProvider } from "@/components/ui/popover/hover-popover/provider";
import { Modal } from "@/components/ui/modal";
import { Text } from "@/components/ui/typography";
import { TooltipProvider } from "@/components/ui/tooltip/provider";
import { RefreshCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { buildFamiliesView, formatParamRange, isPartnerFamily, type FamilyEntry } from "../../lib/view";
import { FamilyDetailPage } from "../family-detail-page";
import { LocalModelsHeader } from "./local-models-header";
import { useModelDeletion } from "./use-model-deletion";

const TEXT_14 = "text-[14px] font-[450] leading-[1.5]";
const TEXT_14_STYLE = { fontVariationSettings: "'opsz' 20" } as const;

export function LocalModelsPage() {
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const createNewChat = useChatStore((s) => s.createNewChat);
  const { handleDownload, handlePause, handleResume, handleCancel } = useDownloadManager();

  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [modelsSearch, setModelsSearch] = useState("");
  const [selectedFamilyId, setSelectedFamilyId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const selectFamily = useCallback((id: string | null) => {
    setSelectedFamilyId(id);
    setModelsSearch("");
  }, []);

  const models = useModelsStore((s) => s.models);
  const loadingModels = useModelsStore((s) => s.loading);
  const modelsInitialized = useModelsStore((s) => s.initialized);
  const modelsError = useModelsStore((s) => s.error);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const getModelState = useModelsStore((s) => s.getModelState);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);
  const installedAtById = useModelsStore((s) => s.installedAtById);

  const { pendingDeleteModel, isDeletingModel, requestDeleteModel, confirmDeleteModel, closeDeleteModal } =
    useModelDeletion(fetchModels);

  useEffect(() => {
    const loadPage = async () => {
      try {
        await fetchModels();
      } finally {
        setLoading(false);
      }
    };

    void loadPage();
  }, [fetchModels]);

  const getSize = useMemo(() => createSizeAccessor(getModelState), [getModelState]);
  const newestTextModelComparator = useMemo(() => createNewestTextModelComparator(), []);

  const sortModels = useCallback(
    (list: PlatformModel[]) => {
      const next = [...list];
      next.sort((a, b) => {
        const sizeDiff = getSize(a) - getSize(b);
        if (sizeDiff !== 0) return sizeDiff;
        return newestTextModelComparator(a, b);
      });
      return next;
    },
    [getSize, newestTextModelComparator],
  );

  const searchLower = searchQuery.trim().toLowerCase();

  const families: FamilyEntry[] = useMemo(
    () => buildFamiliesView({ models, modelPhasesById, installedAtById, searchLower }),
    [models, modelPhasesById, installedAtById, searchLower],
  );

  const selectedFamily = useMemo(
    () => (selectedFamilyId ? families.find((f) => f.familyIdentifier === selectedFamilyId) : undefined),
    [families, selectedFamilyId],
  );

  const handleInstalledModelClick = useCallback(
    (model: PlatformModel) => {
      const newChatId = uuidv4();
      createNewChat(newChatId);
      navigate({
        to: "/chat/$chatId",
        params: { chatId: newChatId },
        search: {
          model: model.repoId,
          modelName: model.name,
          isNew: true,
        },
      });
    },
    [createNewChat, navigate],
  );

  const handleRefresh = useCallback(async () => {
    try {
      setRefreshing(true);
      await fetchModels();
    } finally {
      setRefreshing(false);
    }
  }, [fetchModels]);

  if (loading || loadingModels || !modelsInitialized) {
    return (
      <div className="flex h-[calc(100vh-24px)] items-center justify-center">
        <Loader />
      </div>
    );
  }

  return (
    <TooltipProvider>
      <HoverPopoverProvider>
        <div className="flex h-[calc(100vh-24px)] flex-col bg-background">
          <div className="w-full flex h-full overflow-hidden bg-surface-elevated">
            <div className="flex-1 min-w-0 flex flex-col">
              <LocalModelsHeader
                selectedFamily={selectedFamily}
                isMobile={isMobile}
                searchQuery={searchQuery}
                modelsSearch={modelsSearch}
                onSearchQueryChange={setSearchQuery}
                onModelsSearchChange={setModelsSearch}
                onBack={() => selectFamily(null)}
              />
              <div style={{ borderBottom: "0.5px solid var(--color-border-default)" }}></div>

              <div className="flex-1 overflow-y-auto scrollbar-hide">
                <div className="px-4 py-4">
                  {modelsError ? (
                    <div className="my-4">
                      <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20">
                        <div className="mt-[6px] h-2 w-2 rounded-full bg-red-500" />
                        <div className="flex-1">
                          <div className="text-sm text-red-700 dark:text-red-300">Failed to load models</div>
                          <div className="mt-1 break-all text-xs text-red-700/80 dark:text-red-300/80">
                            {modelsError}
                          </div>
                        </div>
                        <div className="shrink-0">
                          <Button
                            icon={<RefreshCcw size={16} />}
                            kind="secondary"
                            size="sm"
                            loading={refreshing}
                            disabled={refreshing}
                            onClick={handleRefresh}
                          >
                            {refreshing ? "Retrying…" : "Retry"}
                          </Button>
                        </div>
                      </div>
                    </div>
                  ) : models.length === 0 ? (
                    <div className="flex min-h-[320px] items-center justify-center">
                      <span className={`${TEXT_14} text-text-muted`} style={TEXT_14_STYLE}>
                        No models available for this device
                      </span>
                    </div>
                  ) : selectedFamily ? (
                    <FamilyDetailPage
                      family={selectedFamily}
                      searchQuery={modelsSearch}
                      sortModels={sortModels}
                      onDownload={handleDownload}
                      onPause={handlePause}
                      onResume={handleResume}
                      onCancel={handleCancel}
                      onDelete={requestDeleteModel}
                      onOpen={handleInstalledModelClick}
                    />
                  ) : families.length === 0 ? (
                    <div className="flex min-h-[200px] items-center justify-center">
                      <span className="text-[13px] text-text-muted">
                        {searchQuery ? `No families matching "${searchQuery}"` : "No model families"}
                      </span>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-1.5">
                      {families.map((family) => {
                        const sizeRange = formatParamRange(family.paramRange?.min, family.paramRange?.max);
                        const badges: FamilyCardBadge[] = [
                          ...(isPartnerFamily(family.familyName)
                            ? [
                                {
                                  label: "Partner",
                                  tooltip: "Partner of Mirai — co-developed and verified to work on your hardware",
                                  tone: "success" as const,
                                  variant: "filled" as const,
                                },
                              ]
                            : []),
                          {
                            label: `${family.totalCount} ${family.totalCount === 1 ? "model" : "models"}`,
                          },
                          ...(family.installedCount > 0
                            ? [
                                {
                                  label: `${family.installedCount} installed ${family.installedCount === 1 ? "model" : "models"}`,
                                  tone: "success" as const,
                                },
                              ]
                            : []),
                          ...(sizeRange ? [{ label: sizeRange }] : []),
                        ];
                        return (
                          <FamilyCard
                            key={family.familyIdentifier}
                            compact={isMobile}
                            vendorIcon={<ModelVendorIcon vendor={family.vendor} size={16} className="h-4 w-4" />}
                            familyName={family.familyName}
                            vendorName={family.vendor}
                            badges={badges}
                            onClick={() => selectFamily(family.familyIdentifier)}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          <Modal
            open={pendingDeleteModel != null}
            onClose={closeDeleteModal}
            title="Delete model"
            primaryLabel="Delete"
            primaryKind="danger"
            primaryDisabled={isDeletingModel}
            onPrimary={() => {
              void confirmDeleteModel();
            }}
          >
            <Text as="p" color="secondary" opticalSize={20} className="text-[15px] font-[450] leading-[1.6]">
              Are you sure you want to delete model "{pendingDeleteModel?.repoId ?? ""}"? This action cannot be undone.
            </Text>
          </Modal>
        </div>
      </HoverPopoverProvider>
    </TooltipProvider>
  );
}
