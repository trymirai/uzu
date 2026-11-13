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
import { Modal } from "@/components/ui/modal";
import { Text } from "@/components/ui/typography";
import { TooltipProvider } from "@/components/ui/tooltip/provider";
import { RefreshCcw } from "lucide-react";
import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "@/stores/use-app-store";
import { LocalModelsHero } from "./local-models-hero";
import { PageScroller } from "@/components/page-title-bar";
import { v4 as uuidv4 } from "uuid";
import { buildFamiliesView, formatParamRange, type FamilyEntry } from "../../lib/view";
import { FamilyDetailPage } from "../family-detail-page";
import { LocalModelsHeader } from "./local-models-header";
import { useModelDeletion } from "./use-model-deletion";

const TEXT_14 = "text-[14px] font-[450] leading-[1.5]";
const TEXT_14_STYLE = { fontVariationSettings: "'opsz' 20" } as const;

export function LocalModelsPage() {
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const chatWidth = useAppStore((s) => s.chatWidth);
  const createNewChat = useChatStore((s) => s.createNewChat);
  const { handleDownload, handlePause, handleResume, handleCancel } = useDownloadManager();

  const [selectedFamilyId, setSelectedFamilyId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Which way the last move went, so the new level slides in from that side. Null on first load:
  // the page just appears.
  const [direction, setDirection] = useState<"forward" | "back" | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectFamily = useCallback((id: string | null) => {
    setDirection(id ? "forward" : "back");
    setSelectedFamilyId(id);
    scrollRef.current?.scrollTo({ top: 0 });
  }, []);
  const levelAnimation =
    direction === "forward" ? "animate-uzu-forward" : direction === "back" ? "animate-uzu-back" : undefined;

  const models = useModelsStore((s) => s.models);
  const loadingModels = useModelsStore((s) => s.loading);
  const modelsInitialized = useModelsStore((s) => s.initialized);
  const catalogComplete = useModelsStore((s) => s.catalogComplete);
  const catalogRefreshing = useModelsStore((s) => s.catalogRefreshing);
  const modelsError = useModelsStore((s) => s.error);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const refreshModels = useModelsStore((s) => s.refreshModels);
  const getModelState = useModelsStore((s) => s.getModelState);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);
  const installedAtById = useModelsStore((s) => s.installedAtById);

  const { pendingDeleteModel, isDeletingModel, requestDeleteModel, confirmDeleteModel, closeDeleteModal } =
    useModelDeletion(fetchModels);

  useEffect(() => {
    void fetchModels();
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

  const families: FamilyEntry[] = useMemo(
    () => buildFamiliesView({ models, modelPhasesById, installedAtById, searchLower: "" }),
    [models, modelPhasesById, installedAtById],
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
      await refreshModels();
    } finally {
      setRefreshing(false);
    }
  }, [refreshModels]);

  const incompleteCatalog = modelsInitialized && !catalogComplete && !catalogRefreshing;

  if (models.length === 0 && (loadingModels || !modelsInitialized)) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader />
      </div>
    );
  }

  const familyList = (
    <div className="flex flex-col gap-2">
      {families.map((family) => {
        const sizeRange = formatParamRange(family.paramRange?.min, family.paramRange?.max);
        const badges: FamilyCardBadge[] = [
          ...(family.installedCount > 0
            ? [
                {
                  label: `${family.installedCount} downloaded`,
                  tone: "success" as const,
                },
              ]
            : []),
          {
            label: `${family.totalCount} ${family.totalCount === 1 ? "model" : "models"}`,
          },
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
  );

  return (
    <TooltipProvider>
      <div
        className="flex h-full min-h-0 flex-col bg-background"
        style={{ "--chat-width": `${chatWidth}px` } as CSSProperties}
      >
        <div className="w-full flex h-full min-h-0 overflow-clip">
          <div className="flex-1 min-w-0 min-h-0 flex flex-col">
            <PageScroller
              scrollRef={scrollRef}
              className="scrollbar-hide"
              header={
                <LocalModelsHeader
                  selectedFamily={selectedFamily}
                  isMobile={isMobile}
                  onBack={() => selectFamily(null)}
                  animationClassName={levelAnimation}
                />
              }
            >
              <div
                key={selectedFamilyId ?? "families"}
                className={`px-5 pt-4 pb-8 w-full max-w-[calc(var(--chat-width)+2.5rem)] mx-auto ${levelAnimation ?? ""}`}
              >
                {!selectedFamily && models.length > 0 ? <LocalModelsHero /> : null}
                {(modelsError || incompleteCatalog) && (
                  <div className="my-4">
                    <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20">
                      <div className="mt-[6px] h-2 w-2 rounded-full bg-red-500" />
                      <div className="flex-1">
                        <div className="text-sm text-red-700 dark:text-red-300">
                          {models.length > 0 ? "Some models could not be loaded" : "Could not load models"}
                        </div>
                        {modelsError && (
                          <div className="mt-1 break-all text-xs text-red-700/80 dark:text-red-300/80">
                            {modelsError}
                          </div>
                        )}
                      </div>
                      <div className="shrink-0">
                        <Button
                          icon={<RefreshCcw size={16} />}
                          kind="secondary"
                          size="sm"
                          loading={refreshing || catalogRefreshing}
                          disabled={refreshing || catalogRefreshing}
                          onClick={handleRefresh}
                        >
                          {refreshing || catalogRefreshing ? "Retrying…" : "Retry"}
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
                {models.length === 0 ? (
                  modelsError || incompleteCatalog ? null : (
                    <div className="flex min-h-[320px] items-center justify-center">
                      <span className={`${TEXT_14} text-text-muted`} style={TEXT_14_STYLE}>
                        {catalogRefreshing ? "Loading models…" : "No models available for this device"}
                      </span>
                    </div>
                  )
                ) : selectedFamily ? (
                  <FamilyDetailPage
                    family={selectedFamily}
                    searchQuery=""
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
                    <span className="text-[13px] text-text-muted">No model families</span>
                  </div>
                ) : (
                  familyList
                )}
              </div>
            </PageScroller>
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
    </TooltipProvider>
  );
}
