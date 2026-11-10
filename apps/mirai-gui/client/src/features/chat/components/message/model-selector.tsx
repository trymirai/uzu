import { Check, ChevronRight } from "lucide-react";
import { IconArrowRotateClockwise } from "@/components/icons/app-icons";
import { useNavigate } from "@tanstack/react-router";
import React from "react";
import { HoverPopover } from "@/components/ui/popover/hover-popover";
import type { PopoverItem } from "@/components/ui/popover/shared/types";
import { twMerge } from "tailwind-merge";
import { ModelName } from "@/components/model-name";
import { useModelsStore } from "@/stores/use-models-store";
import { modelDownloadPhases } from "@/types/model-manager";

const ACTION_CLASSNAME =
  "flex size-8 items-center justify-center rounded-lg text-label-muted transition-colors duration-150 ease-out enabled:hover:bg-bg-hover enabled:hover:text-label-title outline-hidden focus-visible:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed";

type ModelSelectorProps = {
  selectedModel?: string;
  onModelSelect?: (modelId: string, modelName: string) => void;
  disabled?: boolean;
};

export const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedModel, onModelSelect, disabled = false }) => {
  const navigate = useNavigate();

  const models = useModelsStore((s) => s.models);
  const getModelState = useModelsStore((s) => s.getModelState);
  const loadingModels = useModelsStore((s) => s.loading);

  const installedModelsList = models.filter((m) => getModelState(m.repoId)?.phase === modelDownloadPhases.downloaded);

  const handleModelSelect = (modelId: string) => {
    const selectedModel = installedModelsList.find((model) => model.repoId === modelId);
    if (selectedModel) {
      onModelSelect?.(selectedModel.repoId, selectedModel.name);
    }
  };

  const handleMoreModelsClick = () => {
    navigate({ to: "/local-models" });
  };

  // Opens on hover, like a tooltip, with the models to regenerate with: names only, three at a time
  // with the rest scrolling, and "Download models" fixed under them. Above the button: the answer's
  // actions sit near the composer and the window's bottom, where a long list would run off. Keyboard: Enter or ↓ opens
  // it, Esc closes it (handled by HoverPopover).
  const items: PopoverItem[] = loadingModels
    ? [{ id: "loading", label: "Loading models…", disabled: true }]
    : installedModelsList.length > 0
      ? installedModelsList.map((model) => ({
          id: model.repoId,
          label: (
            <span className="flex min-w-0 flex-1 items-center gap-2.5">
              <ModelName name={model.name} className="flex-1 text-[13px] font-[450] leading-[1.3] text-label-title" />
              <Check
                aria-hidden
                className={twMerge(
                  "size-4 shrink-0 text-label-title",
                  model.repoId === selectedModel ? "opacity-100" : "opacity-0",
                )}
              />
            </span>
          ),
          onClick: () => handleModelSelect(model.repoId),
        }))
      : [{ id: "none", label: "No models available", disabled: true }];
  const footerItems: PopoverItem[] = [
    {
      id: "more",
      label: (
        <span className="flex flex-1 items-center justify-between gap-2.5 text-[13px] font-[450] leading-[1.3] text-label-title">
          Download models
          <ChevronRight className="size-4 text-label-muted" aria-hidden />
        </span>
      ),
      onClick: handleMoreModelsClick,
    },
  ];

  // Clicking Regenerate itself reruns the answer with the model that has the check (the one that
  // wrote it); the menu that opens on hover picks a different one.
  const handleRegenerateClick = () => {
    const current = installedModelsList.find((model) => model.repoId === selectedModel);
    if (current) onModelSelect?.(current.repoId, current.name);
  };

  const button = (
    <button
      type="button"
      aria-label="Regenerate"
      disabled={disabled}
      onClick={handleRegenerateClick}
      className={ACTION_CLASSNAME}
    >
      <IconArrowRotateClockwise size={16} aria-hidden />
    </button>
  );

  return disabled ? (
    button
  ) : (
    <HoverPopover
      trigger={button}
      items={items}
      footerItems={footerItems}
      maxVisibleItems={3}
      side="top"
      align="start"
    />
  );
};
