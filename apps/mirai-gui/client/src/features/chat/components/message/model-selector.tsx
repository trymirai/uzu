import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from "@headlessui/react";
import { ChevronRight, RefreshCw } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import React from "react";
import { twMerge } from "tailwind-merge";
import { useModelsStore } from "@/stores/use-models-store";
import { modelDownloadPhases } from "@/types/model-manager";
import { ModelVendorIcon } from "@/components/model-vendor-icon";

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

  return (
    <div className="w-fit">
      <Listbox value={selectedModel || ""} onChange={handleModelSelect} disabled={disabled}>
        <div>
          <ListboxButton
            aria-label="Regenerate with model"
            title="Regenerate with model"
            className="flex size-8 items-center justify-center rounded-md text-label-muted transition-colors enabled:hover:bg-bg-hover enabled:hover:text-label-title focus:outline-hidden data-[focus]:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <RefreshCw className="size-4" aria-hidden="true" />
          </ListboxButton>

          <Transition
            enter="transition duration-100 ease-out"
            enterFrom="transform scale-95 opacity-0"
            enterTo="transform scale-100 opacity-100"
            leave="transition duration-75 ease-out"
            leaveFrom="transform scale-100 opacity-100"
            leaveTo="transform scale-95 opacity-0"
          >
            <ListboxOptions
              anchor="bottom start"
              className="overscroll-y-contain thin-scrollbar [--anchor-gap:8px] [--anchor-max-height:360px] bg-bg-modal border border-cell-border rounded-[6px] z-50 pointer-events-auto focus:outline-hidden focus:ring-0"
            >
              <div className="p-[6px] flex flex-col gap-2">
                {loadingModels ? (
                  <div className="px-[14px] py-2 leading-[130%] text-sm text-label-muted">Loading models...</div>
                ) : installedModelsList.length > 0 ? (
                  installedModelsList.map((model) => {
                    const isSelected = model.repoId === selectedModel;
                    return (
                      <ListboxOption
                        key={model.repoId}
                        value={model.repoId}
                        className={twMerge(
                          "w-full flex items-center gap-2 px-[14px] py-2 rounded-md text-left transition-colors cursor-pointer hover:bg-bg-hover focus:outline-hidden focus:ring-0",
                          isSelected ? "bg-bg-sub" : "",
                        )}
                      >
                        <ModelVendorIcon vendor={model.vendor} />
                        <span className="text-sm leading-[130%] text-label-title flex items-center gap-2">
                          {model.name}
                        </span>
                        <span className="text-label-muted text-[10px]">Local</span>
                      </ListboxOption>
                    );
                  })
                ) : (
                  <div className="px-[14px] py-2 leading-[130%] text-sm text-label-muted">No models available</div>
                )}
              </div>

              <div className="h-[1px] bg-cell-border" />

              <div className="p-[10px]">
                <button
                  onClick={handleMoreModelsClick}
                  className="w-full flex items-center justify-between px-[14px] py-2 rounded-md hover:bg-bg-hover transition-colors text-label-title focus:outline-hidden focus:ring-0"
                >
                  <span className="text-sm text-label-title">More models</span>
                  <ChevronRight className="h-4 w-4 text-label-muted" />
                </button>
              </div>
            </ListboxOptions>
          </Transition>
        </div>
      </Listbox>
    </div>
  );
};
