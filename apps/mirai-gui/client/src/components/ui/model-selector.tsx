import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from "@headlessui/react";
import { ChevronRight } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import React from "react";
import { twMerge } from "tailwind-merge";
import { useModelsStore } from "../../stores/useModelsStore";
import { modelDownloadPhases } from "@/types/modelManager";
import { ModelVendorIcon } from "@/components/models/ModelVendorIcon";

type ModelSelectorProps = {
  selectedModel?: string;
  onModelSelect?: (modelId: string, modelName: string) => void;
  menuContent: React.ReactNode;
  disabled?: boolean;
};

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelSelect,
  menuContent,
  disabled = false,
}) => {
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

  const variantClasses = disabled
    ? "text-label-muted dark:text-label-muted-dark rounded-md py-1"
    : "text-label-muted dark:text-label-muted-dark group-hover:text-label-title dark:group-hover:text-label-title-dark group-hover:bg-bg-hover dark:group-hover:bg-bg-hover-dark rounded-md py-1";

  return (
    <div className="group w-fit">
      <Listbox value={selectedModel || ""} onChange={handleModelSelect} disabled={disabled}>
        <div>
          <ListboxButton
            className={twMerge(
              "flex items-center px-[6px] transition-colors focus:outline-none focus:ring-0",
              "gap-2",
              variantClasses,
              disabled ? "opacity-60 cursor-not-allowed" : undefined,
            )}
          >
            {menuContent}
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
              anchor="bottom end"
              className="thin-scrollbar [--anchor-gap:8px] [--anchor-max-height:360px] bg-bg-modal dark:bg-bg-modal-dark border border-cell-border dark:border-cell-border-dark rounded-[6px] z-50 pointer-events-auto focus:outline-none focus:ring-0"
            >
              <div className="p-[6px] flex flex-col gap-2">
                {loadingModels ? (
                  <div className="px-[14px] py-2 leading-[130%] text-sm text-label-muted dark:text-label-muted-dark">
                    Loading models...
                  </div>
                ) : installedModelsList.length > 0 ? (
                  installedModelsList.map((model) => {
                    const isSelected = model.repoId === selectedModel;
                    return (
                      <ListboxOption
                        key={model.repoId}
                        value={model.repoId}
                        className={twMerge(
                          "w-full flex items-center gap-2 px-[14px] py-2 rounded-md text-left transition-colors cursor-pointer hover:bg-bg-hover dark:hover:bg-bg-hover-dark focus:outline-none focus:ring-0",
                          isSelected ? "bg-bg-sub dark:bg-bg-sub-dark" : "",
                        )}
                      >
                        <ModelVendorIcon vendor={model.vendor} />
                        <span className="text-sm leading-[130%] text-label-title dark:text-label-title-dark flex items-center gap-2">
                          {model.name}
                        </span>
                        <span className="text-label-muted dark:text-label-muted-dark text-[10px]">Local</span>
                      </ListboxOption>
                    );
                  })
                ) : (
                  <div className="px-[14px] py-2 leading-[130%] text-sm text-label-muted dark:text-label-muted-dark">
                    No models available
                  </div>
                )}
              </div>

              <div className="h-[1px] bg-cell-border dark:bg-cell-border-dark" />

              <div className="p-[10px]">
                <button
                  onClick={handleMoreModelsClick}
                  className="w-full flex items-center justify-between px-[14px] py-2 rounded-md hover:bg-bg-hover dark:hover:bg-bg-hover-dark transition-colors text-label-title dark:text-label-title-dark focus:outline-none focus:ring-0"
                >
                  <span className="text-sm text-label-title dark:text-label-title-dark">More local models</span>
                  <ChevronRight className="h-4 w-4 text-label-muted dark:text-label-muted-dark" />
                </button>
              </div>
            </ListboxOptions>
          </Transition>
        </div>
      </Listbox>
    </div>
  );
};
