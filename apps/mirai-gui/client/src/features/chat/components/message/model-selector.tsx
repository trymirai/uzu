import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from "@headlessui/react";
import { Check, ChevronRight, RefreshCw } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import React from "react";
import { twMerge } from "tailwind-merge";
import { ModelName } from "@/components/model-name";
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
            className="flex size-8 items-center justify-center rounded-lg text-label-muted transition-[background-color,color,scale] duration-150 ease-out enabled:hover:bg-black/[0.06] enabled:hover:text-label-title dark:enabled:hover:bg-white/[0.08] enabled:active:scale-[0.96] focus:outline-hidden data-[focus]:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <RefreshCw className="size-4" aria-hidden="true" />
          </ListboxButton>

          <Transition
            enter="transition-[opacity,scale] duration-200 ease-spring motion-reduce:transition-none"
            enterFrom="scale-95 opacity-0"
            enterTo="scale-100 opacity-100"
            leave="transition-[opacity,scale] duration-[140ms] ease-spring motion-reduce:transition-none"
            leaveFrom="scale-100 opacity-100"
            leaveTo="scale-95 opacity-0"
          >
            <ListboxOptions
              anchor="bottom start"
              className="min-w-[240px] origin-top-left overscroll-y-contain thin-scrollbar [--anchor-gap:8px] [--anchor-max-height:360px] rounded-xl bg-surface-popover shadow-popover p-1 z-50 pointer-events-auto focus:outline-hidden focus:ring-0"
            >
              <div className="flex flex-col gap-px">
                {loadingModels ? (
                  <div className="flex min-h-8 items-center px-2 text-[13px] text-label-muted">Loading models…</div>
                ) : installedModelsList.length > 0 ? (
                  installedModelsList.map((model) => (
                    <ListboxOption
                      key={model.repoId}
                      value={model.repoId}
                      className="group flex min-h-8 w-full cursor-pointer items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors duration-100 ease-out data-[focus]:bg-surface-tertiary focus:outline-hidden"
                    >
                      <ModelVendorIcon vendor={model.vendor} />
                      <ModelName
                        name={model.name}
                        className="flex-1 text-[13px] font-[450] leading-[1.3] text-label-title"
                      />
                      <Check
                        aria-hidden
                        className={twMerge(
                          "size-4 shrink-0 text-label-title",
                          model.repoId === selectedModel ? "opacity-100" : "opacity-0",
                        )}
                      />
                    </ListboxOption>
                  ))
                ) : (
                  <div className="flex min-h-8 items-center px-2 text-[13px] text-label-muted">No models available</div>
                )}
              </div>

              <div className="mx-2 my-1 h-px bg-border-default" />

              <button
                onClick={handleMoreModelsClick}
                className="flex min-h-8 w-full items-center justify-between gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors duration-100 ease-out hover:bg-surface-tertiary focus-visible:bg-surface-tertiary focus:outline-hidden"
              >
                <span className="text-[13px] font-[450] leading-[1.3] text-label-title">More models</span>
                <ChevronRight className="size-4 text-label-muted" aria-hidden />
              </button>
            </ListboxOptions>
          </Transition>
        </div>
      </Listbox>
    </div>
  );
};
