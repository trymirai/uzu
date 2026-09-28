import { useInstalledPickerModels } from "../../hooks/use-picker-models";
import { useChatStore } from "@/stores/use-chat-store";
import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from "@headlessui/react";
import { Link } from "@tanstack/react-router";
import { ModelPicker } from "./chat-input/model-picker";
import { X } from "lucide-react";
import { Fragment } from "react";
import { ModelParamsControls } from "./model-params-controls";

type ModelParamsDrawerProps = {
  open: boolean;
  chatId: string;
  repoId: string | null;
  onClose: () => void;
};

export const ModelParamsDrawer = ({ open, chatId, repoId, onClose }: ModelParamsDrawerProps) => {
  const setChatModel = useChatStore((s) => s.setChatModel);
  const pickerModels = useInstalledPickerModels();

  const onPickModel = (modelId: string) => {
    const model = pickerModels.find((m) => m.id === modelId);
    if (model) setChatModel(chatId, modelId, model.name);
  };

  return (
    <Transition appear show={open} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <TransitionChild
          as={Fragment}
          enter="ease-out duration-200"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-150"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25" />
        </TransitionChild>

        <div className="fixed inset-y-0 right-0 flex max-w-full">
          <TransitionChild
            as={Fragment}
            enter="transform transition ease-out duration-300"
            enterFrom="translate-x-full"
            enterTo="translate-x-0"
            leave="transform transition ease-in duration-200"
            leaveFrom="translate-x-0"
            leaveTo="translate-x-full"
          >
            <DialogPanel className="flex h-full w-[320px] flex-col border-l border-cell-border bg-card-modal px-5 pb-6 pt-[18px] shadow-xl dark:border-cell-border-dark dark:bg-card-modal-dark">
              <div className="flex items-center justify-between gap-3">
                <DialogTitle
                  as="h3"
                  className="text-[17px] font-medium leading-[130%] text-label-title dark:text-label-title-dark"
                >
                  Edit parameters
                </DialogTitle>
                <button
                  onClick={onClose}
                  className="text-label-muted transition-colors hover:text-label-title dark:text-label-muted-dark dark:hover:text-label-title-dark"
                  aria-label="Close"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="mt-4">
                <ModelPicker
                  models={pickerModels}
                  activeModelId={repoId ?? ""}
                  onModelChange={onPickModel}
                  moreModelsLink={{ href: "/local-models", linkAs: Link }}
                  menuPlacement="down"
                />
              </div>

              <div className="mt-4 flex-1 overflow-y-auto">
                {repoId ? <ModelParamsControls repoId={repoId} /> : null}
              </div>
            </DialogPanel>
          </TransitionChild>
        </div>
      </Dialog>
    </Transition>
  );
};
