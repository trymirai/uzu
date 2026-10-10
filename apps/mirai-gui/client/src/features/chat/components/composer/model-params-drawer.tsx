import { useInstalledPickerModels } from "../../hooks/use-picker-models";
import { useChatStore } from "@/stores/use-chat-store";
import { Button, Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from "@headlessui/react";
import { ModelPicker } from "./chat-input/model-picker";
import { X } from "lucide-react";
import { Fragment, useEffect, useState } from "react";
import { ModelParamsControls } from "./model-params-controls";
import { getPlatform } from "@/platform/platform-singleton";
import type { SamplingDefaults } from "@/platform/services/chat";

type ModelParamsDrawerProps = {
  open: boolean;
  chatId: string;
  repoId: string | null;
  onClose: () => void;
};

export const ModelParamsDrawer = ({ open, chatId, repoId, onClose }: ModelParamsDrawerProps) => {
  const setChatModel = useChatStore((s) => s.setChatModel);
  const pickerModels = useInstalledPickerModels();
  const [defaults, setDefaults] = useState<{ repoId: string; sampling: SamplingDefaults | null } | null>(null);
  const samplingReady = defaults?.repoId === repoId && defaults.sampling !== null;

  // This component stays mounted while the drawer is closed. Read defaults on
  // model selection so its controls are ready before the opening transition.
  useEffect(() => {
    if (!repoId || samplingReady) return;
    let alive = true;
    void getPlatform()
      .chat.getSamplingDefaults(repoId)
      .then((sampling) => {
        if (alive) setDefaults({ repoId, sampling });
      })
      .catch(() => {
        if (alive) setDefaults({ repoId, sampling: null });
      });
    return () => {
      alive = false;
    };
  }, [repoId, open, samplingReady]);

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
          <div className="fixed inset-0 bg-black/25" />
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
            <DialogPanel className="flex h-full w-[320px] flex-col border-l border-cell-border bg-bg-modal px-5 pb-6 pt-[18px] shadow-xl">
              <div className="flex items-center justify-between gap-3">
                <DialogTitle as="h3" className="text-[17px] font-medium leading-[130%] text-label-title">
                  Edit parameters
                </DialogTitle>
                <Button
                  onClick={onClose}
                  className="text-label-muted transition-colors hover:text-label-title outline-hidden data-[focus]:shadow-focus"
                  aria-label="Close"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              <div className="mt-4">
                <ModelPicker
                  models={pickerModels}
                  activeModelId={repoId ?? ""}
                  onModelChange={onPickModel}
                  moreModelsLink={{ href: "/local-models" }}
                  menuPlacement="down"
                />
              </div>

              <div className="mt-4 flex-1 min-h-0 overflow-y-auto overscroll-y-contain">
                {repoId &&
                  (defaults?.repoId === repoId ? (
                    <ModelParamsControls
                      key={`${repoId}:${samplingReady}`}
                      repoId={repoId}
                      samplingDefaults={defaults.sampling}
                    />
                  ) : (
                    <p className="text-[12px] text-label-muted">Loading model sampling settings…</p>
                  ))}
              </div>
            </DialogPanel>
          </TransitionChild>
        </div>
      </Dialog>
    </Transition>
  );
};
