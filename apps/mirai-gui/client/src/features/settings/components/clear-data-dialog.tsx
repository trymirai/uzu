import { useEffect, useRef, useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Modal } from "@/components/ui/modal";
import { Text } from "@/components/ui/typography";
import { formatBytes, formatModelSize } from "@/utils/format";
import { useCleanupStore } from "@/stores/use-cleanup-store";
import { useChatStore } from "@/stores/use-chat-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";

type CleanupCategory = "dialogs" | "models" | "logs";
type Step = "select" | "confirm" | "result";

const ALL_CATEGORIES: CleanupCategory[] = ["dialogs", "models", "logs"];

const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`;

const getCategoryLabel = (cat: CleanupCategory): string => {
  switch (cat) {
    case "dialogs":
      return "Dialogs";
    case "models":
      return "Downloaded models";
    case "logs":
      return "Logs";
  }
};

const getCategoryDescription = (
  cat: CleanupCategory,
  preview: ReturnType<typeof useCleanupStore.getState>["preview"],
): string => {
  if (!preview) return "";
  switch (cat) {
    case "dialogs":
      return `${plural(preview.dialogs.count, "chat", "chats")} · ${formatBytes(preview.dialogs.sizeBytes)}`;
    case "models":
      return `${plural(preview.models.count, "model", "models")} · ${formatModelSize(preview.models.sizeBytes) ?? formatBytes(preview.models.sizeBytes)}`;
    case "logs":
      return formatBytes(preview.logs.sizeBytes);
  }
};

const getConfirmSummary = (categories: CleanupCategory[]): string => {
  const labels = categories.map(getCategoryLabel);
  if (labels.length === 0) return "";
  if (labels.length === 1) return labels[0]!;
  return `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
};

type Props = {
  open: boolean;
  onClose: () => void;
};

export function ClearDataDialog({ open, onClose }: Props) {
  const [step, setStep] = useState<Step>("select");
  const [selected, setSelected] = useState<Set<CleanupCategory>>(new Set(ALL_CATEGORIES));
  const previewAppliedRef = useRef(false);

  const preview = useCleanupStore((s) => s.preview);
  const previewLoading = useCleanupStore((s) => s.previewLoading);
  const result = useCleanupStore((s) => s.result);
  const executing = useCleanupStore((s) => s.executing);
  const fetchPreview = useCleanupStore((s) => s.fetchPreview);
  const execute = useCleanupStore((s) => s.execute);
  const reset = useCleanupStore((s) => s.reset);

  useEffect(() => {
    if (!open) {
      previewAppliedRef.current = false;
      return;
    }
    setStep("select");
    reset();
    setSelected(new Set(ALL_CATEGORIES));
    const residentRepoId = useRuntimeSessionStore.getState().residentSession?.repoId;
    const residentIdentifiers = residentRepoId ? [residentRepoId] : [];
    void fetchPreview(residentIdentifiers);
  }, [open, fetchPreview, reset]);

  useEffect(() => {
    if (!preview || previewAppliedRef.current) return;
    previewAppliedRef.current = true;
    setSelected((prev) => {
      const next = new Set(prev);
      if (preview.dialogs.count === 0) next.delete("dialogs");
      if (preview.models.count === 0) next.delete("models");
      if (preview.logs.sizeBytes === 0) next.delete("logs");
      return next;
    });
  }, [preview]);

  const handleClose = () => {
    onClose();
  };

  const handleDelete = async () => {
    const residentRepoId = useRuntimeSessionStore.getState().residentSession?.repoId;
    const residentIdentifiers = residentRepoId ? [residentRepoId] : [];

    await execute([...selected], residentIdentifiers);
    const executed = useCleanupStore.getState().result?.executed ?? [];
    if (executed.includes("dialogs")) {
      await useChatStore.getState().loadSavedChats();
      useChatStore.getState().clearChat();
      useChatStore.getState().setCurrentChatId(null);
    }
    setStep("result");
  };

  const selectedCategories = ALL_CATEGORIES.filter((c) => selected.has(c));
  const loading = previewLoading || executing;

  if (step === "select") {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Clear data"
        primaryLabel="Review"
        primaryKind="danger"
        onPrimary={() => setStep("confirm")}
        primaryDisabled={selectedCategories.length === 0 || loading}
      >
        <div className="flex flex-col gap-3">
          <Text as="p" color="secondary" opticalSize={20} className="text-[13px] leading-[1.5]">
            Select the data categories you want to permanently delete.
          </Text>
          <div className="flex flex-col gap-2">
            {ALL_CATEGORIES.map((cat) => (
              <div
                key={cat}
                className="flex items-start gap-3 pb-2 cursor-pointer"
                onClick={() =>
                  setSelected((prev) => {
                    const next = new Set(prev);
                    if (next.has(cat)) next.delete(cat);
                    else next.add(cat);
                    return next;
                  })
                }
              >
                <Checkbox checked={selected.has(cat)} onChange={() => {}} className="mt-0.5 shrink-0" />
                <div className="flex flex-col gap-0.5">
                  <span className="text-[14px] font-[450] leading-[1.4] text-label-title">{getCategoryLabel(cat)}</span>
                  {preview && (
                    <span className="text-[12px] text-label-muted">{getCategoryDescription(cat, preview)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Modal>
    );
  }

  if (step === "confirm") {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Are you sure?"
        primaryLabel="Delete"
        primaryKind="danger"
        onPrimary={() => void handleDelete()}
        primaryDisabled={loading}
        secondaryLabel="Back"
        onSecondary={() => setStep("select")}
      >
        <div className="flex flex-col gap-3">
          <Text as="p" color="secondary" opticalSize={20} className="text-[14px] leading-[1.6]">
            This will permanently delete: <strong>{getConfirmSummary(selectedCategories)}</strong>.
          </Text>
          <Text as="p" color="secondary" opticalSize={20} className="text-[14px] leading-[1.6]">
            This action cannot be undone.
          </Text>
        </div>
      </Modal>
    );
  }

  const executed = result?.executed ?? [];

  return (
    <Modal open={open} onClose={handleClose} title="Done" primaryLabel="Close" onPrimary={handleClose}>
      <div className="flex flex-col gap-2">
        {ALL_CATEGORIES.filter((c) => selected.has(c)).map((cat) => {
          const done = executed.includes(cat);
          return (
            <div key={cat} className="flex items-center gap-2">
              <span className="text-[14px]">{done ? "✓" : "✗"}</span>
              <Text as="span" color={done ? "primary" : "secondary"} opticalSize={20} className="text-[14px]">
                {getCategoryLabel(cat)} {done ? "cleared" : "failed"}
              </Text>
            </div>
          );
        })}
      </div>
    </Modal>
  );
}
