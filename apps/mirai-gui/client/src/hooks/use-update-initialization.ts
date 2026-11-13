import { useEffect } from "react";
import { useToast } from "@/components/ui/toast/use-toast";
import { dismissUpdateToast, showReadyUpdateToast } from "@/components/update-toast/notifications";
import { UpdateCheckOutcome, UpdatePhase, useUpdateStore } from "@/stores/use-update-store";

const UPDATE_TOAST_ID = "update-toast";
const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const UPDATE_RETRY_INTERVAL_MS = 60 * 60 * 1000;

const startUpdateChecks = (): (() => void) => {
  const { initUpdateCheck, checkForUpdate } = useUpdateStore.getState();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;

  const loop = async (check: () => Promise<UpdateCheckOutcome>) => {
    const outcome = await check();
    if (stopped) return;
    const delay = outcome === UpdateCheckOutcome.Failed ? UPDATE_RETRY_INTERVAL_MS : UPDATE_CHECK_INTERVAL_MS;
    timer = setTimeout(() => void loop(checkForUpdate), delay);
  };
  void loop(initUpdateCheck);

  return () => {
    stopped = true;
    clearTimeout(timer);
  };
};

export const useUpdateInitialization = (enabled: boolean = true) => {
  const toast = useToast();
  const status = useUpdateStore((s) => s.status);
  const { startDownload, applyUpdate, dismiss } = useUpdateStore.getState();

  useEffect(() => (enabled ? startUpdateChecks() : undefined), [enabled]);

  // Downloads start silently; the user only sees the toast once it can install.
  useEffect(() => {
    if (status.phase === UpdatePhase.Idle) {
      dismissUpdateToast(UPDATE_TOAST_ID);
      if (status.downloadError) {
        // In the app's own toast, not react-hot-toast's default one; the raw error code is for the log.
        console.warn("[update] download failed", status.downloadError);
        toast.error("Couldn't download the update", { id: "update-download-failed" });
        useUpdateStore.setState({ status: { phase: UpdatePhase.Idle } });
      }
      return;
    }

    if (status.phase === UpdatePhase.Available) {
      void startDownload(status.version);
      return;
    }

    if (status.phase === UpdatePhase.Downloaded) {
      showReadyUpdateToast({
        version: status.version,
        toastId: UPDATE_TOAST_ID,
        errorMessage: status.applyError,
        onApplyNow: () => applyUpdate(),
        onLater: () => dismiss(),
      });
    }
  }, [status, startDownload, applyUpdate, dismiss, toast]);
};
