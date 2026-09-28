import { useEffect, useRef } from "react";
import { toast } from "react-hot-toast";
import { dismissUpdateToast, showReadyUpdateToast } from "@/components/update-toast/notifications";
import { UpdatePhase, useUpdateStore } from "@/stores/use-update-store";

const UPDATE_TOAST_ID = "update-toast";
const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

export const useUpdateInitialization = (enabled: boolean = true) => {
  const status = useUpdateStore((s) => s.status);
  const { initUpdateCheck, checkForUpdate, startDownload, applyUpdate, dismiss } = useUpdateStore.getState();
  const intervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!enabled) return;

    void initUpdateCheck();

    const schedule = () => {
      intervalRef.current = setTimeout(async () => {
        await checkForUpdate();
        schedule();
      }, UPDATE_CHECK_INTERVAL_MS);
    };
    schedule();

    return () => {
      if (intervalRef.current) {
        clearTimeout(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, initUpdateCheck, checkForUpdate]);

  // Downloads start silently; the user only sees the toast once it can install.
  useEffect(() => {
    if (status.phase === UpdatePhase.Idle) {
      dismissUpdateToast(UPDATE_TOAST_ID);
      if (status.downloadError) {
        toast.error(`Update download failed: ${status.downloadError}`);
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
  }, [status, startDownload, applyUpdate, dismiss]);
};
