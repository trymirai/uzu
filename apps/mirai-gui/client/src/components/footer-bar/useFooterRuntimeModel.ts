import { useToast } from "@/ui-kit";
import { ejectRuntimeSessionAndWait } from "@/features/runtime/ejectRuntimeSession";
import { useModelsStore } from "@/stores/useModelsStore";
import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import { useRuntimeBusy } from "@/features/runtime/runtimeBusy";
import { runtimeSessionEjectReasons } from "@/types/session";
import {
  canFooterEjectActiveSession,
  getFooterModelLabel,
  pickFooterActiveSession,
  type FooterRuntimeStatus,
} from "./footerRuntimeModel";
import { useCallback, useMemo, useState } from "react";

type FooterRuntimeModel =
  | {
      visible: false;
      status: "hidden";
      label: "";
      vendor: "";
      canEject: false;
      onEject: () => Promise<void>;
    }
  | {
      visible: true;
      status: Exclude<FooterRuntimeStatus, "hidden">;
      label: string;
      vendor: string;
      canEject: boolean;
      onEject: () => Promise<void>;
    };

export const useFooterRuntimeModel = (): FooterRuntimeModel => {
  const toast = useToast();
  const chatModels = useModelsStore((state) => state.models);
  const residentSession = useRuntimeSessionStore((state) => state.residentSession);
  const loadingSession = useRuntimeSessionStore((state) => state.loadingSession);
  const ejectingSession = useRuntimeSessionStore((state) => state.ejectingSession);
  const runtimeBusy = useRuntimeBusy();
  const [ejectAwaiting, setEjectAwaiting] = useState(false);

  const sessions = useMemo(
    () => ({ residentSession, loadingSession, ejectingSession }),
    [residentSession, loadingSession, ejectingSession],
  );

  const { session: activeSession, status } = useMemo(() => pickFooterActiveSession(sessions), [sessions]);
  const { name: label, vendor } = useMemo(
    () => getFooterModelLabel(activeSession, chatModels),
    [activeSession, chatModels],
  );

  const onEject = useCallback(async () => {
    if (!residentSession) {
      return;
    }

    setEjectAwaiting(true);
    try {
      const ok = await ejectRuntimeSessionAndWait({
        target: residentSession,
        reason: runtimeSessionEjectReasons.user,
      });
      if (ok) {
        toast.success("Model ejected successfully");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to eject model");
    } finally {
      setEjectAwaiting(false);
    }
  }, [residentSession, toast]);

  if (!activeSession || status === "hidden") {
    return {
      visible: false,
      status: "hidden",
      label: "",
      vendor: "",
      canEject: false,
      onEject,
    };
  }

  return {
    visible: true,
    status,
    label,
    vendor,
    canEject: canFooterEjectActiveSession(sessions, activeSession, status, ejectAwaiting, runtimeBusy),
    onEject,
  };
};
