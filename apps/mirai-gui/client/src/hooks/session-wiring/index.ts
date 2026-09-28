import { useEffect } from "react";
import { bindChatLoading, bindChatSessionState } from "./adapters";

export const useSessionWiring = (enabled: boolean = true) => {
  useEffect(() => {
    if (!enabled) return;
    const unbindState = bindChatSessionState();
    const unbindLoading = bindChatLoading();
    return () => {
      unbindState();
      unbindLoading();
    };
  }, [enabled]);
};
