import { useEffect, useRef } from "react";
import { useToast } from "@/components/ui/toast/use-toast";
import { useChatStore } from "@/stores/use-chat-store";

// A reply can finish and fail to save while another page is open, so the toast lives at the root.
export const useSaveFailureToast = (): void => {
  const toast = useToast();
  const saveFailureCount = useChatStore((s) => s.saveFailureCount);
  const seenRef = useRef(saveFailureCount);
  useEffect(() => {
    if (saveFailureCount === seenRef.current) return;
    seenRef.current = saveFailureCount;
    toast.error("Failed to save the message. It may be missing after a restart.", { id: "chat-save-failed" });
  }, [saveFailureCount, toast]);
};
