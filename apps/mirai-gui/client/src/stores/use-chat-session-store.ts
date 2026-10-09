import { useRuntimeSessionStore } from "./use-runtime-session-store";
import type { RuntimeSessionRef, RuntimeSessionEjectReason } from "@/types/session";
import type { OutputShape } from "@/types/llm-stream";
import { v4 as uuidv4 } from "uuid";
import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";

type SessionKey = RuntimeSessionRef;

type ChatMessageRef = { chatId: string; messageId: string };

type OperationState = "idle" | "running" | "stopping" | "ejecting";

// Stopping may overlap a running effect. Keep cancellation owned by the
// effect, independently of the operation currently displayed by the store.
const chatOperations = new Map<string, { chatId: string; controller: AbortController }>();

type SessionStoreState = {
  isGenerating: boolean;
  isEjecting: boolean;
  isModelLoading: boolean;
  operationState: OperationState;
  operationId: string | null;
  lastEjected: SessionKey | null;
  lastEjectedReason?: RuntimeSessionEjectReason | null;
  isTitleGenerating: boolean;
  titleGenChatId: string | null;
  // Set by cancelActiveRunForChat while that chat's title is generating; the
  // send flow that owns the run consumes it and settles the placeholder.
  titleGenAbortChatId: string | null;
  activeGeneratingChatId: string | null;
  activeAssistantMessageId: string | null;
  activeAssistantMessageText: string | null;
  activeAssistantMessageOutput: OutputShape | null;
  // The assistant placeholder from its creation until the reply settles; it
  // exists before the run does (title generation runs in between).
  loadingMessage: ChatMessageRef | null;
  canceledMessage: ChatMessageRef | null;

  setLoadingMessage: (chatId: string, messageId: string | null) => void;
  setCanceledMessage: (chatId: string, messageId: string | null) => void;
  setGenerating: (running: boolean) => void;
  setEjecting: (ejecting: boolean) => void;
  startModelLoading: () => void;
  endModelLoading: () => void;
  setTitleGenerating: (v: boolean) => void;
  setTitleGenChat: (chatId: string | null) => void;
  consumeTitleGenAbort: (chatId: string) => boolean;
  setLastEjected: (key: SessionKey | null) => void;
  setLastEjectedWithReason: (key: SessionKey | null, reason: RuntimeSessionEjectReason | null) => void;

  setActiveGenerating: (chatId: string, assistantMessageId: string) => void;
  clearActiveGenerating: () => void;
  setActiveAssistantMessageText: (text: string | null) => void;
  setActiveAssistantMessageOutput: (output: OutputShape | null) => void;
  cancelTitleGen: () => Promise<boolean>;
  activeRunId: string | null;
  setActiveRunId: (runId: string | null) => void;
  cancelActiveRunForChat: (chatId: string) => Promise<void>;

  canRun: (key: SessionKey) => boolean;
  canStop: () => boolean;
  canEject: () => boolean;

  withOperation: <T>(
    op: Exclude<OperationState, "idle">,
    effect: (signal: AbortSignal) => Promise<T>,
    chatId?: string,
  ) => Promise<T | null>;
};

export const useChatSessionStore = create<SessionStoreState>((set, get) => ({
  isGenerating: false,
  isEjecting: false,
  isModelLoading: false,
  operationState: "idle",
  operationId: null,
  lastEjected: null,
  lastEjectedReason: null,

  isTitleGenerating: false,
  titleGenChatId: null,
  titleGenAbortChatId: null,
  activeGeneratingChatId: null,
  activeAssistantMessageId: null,
  activeAssistantMessageText: null,
  activeAssistantMessageOutput: null,
  loadingMessage: null,
  canceledMessage: null,

  setLoadingMessage: (chatId, messageId) =>
    set((s) => {
      if (messageId) return { loadingMessage: { chatId, messageId } };
      return s.loadingMessage?.chatId === chatId ? { loadingMessage: null } : {};
    }),
  setCanceledMessage: (chatId, messageId) =>
    set((s) => {
      if (messageId) return { canceledMessage: { chatId, messageId } };
      return s.canceledMessage?.chatId === chatId ? { canceledMessage: null } : {};
    }),
  setGenerating: (running) => set({ isGenerating: running }),
  // The backend can eject on its own timer while a run is being prepared.
  setEjecting: (ejecting) => {
    const st = get().operationState;
    const next = ejecting ? (st === "idle" ? "ejecting" : st) : st === "ejecting" ? "idle" : st;
    set({ isEjecting: ejecting, operationState: next });
  },
  // A model load only ever happens inside a run.
  startModelLoading: () => set({ isModelLoading: true }),
  // Loads never overlap, so the terminal event is not matched to a load;
  // ignoring a mismatched one would strand isModelLoading and block send,
  // stop and eject.
  endModelLoading: () => set({ isModelLoading: false }),
  setTitleGenerating: (v) => set({ isTitleGenerating: v }),
  setTitleGenChat: (chatId) => set({ titleGenChatId: chatId }),
  consumeTitleGenAbort: (chatId) => {
    const requested = get().titleGenAbortChatId === chatId;
    if (requested) set({ titleGenAbortChatId: null });
    return requested;
  },
  setLastEjected: (key) => set({ lastEjected: key, lastEjectedReason: null }),
  setLastEjectedWithReason: (key, reason) => set({ lastEjected: key, lastEjectedReason: reason }),

  setActiveGenerating: (chatId, assistantMessageId) =>
    set({
      activeGeneratingChatId: chatId,
      activeAssistantMessageId: assistantMessageId,
      activeAssistantMessageText: null,
      activeAssistantMessageOutput: null,
    }),
  clearActiveGenerating: () =>
    set({
      activeGeneratingChatId: null,
      activeAssistantMessageId: null,
      activeAssistantMessageText: null,
      activeAssistantMessageOutput: null,
      activeRunId: null,
    }),
  setActiveAssistantMessageText: (text) => set({ activeAssistantMessageText: text }),
  setActiveAssistantMessageOutput: (output) => set({ activeAssistantMessageOutput: output }),

  activeRunId: null,
  setActiveRunId: (runId) => set({ activeRunId: runId }),
  cancelActiveRunForChat: async (chatId) => {
    for (const operation of chatOperations.values()) {
      if (operation.chatId === chatId) operation.controller.abort();
    }
    const { titleGenChatId, cancelTitleGen, activeGeneratingChatId, activeRunId } = get();
    if (titleGenChatId === chatId) {
      set({ titleGenAbortChatId: chatId });
      await cancelTitleGen().catch(() => undefined);
    }
    if (activeGeneratingChatId === chatId && activeRunId) {
      await getPlatform()
        .chat.cancelRun(activeRunId)
        .catch(() => undefined);
    }
  },

  cancelTitleGen: async () => {
    try {
      await getPlatform().chat.cancelTitleGen();
      return true;
    } catch {
      return false;
    }
  },

  canRun: (key) => {
    const s = get();
    const notLoading = !s.isModelLoading;
    const notEjecting = !s.isEjecting && s.operationState !== "ejecting";
    const notRunning = !s.isGenerating && s.operationState !== "running";
    const notStopping = s.operationState !== "stopping";
    const notTitling = !s.isTitleGenerating;
    const resident = useRuntimeSessionStore.getState().residentSession;
    const residentConflict = !!resident && resident.repoId !== key.repoId;
    return !residentConflict && notLoading && notEjecting && notRunning && notStopping && notTitling;
  },
  canStop: () => {
    const s = get();
    // A run still loading its model can be stopped: the backend checks for a cancel first.
    const running = s.isGenerating;
    const notEjecting = !s.isEjecting && s.operationState !== "ejecting";
    const notTitling = !s.isTitleGenerating;
    return running && notEjecting && notTitling && s.operationState !== "stopping";
  },
  canEject: () => {
    const s = get();
    const resident = useRuntimeSessionStore.getState().residentSession;
    const hasResident = !!resident;
    const notLoading = !s.isModelLoading;
    const notEjecting = !s.isEjecting && s.operationState !== "ejecting";
    const notRunning = !s.isGenerating && s.operationState !== "running";
    const notStopping = s.operationState !== "stopping";
    const notTitling = !s.isTitleGenerating;
    return hasResident && notLoading && notEjecting && notRunning && notStopping && notTitling;
  },

  withOperation: async (op, effect, chatId) => {
    const s = get();
    const st: OperationState = s.operationState;

    const isEjectingState = st === "ejecting";
    const isRunningState = st === "running";
    const isStoppingState = st === "stopping";
    const sameOp = st === op;
    const isTitling = s.isTitleGenerating;

    const isBlocked =
      isTitling ||
      sameOp ||
      (op === "running" && (isStoppingState || isEjectingState)) ||
      (op === "stopping" && isEjectingState) ||
      (op === "ejecting" && (isRunningState || isStoppingState));

    if (isBlocked) return Promise.resolve(null);

    const id = uuidv4();
    const controller = new AbortController();
    if (chatId !== undefined) chatOperations.set(id, { chatId, controller });
    set({ operationState: op, operationId: id });
    try {
      const result = await effect(controller.signal);
      return result;
    } finally {
      chatOperations.delete(id);
      const cur = get();
      const same = cur.operationId === id && cur.operationState === op;
      if (same) set({ operationState: "idle", operationId: null });
    }
  },
}));
