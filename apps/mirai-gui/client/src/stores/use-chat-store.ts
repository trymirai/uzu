import { useChatSessionStore } from "./use-chat-session-store";
import { defaultReasoningEffort, useModelParamsStore } from "./use-model-params-store";
import type { Message, PerfStats } from "@/types/message";
import type { ChatMetadata } from "@/platform/services/storage";
import { v4 as uuidv4 } from "uuid";
import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import type { LlmAsyncStream, LlmRunParams, SessionOutputStats } from "@/types/llm-stream";
import { finalizeAssistantMessage, persistMessage, persistMessageError, persistMessagePatch } from "./chat/persistence";
import { runChatTitleGeneration } from "./chat/title-generation";

export type ChatState = {
  chatModels: Record<string, { modelId: string; modelName: string }>;
  lastUsedModel: {
    modelId: string;
    modelName: string;
  } | null;
  messages: Message[];
  currentChatId: string | null;
  savedChats: ChatMetadata[];
  autoSelectSuppressed?: Record<string, boolean>;
  /** Bumped when a message failed to reach disk; the UI reports it. */
  saveFailureCount: number;

  setChatModel: (chatId: string, modelId: string, modelName: string) => void;

  // For a run that may outlive the page: the message enters the view only
  // while that chat is open, the caller persists it by value.
  addMessageTo: (chatId: string, message: Omit<Message, "id" | "timestamp">) => Message;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  discardMessage: (chatId: string, id: string) => Promise<void>;
  switchMessageVersion: (messageId: string, versionIndex: number) => void;
  updateMessagePerf: (messageId: string, perf: Partial<PerfStats>, stats?: SessionOutputStats) => void;
  updateVersionPerf: (
    messageId: string,
    versionIndex: number,
    perf: Partial<PerfStats>,
    stats?: SessionOutputStats,
  ) => void;

  getEffectiveModelForMessage: (
    messageId: string,
    chatId: string,
  ) => { modelId: string | null; modelName: string | null };
  clearChat: () => void;
  setCurrentChatId: (chatId: string | null) => void;
  persistMessage: (chatId: string, messageId: string, message?: Message) => Promise<void>;
  persistMessagePatch: (chatId: string, messageId: string, patch: Partial<Message>) => Promise<void>;
  persistMessageError: (
    chatId: string,
    messageId: string,
    text: string,
    error: string,
    attachmentIds?: string[],
  ) => Promise<void>;
  finalizeAssistantMessage: (
    chatId: string,
    messageId: string,
    text: string,
    parsed?: { chainOfThought?: string; response?: string },
  ) => Promise<void>;
  loadChat: (chatId: string) => Promise<void>;
  loadSavedChats: () => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  createNewChat: (chatId: string) => void;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  generateChatTitle: (chatId: string, userText: string) => Promise<{ ok: boolean; error?: string }>;
  suppressAutoSelect: (chatId: string, value: boolean) => void;

  runChatStream: (params: LlmRunParams) => LlmAsyncStream;
};

export const useChatStore = create<ChatState>((set, get) => ({
  chatModels: {},
  lastUsedModel: null,
  messages: [],
  currentChatId: null,
  savedChats: [],
  autoSelectSuppressed: {},
  saveFailureCount: 0,

  setChatModel: (chatId: string, modelId: string, modelName: string) => {
    set((state) => ({
      chatModels: {
        ...state.chatModels,
        [chatId]: { modelId, modelName },
      },
      lastUsedModel: modelId ? { modelId, modelName } : state.lastUsedModel,
    }));
  },

  addMessageTo: (chatId, messageData) => {
    const message: Message = { ...messageData, id: uuidv4(), timestamp: Date.now() };
    if (get().currentChatId === chatId) set((state) => ({ messages: [...state.messages, message] }));
    return message;
  },

  updateMessage: (id: string, updates: Partial<Message>) => {
    set((state) => ({
      messages: state.messages.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg)),
    }));
  },

  discardMessage: async (chatId: string, id: string) => {
    if (get().currentChatId === chatId) {
      set((s) => ({ messages: s.messages.filter((m) => m.id !== id) }));
    }
    const { storage } = getPlatform();
    try {
      await storage.removeMessage(chatId, id);
      set({ savedChats: await storage.listChats() });
    } catch (e) {
      console.error("[storage] discardMessage failed", { chatId, id }, e);
    }
  },

  switchMessageVersion: (messageId: string, versionIndex: number) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.id !== messageId || !msg.versions) return msg;

        const version = msg.versions[versionIndex];
        if (!version) return msg;

        return {
          ...msg,
          text: version.text,
          modelId: version.modelId,
          modelName: version.modelName,
          currentVersionIndex: versionIndex,
          timestamp: version.timestamp,
          output: version.output || msg.output,
        };
      }),
    }));
  },

  updateMessagePerf: (messageId, perf, stats) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === messageId
          ? {
              ...msg,
              perf: { ...(msg.perf || {}), ...perf },
              stats: stats ?? msg.stats,
            }
          : msg,
      ),
    }));
  },

  updateVersionPerf: (messageId, versionIndex, perf, stats) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.id !== messageId || !msg.versions) return msg;
        const updatedVersions = msg.versions.map((v, i) =>
          i === versionIndex
            ? {
                ...v,
                perf: { ...(v.perf || {}), ...perf },
                stats: stats ?? v.stats,
              }
            : v,
        );
        return { ...msg, versions: updatedVersions };
      }),
    }));
  },

  getEffectiveModelForMessage: (messageId: string, chatId: string) => {
    const { messages, chatModels } = get();
    const message = messages.find((msg) => msg.id === messageId);
    const chatModel = chatModels[chatId];

    if (!message) {
      return chatModel
        ? { modelId: chatModel.modelId, modelName: chatModel.modelName }
        : { modelId: null, modelName: null };
    }

    return {
      modelId: message.modelId || (chatModel ? chatModel.modelId : null),
      modelName: message.modelName || (chatModel ? chatModel.modelName : null),
    };
  },

  clearChat: () => {
    set({ messages: [] });
  },

  setCurrentChatId: (chatId: string | null) => {
    set({ currentChatId: chatId });
  },

  persistMessage: (chatId, messageId, message) => persistMessage({ get, set }, chatId, messageId, message),

  persistMessagePatch: (chatId, messageId, patch) => persistMessagePatch({ get, set }, chatId, messageId, patch),

  persistMessageError: (chatId, messageId, text, error, attachmentIds) =>
    persistMessageError({ get, set }, chatId, messageId, text, error, attachmentIds),

  finalizeAssistantMessage: (chatId, messageId, text, parsed) =>
    finalizeAssistantMessage({ get, set }, chatId, messageId, text, parsed),

  loadChat: async (chatId: string) => {
    const chatData = await getPlatform().storage.loadChat(chatId);
    // The user may have opened another chat while this one was being read.
    if (get().currentChatId !== chatId) return;
    if (chatData) {
      set({
        messages: chatData.messages,
        currentChatId: chatId,
      });

      if (chatData.metadata.modelId && chatData.metadata.modelName) {
        get().setChatModel(chatId, chatData.metadata.modelId, chatData.metadata.modelName);
      }
    }
  },

  loadSavedChats: async () => {
    const savedChats = await getPlatform().storage.listChats();
    set({ savedChats });
  },

  deleteChat: async (chatId: string) => {
    await useChatSessionStore.getState().cancelActiveRunForChat(chatId);
    await getPlatform().storage.deleteChat(chatId);
    const { savedChats } = get();
    const updatedChats = savedChats.filter((chat) => chat.id !== chatId);
    set({ savedChats: updatedChats });

    if (get().currentChatId === chatId) {
      set({ messages: [], currentChatId: null });
    }
  },

  createNewChat: (chatId: string) => {
    set({
      messages: [],
      currentChatId: chatId,
    });
  },

  updateChatTitle: async (chatId: string, title: string) => {
    const { storage } = getPlatform();
    await storage.updateChatTitle(chatId, title);
    const savedChats = await storage.listChats();
    set({ savedChats });
  },

  generateChatTitle: async (chatId: string, userText: string): Promise<{ ok: boolean; error?: string }> => {
    const s = useChatSessionStore.getState();
    const st = s.operationState;
    const blocked =
      s.isGenerating ||
      s.isEjecting ||
      s.isModelLoading ||
      s.isTitleGenerating ||
      st === "stopping" ||
      st === "ejecting";
    if (blocked) return { ok: true };

    useChatSessionStore.getState().setTitleGenerating(true);
    useChatSessionStore.getState().setTitleGenChat(chatId);

    try {
      await runChatTitleGeneration(chatId, userText, get, set);
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      useChatSessionStore.getState().setTitleGenerating(false);
      useChatSessionStore.getState().setTitleGenChat(null);
    }
  },

  suppressAutoSelect: (chatId: string, value: boolean) => {
    set((state) => ({
      autoSelectSuppressed: {
        ...(state.autoSelectSuppressed || {}),
        [chatId]: value,
      },
    }));
  },

  runChatStream: (params: LlmRunParams) => {
    const { getParams, globalReasoningEnabled } = useModelParamsStore.getState();
    const modelParams = getParams(params.repoId);
    return getPlatform().chat.runStream({
      ...params,
      samplingPolicy: modelParams.sampling,
      reasoningEffort: modelParams.reasoningEffort ?? defaultReasoningEffort(globalReasoningEnabled),
    });
  },
}));
