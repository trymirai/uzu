import type { PlatformClient } from "./platform-client";
import type { ChatService } from "./services/chat";
import { runLlmStream, type RunEvent, type RunTransport } from "./services/chat/run-stream";
import type { DownloadEvent, ModelsService } from "./services/models";
import type { SessionLoadingPayload, SessionService, SessionStatePayload } from "./services/session";
import { ChatNotFoundError, type ChatData, type StorageService } from "./services/storage";
import { webDialogs } from "./services/dialogs/web";
import { webSettings } from "./services/settings/web";
import { webSystem } from "./services/system/web";
import { webSystemUi } from "./services/system-ui/web";
import { webUpdater } from "./services/updater/web";
import type { EngineModel } from "@/types/model-manager";
import type { LlmRunParams, SessionOutputStats, TranscriptItem } from "@/types/llm-stream";

/*
 * `pnpm demo:dev`: the browser build with a pretend engine, for design work and recordings.
 * Models download in a few seconds, the first message loads its model, and every answer runs
 * through the same events the desktop backend sends: reading the prompt, thinking, a tool call,
 * then the answer word by word. Chats live in memory until the tab closes.
 */

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/* ------------------------------------------------------------------ models */

const GB = 1024 ** 3;

type DemoModel = Omit<EngineModel, "state"> & { bytes: number; downloaded: boolean };

// Vendor icons as the model registry serves them.
const REGISTRY_ICONS = "https://assets.trymirai.com/registry/icons";
const ICONS = {
  Alibaba: {
    light: `${REGISTRY_ICONS}/alibaba/796fd6ea01ca351f6397e2579d9412162293c410f14cdd3f70e99c4181d68bcd.svg`,
    dark: `${REGISTRY_ICONS}/alibaba/796fd6ea01ca351f6397e2579d9412162293c410f14cdd3f70e99c4181d68bcd.svg`,
  },
  LiquidAI: {
    light: `${REGISTRY_ICONS}/liquidai/858e1f116f012f80f1901198e41684a9f82e8416ccad716dc3fc981dcd2fdcba.svg`,
    dark: `${REGISTRY_ICONS}/liquidai/da45e2a75d8405a12071c24d97137c568ed2d2d72f95805ee83f523c37435168.svg`,
  },
  Meta: {
    light: `${REGISTRY_ICONS}/meta/1b4bfe85e334335d96e7881bdf5755024b9a2ae47395f009a586635f518ee4d2.svg`,
    dark: `${REGISTRY_ICONS}/meta/1b4bfe85e334335d96e7881bdf5755024b9a2ae47395f009a586635f518ee4d2.svg`,
  },
  Google: {
    light: `${REGISTRY_ICONS}/google/422ca3c3cc6d48c1a8339c7d5fe6f3545da84e5f64916126eb6144b947697b5f.svg`,
    dark: `${REGISTRY_ICONS}/google/422ca3c3cc6d48c1a8339c7d5fe6f3545da84e5f64916126eb6144b947697b5f.svg`,
  },
};

const reasoning = { kind: "toggle", defaultEffort: "default" } as const;

const MODELS: DemoModel[] = [
  {
    identifier: "Qwen/Qwen3.8-27B-MIRAI-M",
    vendor: "Alibaba",
    vendorIcons: ICONS.Alibaba,
    name: "Qwen3.8 27B Mirai Medium",
    familyIdentifier: "qwen3.8",
    familyName: "Qwen3.8",
    paramSize: 27e9,
    reasoning,
    supportsTools: true,
    quantization: "MIRAI-M",
    quantizationBits: 4,
    bytes: 16 * GB,
    downloaded: true,
  },
  {
    identifier: "Qwen/Qwen3.8-27B-MIRAI-L",
    vendor: "Alibaba",
    vendorIcons: ICONS.Alibaba,
    name: "Qwen3.8 27B Mirai Large",
    familyIdentifier: "qwen3.8",
    familyName: "Qwen3.8",
    paramSize: 27e9,
    reasoning,
    supportsTools: true,
    quantization: "MIRAI-L",
    quantizationBits: 8,
    bytes: 29 * GB,
    downloaded: false,
  },
  {
    identifier: "LiquidAI/LFM2.5-1.2B-Instruct",
    vendor: "LiquidAI",
    vendorIcons: ICONS.LiquidAI,
    name: "LFM2.5 1.2B Instruct",
    familyIdentifier: "lfm2.5",
    familyName: "LFM2.5",
    paramSize: 1.2e9,
    reasoning: { kind: "unsupported" },
    supportsTools: true,
    bytes: 1.3 * GB,
    downloaded: true,
  },
  {
    identifier: "LiquidAI/LFM2.5-2.6B-Instruct",
    vendor: "LiquidAI",
    vendorIcons: ICONS.LiquidAI,
    name: "LFM2.5 2.6B Instruct",
    familyIdentifier: "lfm2.5",
    familyName: "LFM2.5",
    paramSize: 2.6e9,
    reasoning: { kind: "unsupported" },
    supportsTools: true,
    bytes: 2.8 * GB,
    downloaded: false,
  },
  {
    identifier: "Qwen/Qwen3.5-9B",
    vendor: "Alibaba",
    vendorIcons: ICONS.Alibaba,
    name: "Qwen3.5 9B",
    familyIdentifier: "qwen3.5",
    familyName: "Qwen3.5",
    paramSize: 9e9,
    reasoning,
    supportsTools: true,
    quantization: "MIRAI-M",
    quantizationBits: 4,
    bytes: 5.4 * GB,
    downloaded: false,
  },
  {
    identifier: "meta-llama/Llama-3.2-3B-Instruct",
    vendor: "Meta",
    vendorIcons: ICONS.Meta,
    name: "Llama 3.2 3B Instruct",
    familyIdentifier: "llama-3.2",
    familyName: "Llama-3.2",
    paramSize: 3e9,
    reasoning: { kind: "unsupported" },
    supportsTools: true,
    bytes: 2.1 * GB,
    downloaded: false,
  },
  {
    identifier: "google/gemma-3-4b-it",
    vendor: "Google",
    vendorIcons: ICONS.Google,
    name: "Gemma 3 4B It",
    familyIdentifier: "gemma-3",
    familyName: "Gemma-3",
    paramSize: 4e9,
    reasoning: { kind: "unsupported" },
    supportsTools: false,
    bytes: 2.6 * GB,
    downloaded: false,
  },
];

type Download = { completed: number; paused: boolean; seq: number };
const downloads = new Map<string, Download>();
const downloadListeners = new Set<(event: DownloadEvent) => void>();
const catalogListeners = new Set<() => void>();
const emitDownload = (event: DownloadEvent) => downloadListeners.forEach((listener) => listener(event));

const engineModel = ({ bytes, downloaded, ...model }: DemoModel): EngineModel => {
  const download = downloads.get(model.identifier);
  const completed = downloaded ? bytes : (download?.completed ?? 0);
  const phase = downloaded ? "Downloaded" : !download ? "NotDownloaded" : download.paused ? "Paused" : "Downloading";
  return {
    ...model,
    repoId: model.identifier,
    state: {
      phase,
      totalKbytes: bytes / 1024,
      downloadedKbytes: completed / 1024,
      seq: download?.seq ?? 0,
    },
  };
};

const runDownload = async (identifier: string) => {
  const model = MODELS.find((m) => m.identifier === identifier);
  const download = downloads.get(identifier);
  if (!model || !download) return;
  // About eight seconds whatever the size, in uneven steps like a real connection.
  while (!download.paused && download.completed < model.bytes) {
    await sleep(120);
    if (download.paused || downloads.get(identifier) !== download) return;
    download.completed = Math.min(model.bytes, download.completed + (model.bytes / 66) * (0.6 + Math.random() * 0.8));
    download.seq += 1;
    emitDownload({
      identifier,
      seq: download.seq,
      kind: "progress",
      completedBytes: download.completed,
      totalBytes: model.bytes,
    });
  }
  if (download.completed < model.bytes) return;
  model.downloaded = true;
  download.seq += 1;
  emitDownload({ identifier, seq: download.seq, kind: "done" });
  catalogListeners.forEach((listener) => listener());
};

const demoModels: ModelsService = {
  getModels: async () => ({ models: MODELS.map(engineModel), complete: true, refreshing: false }),
  refreshModels: async () => {},
  startDownload: async (identifier) => {
    const previous = downloads.get(identifier);
    const download = { completed: previous?.completed ?? 0, paused: false, seq: (previous?.seq ?? 0) + 1 };
    downloads.set(identifier, download);
    emitDownload({ identifier, seq: download.seq, kind: "resumed" });
    void runDownload(identifier);
  },
  pauseDownload: async (identifier) => {
    const download = downloads.get(identifier);
    if (!download) return;
    download.paused = true;
    download.seq += 1;
    emitDownload({ identifier, seq: download.seq, kind: "paused" });
  },
  resumeDownload: (identifier) => demoModels.startDownload(identifier),
  deleteModel: async (identifier) => {
    const model = MODELS.find((m) => m.identifier === identifier);
    if (model) model.downloaded = false;
    const seq = (downloads.get(identifier)?.seq ?? 0) + 1;
    downloads.delete(identifier);
    emitDownload({ identifier, seq, kind: "deleted" });
    catalogListeners.forEach((listener) => listener());
  },
  onDownloadEvent: (listener) => {
    downloadListeners.add(listener);
    return () => downloadListeners.delete(listener);
  },
  onModelsChanged: (listener) => {
    catalogListeners.add(listener);
    return () => catalogListeners.delete(listener);
  },
};

/* ------------------------------------------------------------------ session */

let resident: string | null = null;
const stateListeners = new Set<(payload: SessionStatePayload) => void>();
const loadingListeners = new Set<(payload: SessionLoadingPayload) => void>();
const emitState = (payload: SessionStatePayload) => stateListeners.forEach((listener) => listener(payload));

const demoSession: SessionService = {
  onSessionState: (listener) => {
    stateListeners.add(listener);
    return () => stateListeners.delete(listener);
  },
  onSessionLoading: (listener) => {
    loadingListeners.add(listener);
    return () => loadingListeners.delete(listener);
  },
  ejectSession: async ({ repoId }) => {
    emitState({ active: true, repoId, isEjecting: true });
    await sleep(900);
    resident = null;
    emitState({ active: false, repoId, isEjecting: false });
  },
};

const loadModel = async (repoId: string) => {
  if (resident === repoId) return;
  if (resident) emitState({ active: false, repoId: resident });
  loadingListeners.forEach((listener) => listener({ status: "start", repoId }));
  await sleep(2400);
  resident = repoId;
  emitState({ active: true, repoId });
};

/* ------------------------------------------------------------------ answers */

type Script = {
  name: string;
  thinking: string;
  tool?: string;
  answer: string;
};

const SCRIPTS: Array<{ match: RegExp; script: Script }> = [
  {
    match: /track_caller/i,
    script: {
      name: "Track caller on trait fn",
      thinking:
        "The question is whether an attribute on a trait method declaration reaches the implementations. track_caller is special: the reference says it applies to the trait's prototype and every implementation inherits it. Worth checking the dyn case, which goes through a shim.",
      tool: "search_rust_reference",
      answer: [
        "Yes. Put `#[track_caller]` on the method in the trait declaration and every implementation picks it up, including ones that override a default method. The impls don't need the attribute themselves.",
        '```rust\nuse std::panic::Location;\n\ntrait Report {\n    #[track_caller]\n    fn report(&self);\n}\n\nstruct S;\n\nimpl Report for S {\n    fn report(&self) {\n        // No #[track_caller] here: it comes from the trait.\n        eprintln!("called from {}", Location::caller());\n    }\n}\n```',
        "One caveat: calls through `dyn Report` or a function pointer go through a shim, so they lose the real caller and report the function's own definition instead. Direct, statically dispatched calls keep it.",
      ].join("\n\n"),
    },
  },
  {
    match: /version|since when|since what/i,
    script: {
      name: "Rust release",
      thinking: "When did this land? It came with the stabilisation of track_caller itself, in 1.46.",
      tool: "search_release_notes",
      answer:
        "Since Rust 1.46.0, released on 27 August 2020, the same release that stabilized `#[track_caller]`.\n\nInheriting it from the trait declaration was part of that from the start, so there's no separate version to wait for.",
    },
  },
  {
    // A long think, for trying what the app offers while you wait.
    match: /think hard|take your time|count to/i,
    script: {
      name: "A long think",
      thinking: Array.from(
        { length: 14 },
        (_, i) =>
          `Step ${i + 1}: weighing another way to approach this, checking it against what came before, and keeping only what holds up.`,
      ).join(" "),
      answer:
        "That took a while. In the desktop app a long think like this is uzu working through the problem on your Mac, one token at a time.",
    },
  },
  {
    match: /date|time|today|day is it/i,
    script: {
      name: "Today",
      thinking: "I need the current date, and there's a tool for that.",
      tool: "get_current_date_time",
      answer: `It's ${new Date().toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}, ${new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })} on this Mac.`,
    },
  },
];

const fallback = (prompt: string): Script => ({
  name: prompt.split(/\s+/).slice(0, 5).join(" ") || "New chat",
  thinking:
    "Let me read that carefully and answer plainly. Nothing here needs to leave the device, so I can work from what I know.",
  answer: [
    "This is the demo engine, so the answer is a stand-in, but everything around it is real: the model loaded on this machine, read your prompt, thought it through, and is now writing word by word.",
    "In the desktop app the same steps come from **uzu**, running the model on Apple silicon. Nothing you type is sent anywhere.",
    "- Reading: the cat puts its goggles on\n- Thinking: it looks up and drifts\n- Tools: its eyes turn like gears\n- Writing: its eyes become lines of text",
  ].join("\n\n"),
});

const words = (text: string) => text.match(/\S+\s*|\s+/g) ?? [];

const stats = (input: number, output: number, ms: number): SessionOutputStats => ({
  tokensPerForwardPass: 2.4,
  prefillStats: { duration: 0.38, tokensCount: input, tokensPerSecond: input / 0.38 },
  generateStats: { duration: ms / 1000, tokensCount: output, tokensPerSecond: output / (ms / 1000) },
  totalStats: { duration: 0.38 + ms / 1000, tokensCountInput: input, tokensCountOutput: output },
});

const canceled = new Set<string>();

const demoTransport: RunTransport = {
  start: async (runId: string, params: LlmRunParams, onEvent: (event: RunEvent) => void) => {
    const live = () => !canceled.has(runId);
    const prompt = params.messages.filter((m) => m.role === "user").at(-1)?.content ?? "";
    const script = SCRIPTS.find(({ match }) => match.test(prompt))?.script ?? fallback(prompt);

    await loadModel(params.repoId);
    if (!live()) return;
    // Prefill: the prompt is read before anything is shown.
    await sleep(900 + Math.min(1200, prompt.length * 6));
    if (!live()) return;
    if (params.modelChatNamingEnabled) onEvent({ type: "chatName", name: script.name });

    const items: TranscriptItem[] = [];
    const push = (item: TranscriptItem) => {
      items.push(item);
      onEvent({ type: "transcript", items: [...items] });
    };
    const stream = async (index: number, text: string, pace: number) => {
      for (const word of words(text)) {
        if (!live()) return;
        onEvent({ type: "transcriptDelta", index, delta: word });
        const item = items[index];
        if (item && (item.type === "text" || item.type === "thinking"))
          items[index] = { ...item, text: item.text + word };
        await sleep(pace + Math.random() * pace);
      }
    };

    const thinks = params.reasoningEffort !== "disabled";
    if (thinks) {
      push({ type: "thinking", text: "" });
      await stream(items.length - 1, script.thinking, 45);
      if (!live()) return;
      items[items.length - 1] = { type: "thinking", text: script.thinking, completed: true };
      onEvent({ type: "transcript", items: [...items] });
    }
    if (script.tool) {
      push({ type: "toolCall", name: script.tool, called: false });
      await sleep(1800);
      if (!live()) return;
      items[items.length - 1] = { type: "toolCall", name: script.tool, called: true };
      onEvent({ type: "transcript", items: [...items] });
      await sleep(300);
    }
    const started = performance.now();
    push({ type: "text", text: "" });
    await stream(items.length - 1, script.answer, 22);
    if (!live()) return;
    const output = words(script.answer).length;
    onEvent({
      type: "done",
      text: script.answer,
      finishReason: "Stop",
      stats: stats(Math.ceil(prompt.length / 4) + 24, output, performance.now() - started),
      transcript: [...items],
      ...(params.modelChatNamingEnabled ? { chatName: script.name } : {}),
    });
  },
  cancel: async (runId) => {
    canceled.add(runId);
  },
};

const demoChat: ChatService = {
  runStream: (params) => runLlmStream(demoTransport, params),
  cancelRun: (runId) => demoTransport.cancel(runId),
  generateTitle: async ({ userText }) => fallback(userText).name,
  cancelTitleGen: async () => {},
  getSamplingDefaults: async () => ({
    type: "Stochastic",
    temperature: 0.6,
    topK: 20,
    topP: 0.95,
    minP: null,
    repetitionPenalty: 1.1,
    suffixRepetitionLength: 1024,
  }),
};

/* ------------------------------------------------------------------ storage */

const chats = new Map<string, ChatData>();
let globalInstructions: string | null = null;

const touch = (chat: ChatData) => {
  chat.metadata.updatedAt = Date.now();
  chat.metadata.messageCount = chat.messages.length;
};

const demoStorage: StorageService = {
  listChats: async () => [...chats.values()].map((chat) => ({ ...chat.metadata })),
  loadChat: async (chatId) => structuredClone(chats.get(chatId) ?? null),
  createOrReplaceChat: async (chat) => {
    chats.set(chat.metadata.id, structuredClone(chat));
  },
  appendMessage: async (chatId, message) => {
    const chat = chats.get(chatId);
    if (!chat) throw new ChatNotFoundError(chatId);
    chat.messages.push(structuredClone(message));
    touch(chat);
  },
  updateStoredMessage: async (chatId, messageId, patch) => {
    const chat = chats.get(chatId);
    const index = chat?.messages.findIndex((m) => m.id === messageId) ?? -1;
    if (!chat || index < 0) return;
    chat.messages[index] = { ...chat.messages[index]!, ...structuredClone(patch) };
    touch(chat);
  },
  editUserMessage: async (chatId, messageId, text) => {
    const chat = chats.get(chatId);
    const index = chat?.messages.findIndex((m) => m.id === messageId) ?? -1;
    if (!chat || index < 0) throw new ChatNotFoundError(chatId);
    chat.messages = [...chat.messages.slice(0, index), { ...chat.messages[index]!, text }];
    touch(chat);
    return structuredClone(chat);
  },
  removeMessage: async (chatId, messageId) => {
    const chat = chats.get(chatId);
    if (!chat) return;
    chat.messages = chat.messages.filter((m) => m.id !== messageId);
    touch(chat);
  },
  updateChatTitle: async (chatId, title, expectedTitle) => {
    const chat = chats.get(chatId);
    if (!chat || (expectedTitle !== undefined && chat.metadata.title !== expectedTitle)) return false;
    chat.metadata.title = title;
    touch(chat);
    return true;
  },
  deleteChat: async (chatId) => {
    chats.delete(chatId);
  },
  exportAllChatsZip: async () => null,
  saveBinaryFile: async () => false,
  saveGlobalInstructions: async (content) => {
    globalInstructions = content;
  },
  loadGlobalInstructions: async () => globalInstructions,
  previewCleanup: async () => ({
    dialogs: { count: chats.size, sizeBytes: 0 },
    models: { count: 0, sizeBytes: 0 },
    logs: { sizeBytes: 0 },
  }),
  executeCleanup: async (categories) => ({ executed: categories, modelsSkipped: [] }),
};

export class DemoPlatformClient implements PlatformClient {
  readonly chat = demoChat;
  readonly session = demoSession;
  readonly models = demoModels;
  readonly storage = demoStorage;
  readonly settings = webSettings;
  readonly system = webSystem;
  readonly systemUi = webSystemUi;
  readonly updater = webUpdater;
  readonly dialogs = webDialogs;
}
