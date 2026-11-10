import JSZip from "jszip";
import { beforeEach, expect, it, vi } from "vitest";
import type { ChatData } from ".";
import type { EngineModel, ModelDownloadPhase } from "@/types/model-manager";
import type { ModelsService } from "../models";

const mocks = vi.hoisted(() => ({
  files: new Map<string, string>(),
  models: [] as EngineModel[],
  deleteModel: vi.fn<(identifier: string) => Promise<void>>(),
}));

vi.mock("../shared/browser-db", () => ({
  browserFiles: {
    get: async (key: string) => mocks.files.get(key),
    put: async (key: string, value: string) => void mocks.files.set(key, value),
    delete: async (key: string) => void mocks.files.delete(key),
    keys: async () => [...mocks.files.keys()],
  },
}));
vi.mock("../../platform-singleton", () => ({
  getPlatform: () => ({
    models: {
      getModels: async () => ({ models: mocks.models, complete: true, refreshing: false }),
      deleteModel: mocks.deleteModel,
    } satisfies Partial<ModelsService>,
  }),
}));

const { webStorage } = await import("./web");

const original: ChatData = {
  metadata: { id: "browser-chat", title: "My chat", createdAt: 1000, updatedAt: 1000, messageCount: 2 },
  messages: [
    { id: "user", sender: "user", text: "Hello", timestamp: 1000 },
    { id: "assistant", sender: "assistant", text: "Hi", timestamp: 2000 },
  ],
};

const model = (identifier: string, phase: ModelDownloadPhase): EngineModel => ({
  identifier,
  repoId: `vendor/${identifier}`,
  name: identifier,
  vendor: "Vendor",
  quantization: null,
  reasoning: { kind: "unsupported" },
  supportsTools: false,
  state: { phase, downloadedKbytes: 10, totalKbytes: 10, seq: 1 },
});

beforeEach(() => {
  mocks.files.clear();
  mocks.models = [];
  mocks.deleteModel.mockReset().mockResolvedValue();
});

it("persists complete conversations and global instructions in browser storage", async () => {
  await webStorage.createOrReplaceChat(original);
  await webStorage.saveGlobalInstructions("Be concise");
  expect(mocks.files.get("chats/browser-chat.md")).toContain("Hello");
  expect(await webStorage.listChats()).toMatchObject([original.metadata]);
  expect((await webStorage.loadChat("browser-chat"))?.messages.map((message) => message.text)).toEqual(["Hello", "Hi"]);
  expect(await webStorage.loadGlobalInstructions()).toBe("Be concise");
});

it("serializes browser chat mutations and retains a manual title over a stale generated one", async () => {
  await webStorage.createOrReplaceChat(original);
  const append = { id: "next", sender: "user" as const, text: "Next question", timestamp: 3000 };
  await Promise.all([
    webStorage.updateChatTitle("browser-chat", "Manual title"),
    webStorage.updateChatTitle("browser-chat", "Generated title", "My chat"),
    webStorage.appendMessage("browser-chat", append),
  ]);
  const saved = await webStorage.loadChat("browser-chat");
  expect(saved?.metadata).toMatchObject({ title: "Manual title", messageCount: 3 });
  expect(saved?.messages.at(-1)?.text).toBe("Next question");
  const edited = await webStorage.editUserMessage("browser-chat", "user", "Replacement");
  expect(edited.messages.map((message) => message.text)).toEqual(["Replacement"]);
  expect(edited.metadata.title).toBe("Manual title");
});

it("exports browser chats as readable markdown and removes deleted conversations", async () => {
  await webStorage.createOrReplaceChat(original);
  const zip = await JSZip.loadAsync((await webStorage.exportAllChatsZip())!);
  expect(await zip.file("browser-chat.md")!.async("string")).toContain("Hello");
  await webStorage.deleteChat("browser-chat");
  expect(await webStorage.loadChat("browser-chat")).toBeNull();
  expect(await webStorage.exportAllChatsZip()).toBeNull();
});

it.each(["resident", "vendor/resident"])("keeps a resident model identified by %s during cleanup", async (resident) => {
  mocks.models = [
    model("resident", "Downloaded"),
    model("other", "Downloaded"),
    model("failed", "Error"),
    model("paused", "Paused"),
  ];
  expect((await webStorage.previewCleanup([resident])).models).toEqual({ count: 1, sizeBytes: 10_240 });
  expect(await webStorage.executeCleanup(["models"], [resident])).toEqual({
    executed: ["models"],
    modelsSkipped: ["resident", "paused"],
  });
  expect(mocks.deleteModel.mock.calls).toEqual([["vendor/other"], ["vendor/failed"]]);
});

it("deletes every chat and reports a failed model deletion as not executed", async () => {
  await webStorage.createOrReplaceChat(original);
  await webStorage.createOrReplaceChat({ ...original, metadata: { ...original.metadata, id: "second" } });
  mocks.models = [model("other", "Downloaded")];
  mocks.deleteModel.mockRejectedValueOnce(new Error("Storage is busy"));
  vi.spyOn(console, "error").mockImplementation(() => {});
  expect(await webStorage.executeCleanup(["dialogs", "models"])).toEqual({ executed: ["dialogs"], modelsSkipped: [] });
  expect(await webStorage.listChats()).toEqual([]);
});

it("does not report unsupported log cleanup as completed", async () => {
  expect(await webStorage.executeCleanup(["logs"])).toEqual({ executed: [], modelsSkipped: [] });
});
