import "fake-indexeddb/auto";
import JSZip from "jszip";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { webStorage } from "./web";
import { browserDb } from "../shared/browser-db";
import type { ChatData } from ".";
import { webModels } from "../models/web";
import type { EngineModel } from "@/types/model-manager";

const original: ChatData = {
  metadata: { id: "browser-chat", title: "My chat", createdAt: 1000, updatedAt: 1000, messageCount: 2 },
  messages: [
    { id: "user", sender: "user", text: "Hello", timestamp: 1000 },
    { id: "assistant", sender: "assistant", text: "Hi", timestamp: 2000 },
  ],
};

beforeEach(async () => {
  for (const key of await browserDb.keys("files")) await browserDb.delete("files", key);
});
afterEach(() => vi.restoreAllMocks());

it("persists complete conversations and global instructions in browser storage", async () => {
  await webStorage.createOrReplaceChat(original);
  await webStorage.saveGlobalInstructions("Be concise");
  expect(await browserDb.get("files", "chats/browser-chat.md")).toContain("Hello");
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

it.each(["catalog:resident", "vendor/resident"])(
  "preserves a resident model identified by %s during cleanup",
  async (resident) => {
    const model = (identifier: string, repoId: string): EngineModel => ({
      identifier,
      repoId,
      name: identifier,
      vendor: "Vendor",
      quantization: null,
      reasoning: { kind: "unsupported" },
      supportsTools: false,
      state: { phase: "Downloaded", downloadedKbytes: 10, totalKbytes: 10, seq: 1 },
    });
    vi.spyOn(webModels, "getModels").mockResolvedValue({
      complete: true,
      refreshing: false,
      models: [model("catalog:resident", "vendor/resident"), model("catalog:other", "vendor/other")],
    });
    const remove = vi.spyOn(webModels, "deleteModel").mockResolvedValue();
    expect((await webStorage.previewCleanup([resident])).models).toEqual({ count: 1, sizeBytes: 10_240 });
    expect(await webStorage.executeCleanup(["models"], [resident])).toEqual({
      executed: ["models"],
      modelsSkipped: ["catalog:resident"],
    });
    expect(remove).toHaveBeenCalledExactlyOnceWith("catalog:other");
  },
);

it("does not report unsupported log cleanup as completed", async () => {
  expect(await webStorage.executeCleanup(["logs"])).toEqual({ executed: [], modelsSkipped: [] });
});
