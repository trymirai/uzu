import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatNotFoundError, type ChatData } from ".";
import { serializeToMarkdown } from "./markdown/serialize";

const files = vi.hoisted(() => ({ load: vi.fn(), save: vi.fn(), list: vi.fn(), delete: vi.fn() }));
vi.mock("./chat-files", () => ({
  loadChatFile: files.load,
  saveChatFile: files.save,
  listChatFiles: files.list,
  deleteChatFile: files.delete,
}));

import { chatRepository } from "./chat-repository";
import { webStorage } from "./web";

const timestamp = 1_700_000_000_000;
const now = timestamp + 100_000;
const original: ChatData = {
  metadata: {
    id: "chat",
    title: "Existing title",
    modelId: "vendor/model",
    modelName: "Model",
    createdAt: timestamp,
    updatedAt: timestamp,
    messageCount: 4,
  },
  messages: [
    { id: "u1", sender: "user", text: "First question", timestamp },
    { id: "a1", sender: "assistant", text: "First answer", timestamp: timestamp + 1_000 },
    {
      id: "u2",
      sender: "user",
      text: "Old question",
      timestamp: timestamp + 2_000,
      attachmentIds: ["image-id", "document-id"],
    },
    { id: "a2", sender: "assistant", text: "Old answer", timestamp: timestamp + 3_000 },
  ],
};
let disk: Map<string, string>;

beforeEach(() => {
  vi.resetAllMocks();
  vi.spyOn(Date, "now").mockReturnValue(now);
  disk = new Map([["chat.md", serializeToMarkdown(original)]]);
  files.load.mockImplementation(async (path: string) => disk.get(path) ?? null);
  files.save.mockImplementation(async (path: string, content: string) => {
    disk.set(path, content);
  });
});

afterEach(() => vi.restoreAllMocks());

describe("editUserMessage", () => {
  it("saves the edited prompt and truncates later messages while preserving attachments and metadata", async () => {
    const before = (await chatRepository.loadChat("chat"))!;
    const saved = await chatRepository.editUserMessage("chat", "u2", "Edited question\nwith a second line");

    expect(saved.messages.map((message) => message.id)).toEqual(["u1", "a1", "u2"]);
    expect(saved.messages.slice(0, 2)).toEqual(before.messages.slice(0, 2));
    expect(saved.messages[2]).toMatchObject({
      id: "u2",
      sender: "user",
      text: "Edited question\nwith a second line",
      timestamp: before.messages[2]!.timestamp,
      attachmentIds: ["image-id", "document-id"],
    });
    expect(saved.metadata).toEqual({ ...before.metadata, messageCount: 3, updatedAt: now });
    expect(await chatRepository.loadChat("chat")).toEqual(saved);
    expect(files.save).toHaveBeenCalledTimes(1);
  });

  it("loads the latest metadata inside the same queue as other chat mutations", async () => {
    const renamed = chatRepository.updateChatTitle("chat", "New title");
    const edited = chatRepository.editUserMessage("chat", "u1", "New first question");
    await renamed;
    const saved = await edited;
    expect(saved.metadata.title).toBe("New title");
    expect(saved.metadata.messageCount).toBe(1);
    expect(saved.messages.map((message) => message.text)).toEqual(["New first question"]);
    expect(await chatRepository.loadChat("chat")).toEqual(saved);
  });

  it.each([
    ["missing-chat", "u2", "Chat missing-chat not found"],
    ["chat", "missing-message", "Message missing-message not found"],
    ["chat", "a1", "Only user messages can be edited"],
  ])("rejects invalid target %s/%s without writing", async (chatId, messageId, error) => {
    const before = disk.get("chat.md");
    await expect(chatRepository.editUserMessage(chatId, messageId, "Edit")).rejects.toThrow(error);
    expect(files.save).not.toHaveBeenCalled();
    expect(disk.get("chat.md")).toBe(before);
  });

  it("rejects a failed write without losing the old conversation and allows retry", async () => {
    const before = disk.get("chat.md");
    files.save.mockRejectedValueOnce(new Error("disk full"));
    await expect(chatRepository.editUserMessage("chat", "u2", "Edit")).rejects.toThrow("disk full");
    expect(disk.get("chat.md")).toBe(before);
    expect((await chatRepository.loadChat("chat"))?.messages.map((message) => message.text)).toEqual(
      original.messages.map((message) => message.text),
    );
    const retried = await chatRepository.editUserMessage("chat", "u2", "Retried edit");
    expect(retried.messages.at(-1)?.text).toBe("Retried edit");
    expect(await chatRepository.loadChat("chat")).toEqual(retried);
  });

  it("waits for the save to succeed before returning the replacement chat", async () => {
    let finishSave!: () => void;
    files.save.mockImplementationOnce(
      (path: string, content: string) =>
        new Promise<void>((resolve) => {
          finishSave = () => {
            disk.set(path, content);
            resolve();
          };
        }),
    );
    const settled = vi.fn();
    const edited = chatRepository.editUserMessage("chat", "u2", "Edit").then(settled);
    await vi.waitFor(() => expect(files.save).toHaveBeenCalled());
    expect(settled).not.toHaveBeenCalled();
    expect((await chatRepository.loadChat("chat"))?.messages).toHaveLength(4);
    finishSave();
    await edited;
    expect(settled).toHaveBeenCalledOnce();
    expect((await chatRepository.loadChat("chat"))?.messages).toHaveLength(3);
  });

  it("rejects edits in the browser provider, which has no stored chats", async () => {
    await expect(webStorage.editUserMessage("chat", "u2", "Edit")).rejects.toBeInstanceOf(ChatNotFoundError);
  });
});

describe("updateChatTitle", () => {
  const generated = "Generated title";

  it("updates the title when the stored value matches", async () => {
    const before = (await chatRepository.loadChat("chat"))!;
    expect(await chatRepository.updateChatTitle("chat", generated, original.metadata.title)).toBe(true);
    const saved = (await chatRepository.loadChat("chat"))!;

    expect(saved.metadata).toEqual({ ...before.metadata, title: generated, updatedAt: now });
    expect(saved.messages).toEqual(before.messages);
    expect(files.save).toHaveBeenCalledOnce();
  });

  it("does not overwrite the title if the expected value changed", async () => {
    expect(await chatRepository.updateChatTitle("chat", generated, "Different title")).toBe(false);

    expect((await chatRepository.loadChat("chat"))?.metadata).toMatchObject(original.metadata);
    expect(files.save).not.toHaveBeenCalled();
  });

  it("reports a rejected CAS even when the proposed title happens to match a manual rename", async () => {
    const renamed = "My chosen title";
    expect(await chatRepository.updateChatTitle("chat", renamed)).toBe(true);
    expect(await chatRepository.updateChatTitle("chat", renamed, original.metadata.title)).toBe(false);
    // A rejected write must not advance the caller's expected title to `renamed`.
    expect(await chatRepository.updateChatTitle("chat", generated, original.metadata.title)).toBe(false);

    expect((await chatRepository.loadChat("chat"))?.metadata.title).toBe(renamed);
    expect(files.save).toHaveBeenCalledOnce();
  });

  it("reports missing chats and the browser provider as unapplied updates", async () => {
    expect(await chatRepository.updateChatTitle("missing-chat", generated)).toBe(false);
    expect(await webStorage.updateChatTitle("chat", generated)).toBe(false);
    expect(files.save).not.toHaveBeenCalled();
  });

  it("preserves a manual rename when a stale generated title and a message save are queued", async () => {
    const renamed = "My chosen title";
    const message = { id: "a3", sender: "assistant" as const, text: "New answer", timestamp: now };
    await Promise.all([
      chatRepository.updateChatTitle("chat", renamed),
      chatRepository.updateChatTitle("chat", generated, original.metadata.title),
      chatRepository.appendMessage("chat", message),
    ]);

    const saved = (await chatRepository.loadChat("chat"))!;
    expect(saved.metadata).toMatchObject({ title: renamed, messageCount: 5 });
    expect(saved.messages.at(-1)).toMatchObject(message);
    expect(files.save).toHaveBeenCalledTimes(2);
  });
});
