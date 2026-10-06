import { expect, it } from "vitest";
import type { ChatData } from "..";
import { extractMessageBlocks, parseMessage } from "./parse";
import { serializeToMarkdown } from "./serialize";

const chat: ChatData = {
  metadata: {
    id: "chat",
    title: "Untitled",
    messageCount: 2,
    createdAt: 1_700_000_000_000,
    updatedAt: 1_700_000_000_000,
    modelId: "vendor/model",
    modelName: "Model",
  },
  messages: [
    { id: "u1", text: "hello", sender: "user", timestamp: 1_700_000_000_000 },
    { id: "a1", text: "", sender: "assistant", modelName: "Model", timestamp: 1_700_000_000_001, perf: {} },
  ],
};

it("keeps an empty assistant placeholder through a serialize/parse round-trip", () => {
  const markdown = serializeToMarkdown(chat);
  const messages = extractMessageBlocks(markdown).map(parseMessage);

  expect(messages.map((m) => m?.id)).toEqual(["u1", "a1"]);
  expect(messages[1]).toMatchObject({ sender: "assistant", text: "" });
});
