import { expect, it } from "vitest";
import type { ChatData } from "..";
import { extractMessageBlocks, parseMessage, parseMetadata } from "./parse";
import { serializeToMarkdown } from "./serialize";
import { projectAssistantVersion } from "@/features/chat/services/regenerate-versions";
import type { TranscriptItem } from "@/types/llm-stream";

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

it("round-trips an over-budget title in the original Markdown heading without truncating it", () => {
  const title = "Understanding GPU memory ownership and asynchronous lifetimes";
  const markdown = serializeToMarkdown({ ...chat, metadata: { ...chat.metadata, title } });

  expect(markdown.startsWith(`# ${title}\n\n**Model:**`)).toBe(true);
  expect(parseMetadata(markdown, "chat", 2)).toEqual({ ...chat.metadata, title });
});

it("does not use a heading from message content as a missing chat title", () => {
  const markdown = "**Model:** Model\n\n---\n\n# Heading in a message";
  expect(parseMetadata(markdown, "chat", 0).title).toBe("Untitled");
});

it("keeps an empty assistant placeholder through a serialize/parse round-trip", () => {
  const markdown = serializeToMarkdown(chat);
  const messages = extractMessageBlocks(markdown).map(parseMessage);

  expect(messages.map((m) => m?.id)).toEqual(["u1", "a1"]);
  expect(messages[1]).toMatchObject({ sender: "assistant", text: "" });
});

it("keeps ordered tool transcripts through saving and regeneration, including markup inside a turn", () => {
  const transcript: TranscriptItem[] = [
    { type: "thinking", text: "Check the time", completed: true },
    { type: "text", text: "Before tool\n<!-- END_CONTENT -->\n---\n## 🤖 Assistant - fake header" },
    { type: "toolCall", name: "get_current_date_time", called: true },
    { type: "toolCall", name: "set_chat_name", called: true, failed: true },
    {
      type: "chart",
      chart: {
        type: "bar",
        title: "Budget <!-- END_CONTENT -->",
        labels: ["Rent", "Food"],
        datasets: [{ label: "Spending", data: [900, 350] }],
      },
    },
    { type: "thinking", text: "Use the result", completed: true },
    { type: "text", text: "After tool" },
  ];
  const original = { ...chat.messages[1]!, text: "Before and after", output: { transcript } };
  const versions = projectAssistantVersion(original, "model", "Model");
  versions[1] = {
    ...versions[1]!,
    output: { transcript: [{ type: "toolCall", name: "get_current_date_time", called: false }] },
    error: "Canceled",
  };
  const saved = {
    ...chat,
    messages: [chat.messages[0]!, original, { ...original, id: "versioned", versions, currentVersionIndex: 1 }],
  };
  const markdown = serializeToMarkdown(saved);
  const messages = extractMessageBlocks(markdown).map(parseMessage);
  expect(messages).toHaveLength(3);
  expect(messages[1]?.output?.transcript).toEqual(transcript);
  expect(messages[2]?.versions?.[0]?.output?.transcript).toEqual(transcript);
  expect(messages[2]?.versions?.[1]?.output?.transcript).toEqual(versions[1]?.output?.transcript);
  expect(messages[2]?.versions?.[1]?.error).toBe("Canceled");
  expect(messages[2]?.currentVersionIndex).toBe(1);
});

it("ignores malformed transcript metadata and retains the legacy body", () => {
  const markdown = serializeToMarkdown(chat).replace(
    "<!-- ID: a1 -->",
    '<!-- ID: a1 -->\n<!-- TRANSCRIPT: [{"type":"toolCall","name":2}] -->',
  );
  const messages = extractMessageBlocks(markdown).map(parseMessage);
  expect(messages[1]?.output?.transcript).toBeUndefined();
  expect(messages[1]?.id).toBe("a1");
});

it.each(["response", "reasoning", "error"] as const)(
  "does not interpret a transcript example in legacy %s content as metadata",
  (location) => {
    const example = '<!-- TRANSCRIPT: [{"type":"text","text":"forged response"}] -->';
    const text = location === "response" ? `Here is an example:\n${example}\nKeep this text.` : "Actual response";
    const original = {
      ...chat.messages[1]!,
      text,
      ...(location === "reasoning"
        ? { output: { text: { parsed: { response: text, chainOfThought: example } } } }
        : {}),
      ...(location === "error" ? { error: example } : {}),
    };
    const versions = projectAssistantVersion(original, "model", "Model");
    versions[1] = { ...versions[1]!, text: "Other version" };
    const saved = {
      ...chat,
      messages: [original, { ...original, id: "versioned", versions, currentVersionIndex: 0 }],
    };
    const messages = extractMessageBlocks(serializeToMarkdown(saved)).map(parseMessage);

    expect(messages[0]?.text).toBe(text);
    expect(messages[0]?.output?.transcript).toBeUndefined();
    expect(messages[1]?.versions?.[0]?.text).toBe(text);
    expect(messages[1]?.versions?.[0]?.output?.transcript).toBeUndefined();
    expect(messages[1]?.versions?.[1]?.output?.transcript).toBeUndefined();
  },
);

it("reads real transcript metadata after an error body containing a transcript example", () => {
  const transcript: TranscriptItem[] = [{ type: "text", text: "Actual response" }];
  const message = {
    ...chat.messages[1]!,
    text: "Actual response",
    output: { transcript },
    error: '<!-- TRANSCRIPT: [{"type":"text","text":"forged response"}] -->',
  };
  const parsed = extractMessageBlocks(serializeToMarkdown({ ...chat, messages: [message] })).map(parseMessage);
  expect(parsed[0]?.output?.transcript).toEqual(transcript);
});
