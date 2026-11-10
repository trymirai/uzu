import { expect, it } from "vitest";
import { assistantHistoryMessage, parsedFrom, sameHistoryMessage, sanitize, type HistoryMessage } from "./text";

// Ported from the tests in desktop/src/chat/messages.rs. The build_messages and
// build_tail_messages tests need uzu ChatMessage objects and are not ported.

it("drops replacement and control characters but keeps emoji sequences", () => {
  expect(sanitize("a�b\u0000c﻿d\n")).toBe("abcd\n");
  expect(sanitize("👩‍💻 ❤️ 👨‍👩‍👧")).toBe("👩‍💻 ❤️ 👨‍👩‍👧");
});

it("does not reuse incomplete or promoted generated history", () => {
  expect(assistantHistoryMessage("\n ", "reasoning", false)).toBeUndefined();
  expect(assistantHistoryMessage("answer", "reasoning", true)).toBeUndefined();
});

it("keeps reasoning in exact generated history", () => {
  const message = assistantHistoryMessage("answer", "reasoning", false);
  expect(message?.content).toBe("answer");
  expect(message?.reasoningContent).toBe("reasoning");
});

// Ported from desktop/src/chat/mod.rs. plan_run's TS counterpart is the private
// BrowserChatRuntime.continuation; only its per-message comparison
// (sameHistoryMessage) is reachable here, so the repo and effort checks are not ported.
const message = (role: HistoryMessage["role"], content: string, reasoningContent?: string): HistoryMessage => ({
  role,
  content,
  ...(reasoningContent === undefined ? {} : { reasoningContent }),
});
const isPrefix = (history: HistoryMessage[], request: HistoryMessage[]) =>
  history.every((message, index) => sameHistoryMessage(message, request[index]!));

it("continues the resident session from normalized client history", () => {
  const question = message("user", "question");
  const rawText = "An\u0000 answer� 👩‍💻\n";
  const rawReasoning = "\nConsider﻿ this�.\n";
  const projected = assistantHistoryMessage(rawText, rawReasoning, false)!;
  expect(projected.content).toBe("An answer 👩‍💻\n");
  expect(projected.reasoningContent).toBe("Consider this.");
  const remembered = [question, projected];

  // The next request uses the finalized client fields, not raw engine output.
  const parsed = parsedFrom(rawText, rawReasoning)!;
  const request = [question, message("assistant", sanitize(rawText), parsed.chainOfThought), message("user", "next")];
  expect(isPrefix(remembered, request)).toBe(true);
});

it("continues only when the reasoning prefix matches", () => {
  const history = [message("user", "question"), message("assistant", "answer", "first path")];

  const continued = [...history, message("user", "next")];
  expect(isPrefix(history, continued)).toBe(true);

  const changed = [history[0]!, message("assistant", "answer", "second path"), message("user", "next")];
  expect(isPrefix(history, changed)).toBe(false);
});
