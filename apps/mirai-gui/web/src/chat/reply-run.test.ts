import { expect, it } from "vitest";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import type { SessionOutputFinishReason } from "@/types/llm-stream";
import { ReplyRun, ReplyText } from "./reply-run";
import type { Reply } from "./transcript";
import { chatErrorMessage } from "./uzu-values";

// Ported from the tests in desktop/src/chat/stream.rs.

it("gives an engine context overflow a readable message", () => {
  expect(chatErrorMessage({ type: "Backend", error: { type: "ContextOverflow" } })).toBe(
    "Conversation is too long for this model's context window.",
  );
});

it("passes other backend errors through", () => {
  expect(chatErrorMessage({ type: "Backend", error: { type: "NoSeedToken" } })).toBe(
    "Backend error: No seed token (both state and input are empty)",
  );
});

// ChatMessage::assistant().with_text(text).with_reasoning(reasoning)
const reply = (text: string, reasoning: string, finishReason?: SessionOutputFinishReason): Reply => ({
  message: {
    role: "assistant",
    metadata: "[]",
    content: [
      { type: "text", value: text },
      { type: "reasoning", value: reasoning },
    ],
  },
  ...(finishReason ? { finishReason } : {}),
});

it("keeps completed tool turns and the partial last turn in the final text", () => {
  const snapshots = [
    reply("Let", "First thought"),
    reply("Let me check.", "First thought", "ToolCalls"),
    reply("", "Second"),
    reply("", "Second thought", "ToolCalls"),
    reply("The", "Final thought"),
    reply("The answer.", "Final thought", "Stop"),
  ];
  const text = new ReplyText();
  for (const snapshot of snapshots) if (snapshot.finishReason) text.update(snapshot);
  expect(text.raw()).toEqual(["Let me check.\n\nThe answer.", "First thought\n\nSecond thought\n\nFinal thought"]);

  const interrupted = new ReplyText();
  for (const snapshot of snapshots.slice(0, 5)) if (snapshot.finishReason) interrupted.update(snapshot);
  interrupted.update(snapshots[4]!);
  expect(interrupted.raw()).toEqual(["Let me check.\n\nThe", "First thought\n\nSecond thought\n\nFinal thought"]);
});

it("keeps tool-call reasoning out of a promoted answer", () => {
  const text = new ReplyText();
  text.update(reply("", "Name the chat", "ToolCalls"));
  text.update(reply("", "Actual answer", "Stop"));
  expect(text.effective(true)).toEqual(["Actual answer", "Name the chat"]);

  // final_parsed is private to reply-run.ts; its output is the done event's `parsed`.
  const finish = (replies: Reply[]) => {
    const events: RunEvent[] = [];
    const run = new ReplyRun((event) => events.push(event), true, true);
    for (const reply of replies) run.reply(reply);
    run.finish(false, undefined);
    const done = events.at(-1);
    if (done?.type !== "done") throw new Error("expected a done event");
    return done.parsed;
  };
  // final_parsed("Actual answer", "Name the chat", true)
  expect(finish([reply("", "Name the chat", "ToolCalls"), reply("", "Actual answer", "Stop")])?.chainOfThought).toBe(
    "Name the chat",
  );
  // final_parsed("Actual answer", "", true)
  const parsed = finish([reply("", "Actual answer", "Stop")]);
  expect(parsed?.chainOfThought).toBe("");
  expect(parsed?.response).toBe("Actual answer");
});
