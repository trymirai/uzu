import { expect, it } from "vitest";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import type { TranscriptItem } from "@/types/llm-stream";
import { showChart } from "./tools";
import {
  hiddenCall,
  Transcript,
  type ContentBlock,
  type Reply,
  type ToolCallValue,
  type TranscriptMessage,
} from "./transcript";

// Ported from the tests in desktop/src/chat/transcript.rs.

// readMessage serializes an empty metadata map as "[]".
const NO_METADATA = "[]";
const assistant = (...content: ContentBlock[]): TranscriptMessage => ({
  role: "assistant",
  metadata: NO_METADATA,
  content,
});
const tool = (...content: ContentBlock[]): TranscriptMessage => ({ role: "tool", metadata: NO_METADATA, content });
const text = (value: string): ContentBlock => ({ type: "text", value });
const reasoning = (value: string): ContentBlock => ({ type: "reasoning", value });
// uzu's ToolCallCandidate (and any other block readBlock does not copy).
const candidate: ContentBlock = { type: "other" };
const call = (name: string, identifier?: string, args: unknown = {}): ToolCallValue => ({
  ...(identifier === undefined ? {} : { identifier }),
  name,
  arguments: JSON.stringify(args),
});
const namingCall = (hidden: unknown): ToolCallValue => call("set_chat_name", undefined, { name: "A name", hidden });
const calls = (value: ToolCallValue): ContentBlock => ({ type: "toolCall", call: value });
const resultBlock = (name: string | undefined, identifier: string | undefined, value: unknown): ContentBlock => ({
  type: "toolResult",
  ...(identifier === undefined ? {} : { identifier }),
  ...(name === undefined ? {} : { name }),
  value: JSON.stringify(value),
});
const result = (name?: string, identifier?: string): TranscriptMessage => tool(resultBlock(name, identifier, "result"));
const reply = (message: TranscriptMessage, finished = false): Reply => ({
  message,
  ...(finished ? { finishReason: "ToolCalls" as const } : {}),
});
const thinking = (text: string, completed: boolean): TranscriptItem => ({ type: "thinking", text, completed });

/**
 * Applies the streamed events the way the client does and checks them, after
 * every reply, against a transcript that always takes full snapshots.
 */
class StreamOracle {
  transcript = new Transcript();
  oracle = new Transcript();
  displayed: TranscriptItem[] = [];
  previous?: Reply;

  send(reply: Reply, promoteReasoning: boolean): RunEvent | undefined {
    const event = this.transcript.streamUpdate(this.previous, reply, promoteReasoning);
    this.oracle.update(reply);
    this.oracle.setReasoningPromotion(promoteReasoning);
    const serialized = event && (JSON.parse(JSON.stringify(event)) as RunEvent);
    if (event?.type === "transcript") this.displayed = structuredClone(event.items);
    else if (event?.type === "transcriptDelta") {
      const item = this.displayed[event.index];
      if (item?.type !== "text" && item?.type !== "thinking")
        throw new Error(`delta targets a non-text item: ${JSON.stringify(item)}`);
      item.text += event.delta;
    } else if (event) throw new Error("unexpected transcript event");
    expect(this.displayed).toEqual(this.oracle.items(false));
    this.previous = reply;
    return serialized;
  }

  toolResults(results: TranscriptMessage[]) {
    this.transcript.flushReply(this.previous);
    this.transcript.toolResults(results);
    this.oracle.toolResults(results);
    this.displayed = this.transcript.items(false);
    expect(this.displayed).toEqual(this.oracle.items(false));
  }

  finish() {
    this.transcript.flushReply(this.previous);
    expect(this.transcript.items(true)).toEqual(this.oracle.items(true));
  }
}

it("streams deltas that preserve whitespace, Unicode and decoder corrections", () => {
  for (const asReasoning of [false, true]) {
    const stream = new StreamOracle();
    for (const value of [
      "",
      "  ",
      "  A",
      "  A ",
      "  A \n",
      "  A \nB",
      "  A \nB�",
      "  A \nB🙂",
      "  A \nB🙂‍",
      "  A \nB🙂‍↔️",
      "  A \nB🙂‍↔️\u0000",
      "Replaced",
      "R",
    ]) {
      stream.send(reply(assistant(asReasoning ? reasoning(value) : text(value))), false);
    }
    stream.finish();
  }
});

it("streams growing text as a delta against the last snapshot", () => {
  const stream = new StreamOracle();
  stream.send(reply(assistant(text("Hello"))), false);
  const event = stream.send(reply(assistant(text("Hello world"))), false);
  expect(event).toEqual({ type: "transcriptDelta", index: 0, delta: " world" });
  // The Rust test also asserts that the transcript still owns only the initial
  // message and is marked dirty until finish; those are private ownership
  // details (avoiding a clone per token), so only the event sequence is ported.
  stream.finish();
});

it("streams hidden naming candidates and merged reasoning without extra snapshots", () => {
  const stream = new StreamOracle();
  const initial = assistant(reasoning("Before"));
  stream.send(reply(initial), false);
  for (let index = 0; index < 2; index++) {
    // Each candidate is a different partial "<tool_call>…" value in Rust; both copy to an opaque block.
    expect(stream.send(reply(assistant(...initial.content, candidate)), false)).toBeUndefined();
  }
  stream.send(reply(assistant(...initial.content, calls(namingCall(true))), true), false);
  stream.toolResults([result("set_chat_name")]);
  stream.send(reply(assistant(reasoning("After "))), false);
  const event = stream.send(reply(assistant(reasoning("After naming"))), false);
  expect(event).toEqual({ type: "transcriptDelta", index: 0, delta: " naming" });
  expect(stream.displayed).toEqual([thinking("Before\n\nAfter naming", false)]);
  stream.send(reply(assistant(reasoning("After naming"), text("Answer"))), false);
  stream.finish();
});

it("requires snapshots for structural changes, promotion changes and terminal metadata", () => {
  const stream = new StreamOracle();
  let message = assistant(reasoning("  Answer "));
  stream.send(reply(message), false);
  const promoted = stream.send(reply(message), true);
  expect(promoted?.type).toBe("transcript");
  message = assistant(reasoning("  Answer now "));
  expect(stream.send(reply(message), true)?.type).toBe("transcriptDelta");
  expect(stream.send(reply(message), false)?.type).toBe("transcript");
  message = assistant(reasoning("  Answer now "), text("Visible"));
  expect(stream.send(reply(message), false)?.type).toBe("transcript");
  message = assistant(reasoning("Corrected"), text("Visible text"));
  expect(stream.send(reply(message), false)?.type).toBe("transcript");
  // A new object: the transcripts keep references to earlier messages.
  message = { ...message, metadata: JSON.stringify([["test", "true"]]) };
  expect(stream.send(reply(message), false)?.type).toBe("transcript");
  const terminal: Reply = { message, finishReason: "Cancelled" };
  expect(stream.send(terminal, false)?.type).toBe("transcript");
  stream.finish();

  // Cancellation and errors may arrive without a final reply. Flush the
  // latest streamed suffix before completing the current thinking item.
  for (const promote of [false, true]) {
    const stream = new StreamOracle();
    stream.send(reply(assistant(reasoning("Partial"))), promote);
    stream.send(reply(assistant(reasoning("Partial answer"))), promote);
    stream.finish();
  }
});

it("streams text after a displayed chart as deltas that exclude the chart", () => {
  const stream = new StreamOracle();
  stream.send(reply(assistant(calls(call("show_chart", "chart"))), true), false);
  stream.toolResults([
    tool(
      resultBlock("show_chart", "chart", {
        type: "bar",
        title: "Previously displayed chart",
        labels: ["A"],
        datasets: [{ label: "Values", data: [2] }],
      }),
    ),
  ]);
  expect(stream.displayed[0]?.type).toBe("chart");
  let value = "";
  for (let index = 0; index < 512; index++) {
    value += "x";
    const event = stream.send(reply(assistant(text(value))), false);
    expect(event).toBeDefined();
    if (index > 0) expect(event).toEqual({ type: "transcriptDelta", index: 1, delta: "x" });
  }
  stream.finish();
  // The Rust test also measures serialized wire bytes for 512 and 1024
  // characters to check linear growth; byte-size assertions are not ported.
});

it("completes tools in chronological order before another assistant token", () => {
  const transcript = new Transcript();
  transcript.update(reply(assistant(reasoning("Need time"))));
  expect(transcript.items(false)).toEqual([thinking("Need time", false)]);
  transcript.update(
    reply(assistant(reasoning("Need time"), text("Checking."), calls(call("get_current_date_time", "date"))), true),
  );
  const initial = transcript.items(false);
  expect(initial[0]).toEqual(thinking("Need time", true));
  expect(initial[1]).toEqual({ type: "text", text: "Checking." });
  expect(initial[2]).toEqual({ type: "toolCall", name: "get_current_date_time", called: false });
  transcript.toolResults([result("get_current_date_time", "date")]);
  expect(transcript.items(false)[2]).toEqual({ type: "toolCall", name: "get_current_date_time", called: true });
  transcript.update(reply(assistant(text("It is noon."))));
  const items = transcript.items(true);
  expect(items).toHaveLength(4);
  expect(items[3]).toEqual({ type: "text", text: "It is noon." });
});

it("bridges thinking across hidden naming without collapsing it between turns", () => {
  const transcript = new Transcript();
  const message = assistant(reasoning("A name"), text("\n"));
  transcript.update(reply(assistant(...message.content, candidate)));
  expect(transcript.items(false)).toEqual([thinking("A name", false)]);
  transcript.update(reply(assistant(...message.content, calls(namingCall(true))), true));
  expect(transcript.items(false)).toEqual([thinking("A name", false)]);
  transcript.toolResults([result("set_chat_name")]);
  expect(transcript.items(false)).toEqual([thinking("A name", false)]);
  expect(transcript.items(true)).toEqual([thinking("A name", true)]);
  transcript.update(reply(assistant(reasoning("Answer"))));
  expect(transcript.items(false)).toEqual([thinking("A name\n\nAnswer", false)]);
  transcript.update(reply(assistant(reasoning("Answer reasoning"))));
  expect(transcript.items(false)).toEqual([thinking("A name\n\nAnswer reasoning", false)]);
  transcript.update(reply(assistant(reasoning("Answer reasoning"), text("The answer."))));
  expect(transcript.items(false)).toEqual([
    thinking("A name\n\nAnswer reasoning", true),
    { type: "text", text: "The answer." },
  ]);
});

it("keeps explicit naming visible and lets it separate thinking", () => {
  const transcript = new Transcript();
  transcript.update(reply(assistant(reasoning("Requested rename"), calls(namingCall(false))), true));
  transcript.toolResults([result("set_chat_name")]);
  transcript.update(reply(assistant(reasoning("Now answer"))));
  expect(transcript.items(false)).toEqual([
    thinking("Requested rename", true),
    { type: "toolCall", name: "set_chat_name", called: true },
    thinking("Now answer", false),
  ]);
});

it("hides naming calls only for true hidden values", () => {
  const visible = [
    call("set_chat_name"),
    namingCall(false),
    namingCall("false"),
    namingCall("maybe"),
    namingCall(1),
    namingCall(null),
    { name: "set_chat_name", arguments: "not json" },
    call("get_current_date_time", undefined, { hidden: true }),
  ];
  for (const value of visible) {
    expect(hiddenCall(value)).toBe(false);
    const transcript = new Transcript();
    transcript.update(reply(assistant(calls(value)), true));
    expect(transcript.items(false)).toEqual([expect.objectContaining({ type: "toolCall", called: false })]);
  }
  expect(hiddenCall(namingCall(true))).toBe(true);
  expect(hiddenCall(namingCall(" TRUE "))).toBe(true);
});

it("closes thinking on a terminal assistant reason without visible output", () => {
  const transcript = new Transcript();
  transcript.update({ message: assistant(reasoning("Finished")), finishReason: "Stop" });
  expect(transcript.items(false)).toEqual([thinking("Finished", true)]);
});

it("matches each result to one call by ID or name and never completes unrelated calls", () => {
  const transcript = new Transcript();
  transcript.update(
    reply(
      assistant(
        calls(namingCall(true)),
        calls(call("get_current_date_time")),
        calls(call("get_current_date_time")),
        calls(call("get_current_date_time", "final")),
      ),
      true,
    ),
  );
  transcript.toolResults([result("set_chat_name"), result("get_current_date_time")]);
  const called = (transcript: Transcript) =>
    transcript.items(false).map((item) => {
      if (item.type !== "toolCall") throw new Error("only tool calls");
      return item.called;
    });
  expect(called(transcript)).toEqual([true, false, false]);
  transcript.toolResults([result(undefined, "final")]);
  expect(called(transcript)).toEqual([true, false, true]);
  transcript.toolResults([result()]);
  expect(called(transcript)).toEqual([true, false, true]);
  transcript.toolResults([result("get_current_date_time")]);
  expect(called(transcript)).toEqual([true, true, true]);
});

it("does not fall back to a shared name when IDs mismatch", () => {
  const transcript = new Transcript();
  transcript.update(reply(assistant(calls(call("date", "a"))), true));
  transcript.toolResults([result("date", "b")]);
  expect(transcript.items(false)).toEqual([{ type: "toolCall", name: "date", called: false }]);
});

it("marks only the matching call as failed on a tool error", () => {
  const transcript = new Transcript();
  transcript.update(
    reply(
      assistant(
        calls(call("set_chat_name", "failed")),
        calls(call("set_chat_name", "succeeded")),
        calls(call("set_chat_name", "pending")),
      ),
      true,
    ),
  );
  transcript.toolResults([
    result("set_chat_name", "succeeded"),
    tool(resultBlock("set_chat_name", "failed", { error: "Chat names must contain 3 to 50 characters" })),
  ]);
  const items = transcript.items(false);
  expect(items).toEqual([
    { type: "toolCall", name: "set_chat_name", called: true, failed: true },
    { type: "toolCall", name: "set_chat_name", called: true },
    { type: "toolCall", name: "set_chat_name", called: false },
  ]);
  const serialized = JSON.parse(JSON.stringify(items)) as Record<string, unknown>[];
  expect(serialized[0]!.failed).toBe(true);
  expect(serialized[1]).not.toHaveProperty("failed");
  expect(serialized[2]).not.toHaveProperty("failed");
});

it("replaces the matching call with its completed chart in transcript order", () => {
  const chart = (title: string) => ({
    type: "bar",
    title,
    labels: ["A"],
    datasets: [{ label: "Values", data: [2] }],
  });
  const first = chart("First chart");
  const second = chart("Second chart");
  // The transcript must use the canonical result, including markup argument coercion.
  const firstCall = call("show_chart", "first", { chart: JSON.stringify(first) });
  const secondCall = call("show_chart", "second", { chart: second });
  // What nativeTools returns for the call: the tool's value as JSON.
  const execute = (value: ToolCallValue): unknown => showChart(JSON.parse(value.arguments) as Record<string, unknown>);
  const firstResult = execute(firstCall);
  const secondResult = execute(secondCall);
  const chartResult = (identifier: string, value: unknown) => tool(resultBlock("show_chart", identifier, value));
  const transcript = new Transcript();
  transcript.update(
    reply(
      assistant(
        reasoning("Compare the data"),
        calls(firstCall),
        calls(call("get_current_date_time", "date")),
        calls(secondCall),
        text("The comparison."),
      ),
      true,
    ),
  );
  transcript.toolResults([chartResult("second", secondResult)]);
  let items = transcript.items(false);
  expect(items[1]).toMatchObject({ type: "toolCall", called: false });
  expect(items[3]).toMatchObject({ type: "chart", chart: { title: "Second chart" } });
  transcript.toolResults([chartResult("first", firstResult), result("get_current_date_time", "date")]);
  items = transcript.items(true);
  expect(items).toHaveLength(5);
  expect(items[0]).toMatchObject({ type: "thinking", completed: true });
  expect(items[1]).toMatchObject({ type: "chart", chart: { title: "First chart" } });
  expect(items[2]).toEqual({ type: "toolCall", name: "get_current_date_time", called: true });
  expect(items[3]).toMatchObject({ type: "chart", chart: { title: "Second chart" } });
  expect(items[4]).toEqual({ type: "text", text: "The comparison." });
  const serialized = JSON.parse(JSON.stringify(items[1])) as { type: string; chart: { title: string } };
  expect(serialized.type).toBe("chart");
  expect(serialized.chart.title).toBe("First chart");
});

it("keeps chart errors and invalid chart results as failed tool calls", () => {
  for (const value of [
    { error: "Invalid chart" },
    "Chart displayed",
    { type: "bar", title: "Invalid", labels: ["A", "B"], datasets: [{ label: "Values", data: [1] }] },
    {
      type: "bar",
      title: "Invalid",
      labels: ["A"],
      datasets: [{ label: "Values", data: [1] }],
      options: { onClick: "alert(1)" },
    },
  ]) {
    const transcript = new Transcript();
    transcript.update(reply(assistant(calls(call("show_chart"))), true));
    transcript.toolResults([tool(resultBlock("show_chart", undefined, value))]);
    expect(transcript.items(true)).toEqual([{ type: "toolCall", name: "show_chart", called: true, failed: true }]);
  }
});

it("promotes only the current assistant turn and keeps partial activity when finishing", () => {
  const transcript = new Transcript();
  transcript.update(reply(assistant(reasoning("Find time"), calls(call("date"))), true));
  transcript.toolResults([result("date")]);
  transcript.update(reply(assistant(reasoning("Actual answer"))));
  transcript.setReasoningPromotion(true);
  const items = transcript.items(true);
  expect(items[0]).toEqual(thinking("Find time", true));
  expect(items[2]).toEqual({ type: "text", text: "Actual answer" });
  transcript.setReasoningPromotion(false);
  expect(transcript.items(true)[2]).toEqual(thinking("Actual answer", true));
});

it("keeps reasoning that was already displayed as an answer when an error ends the run", () => {
  const transcript = new Transcript();
  transcript.update(reply(assistant(reasoning("Partial answer"))));
  transcript.setReasoningPromotion(true);
  const displayed = transcript.items(false);
  expect(displayed).toEqual([{ type: "text", text: "Partial answer" }]);
  // An error finishes the existing view without reclassifying its text.
  expect(transcript.items(true)).toEqual(displayed);
});
