import { describe, expect, it, vi } from "vitest";
import type { LlmRunParams, SessionOutputStats, TranscriptItem } from "@/types/llm-stream";
import { runLlmStream, type RunEvent, type RunTransport } from "./run-stream";

const params: LlmRunParams = { repoId: "vendor/model", messages: [{ role: "user", content: "hi" }] };

const stats: SessionOutputStats = {
  prefillStats: { duration: 0.1, tokensCount: 3, tokensPerSecond: 30 },
  generateStats: { duration: 1, tokensCount: 10, tokensPerSecond: 10 },
  totalStats: { duration: 1.1, tokensCountInput: 3, tokensCountOutput: 10 },
};

const fakeTransport = (start?: RunTransport["start"]) => {
  let emit: (event: RunEvent) => void = () => {
    throw new Error("run was not started");
  };
  const transport: RunTransport = {
    start:
      start ??
      ((_runId, _params, onEvent) => {
        emit = onEvent;
        return new Promise(() => {});
      }),
    cancel: vi.fn(() => Promise.resolve()),
  };
  return { transport, emit: (event: RunEvent) => emit(event) };
};

const readAll = async (stream: ReadableStream<string>): Promise<string> => {
  const reader = stream.getReader();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return text;
    text += value;
  }
};

describe("runLlmStream", () => {
  it("replays ordered transcript snapshots including explicit naming calls", async () => {
    const initial: TranscriptItem[] = [
      { type: "thinking", text: "Check the clock", completed: true },
      { type: "toolCall", name: "set_chat_name", called: true, failed: true },
      { type: "toolCall", name: "get_current_date_time", called: false },
    ];
    const { transport } = fakeTransport(async (_id, _params, emit) => {
      emit({ type: "transcript", items: initial });
    });
    const run = runLlmStream(transport, params);
    const listener = vi.fn();
    run.onTranscript(listener);
    expect(listener).toHaveBeenCalledExactlyOnceWith(initial);
    await run.cancel();
    expect((await run.result).transcript).toEqual(initial);
  });

  it("appends Unicode text and reasoning without changing previously published snapshots", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);
    const initial: TranscriptItem[] = [
      { type: "thinking", text: "考" },
      { type: "toolCall", name: "get_current_date_time", called: true },
      { type: "text", text: "Hello" },
    ];
    const snapshots: TranscriptItem[][] = [];
    run.onTranscript((items) => snapshots.push(items));
    emit({ type: "transcript", items: initial });
    emit({ type: "transcriptDelta", index: 0, delta: "える 🧠" });
    emit({ type: "transcriptDelta", index: 2, delta: ", თბილისი 👋" });
    emit({ type: "done", text: "Hello, თბილისი 👋", stats });

    expect(snapshots).toEqual([
      initial,
      [{ type: "thinking", text: "考える 🧠" }, initial[1], initial[2]],
      [{ type: "thinking", text: "考える 🧠" }, initial[1], { type: "text", text: "Hello, თბილისი 👋" }],
    ]);
    expect(initial[0]).toEqual({ type: "thinking", text: "考" });
    expect(initial[2]).toEqual({ type: "text", text: "Hello" });
    expect(snapshots[1]?.[1]).toBe(initial[1]);
    expect(snapshots[2]?.[0]).toBe(snapshots[1]?.[0]);
    expect((await run.result).transcript).toBe(snapshots[2]);
    expect(await readAll(run.stream)).toBe("");
  });

  it("replays deltas received before subscription as one complete snapshot", async () => {
    const { transport } = fakeTransport(async (_id, _params, emit) => {
      emit({ type: "transcript", items: [{ type: "thinking", text: "Let me" }] });
      emit({ type: "transcriptDelta", index: 0, delta: " think" });
    });
    const run = runLlmStream(transport, params);
    const listener = vi.fn();
    run.onTranscript(listener);
    expect(listener).toHaveBeenCalledExactlyOnceWith([{ type: "thinking", text: "Let me think" }]);
    await run.cancel();
  });

  it("replaces delta state with structural and final snapshots", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);
    const listener = vi.fn();
    run.onTranscript(listener);
    emit({ type: "transcript", items: [{ type: "thinking", text: "Draft" }] });
    emit({ type: "transcriptDelta", index: 0, delta: " reasoning" });
    emit({
      type: "transcript",
      items: [
        { type: "thinking", text: "Draft reasoning", completed: true },
        { type: "toolCall", name: "get_current_date_time", called: false },
      ],
    });
    emit({ type: "transcript", items: [{ type: "text", text: "Revised" }] });
    emit({ type: "transcriptDelta", index: 0, delta: " answer" });
    expect(listener).toHaveBeenLastCalledWith([{ type: "text", text: "Revised answer" }]);
    const final: TranscriptItem[] = [{ type: "text", text: "Final answer" }];
    emit({ type: "done", text: "Final answer", stats, transcript: final });
    expect(listener).toHaveBeenLastCalledWith(final);
    expect((await run.result).transcript).toEqual(final);
  });

  it.each([undefined, [{ type: "toolCall", name: "get_current_date_time", called: false }]] as const)(
    "reports an invalid delta target instead of silently losing text (%j)",
    async (initial) => {
      const { transport, emit } = fakeTransport();
      const run = runLlmStream(transport, params);
      if (initial) emit({ type: "transcript", items: [...initial] });
      emit({ type: "transcriptDelta", index: 0, delta: "Lost text" });
      await expect(readAll(run.stream)).rejects.toThrow("Invalid transcript delta at index 0");
      expect((await run.result).error).toBe("Invalid transcript delta at index 0");
      expect((await run.result).transcript).toEqual(initial);
    },
  );

  it.each(["done", "error", "cancel"] as const)(
    "preserves an incrementally assembled transcript through %s and ignores late updates",
    async (end) => {
      const { transport, emit } = fakeTransport();
      const run = runLlmStream(transport, params);
      const items: TranscriptItem[] = [
        { type: "text", text: "Checking." },
        { type: "toolCall", name: "get_current_date_time", called: true },
        { type: "text", text: "It is noon." },
      ];
      const listener = vi.fn();
      run.onTranscript(listener);
      emit({ type: "transcript", items: [...items.slice(0, -1), { type: "text", text: "It is" }] });
      emit({ type: "transcriptDelta", index: 2, delta: " noon." });
      expect(listener).toHaveBeenLastCalledWith(items);
      listener.mockClear();
      if (end === "done") emit({ type: "done", text: "Checking.\n\nIt is noon.", stats });
      else {
        if (end === "cancel") await run.cancel();
        else {
          emit({ type: "error", error: "Stopped" });
          await expect(readAll(run.stream)).rejects.toThrow("Stopped");
        }
      }
      emit({ type: "transcript", items: [] });
      emit({ type: "transcriptDelta", index: 2, delta: " Ignore this." });
      expect(listener).not.toHaveBeenCalled();
      expect((await run.result).transcript).toEqual(items);
    },
  );

  it("streams chunks, closes on done and resolves the result", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);
    const parsed = vi.fn();
    run.onParsed(parsed);

    emit({ type: "chunk", delta: "Hel" });
    emit({ type: "chunk", delta: "lo", parsed: { chainOfThought: "thinking" } });
    emit({ type: "done", text: "Hello", stats, finishReason: "Stop", parsed: { response: "Hello" } });

    expect(await readAll(run.stream)).toBe("Hello");
    expect(await run.result).toEqual({ text: "Hello", stats, finishReason: "Stop", parsed: { response: "Hello" } });
    expect(parsed.mock.calls).toEqual([[{ chainOfThought: "thinking" }], [{ response: "Hello" }]]);
  });

  it("ignores events that arrive after done", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);

    emit({ type: "done", text: "A", stats, finishReason: "Stop" });
    emit({ type: "chunk", delta: "late" });
    emit({ type: "error", error: "late failure" });

    expect(await readAll(run.stream)).toBe("");
    expect((await run.result).error).toBeUndefined();
  });

  it("reports a backend error on both the stream and the result", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);

    emit({ type: "error", error: "model exploded" });

    await expect(readAll(run.stream)).rejects.toThrow("model exploded");
    expect(await run.result).toMatchObject({ text: "", error: "model exploded" });
  });

  it("turns a rejected start into an error instead of hanging", async () => {
    const { transport } = fakeTransport(() => Promise.reject(new Error("invalid payload")));
    const run = runLlmStream(transport, params);

    expect(await run.result).toMatchObject({ text: "", error: "Error: invalid payload" });
    await expect(readAll(run.stream)).rejects.toThrow("invalid payload");
  });

  it("settles as cancelled on cancel and drops whatever the backend sends next", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);

    emit({ type: "chunk", delta: "partial" });
    await run.cancel();
    await run.cancel();
    emit({ type: "done", text: "partial and more", stats, finishReason: "Stop" });

    expect(transport.cancel).toHaveBeenCalledWith(run.runId);
    expect(await run.result).toMatchObject({ text: "", finishReason: "Cancelled" });
  });

  it("settles as cancelled even when the cancel call fails", async () => {
    const { transport } = fakeTransport();
    transport.cancel = () => Promise.reject(new Error("ipc down"));
    const run = runLlmStream(transport, params);

    await expect(run.cancel()).rejects.toThrow("ipc down");
    expect(await run.result).toMatchObject({ finishReason: "Cancelled" });
  });

  it("settles as cancelled when the reader is cancelled", async () => {
    const { transport } = fakeTransport();
    const run = runLlmStream(transport, params);

    await run.stream.getReader().cancel();

    expect(await run.result).toMatchObject({ finishReason: "Cancelled" });
  });

  it("keeps the first outcome when cancel races a finished run", async () => {
    const { transport, emit } = fakeTransport();
    const run = runLlmStream(transport, params);

    emit({ type: "done", text: "Done", stats, finishReason: "Stop" });
    await run.cancel();

    expect(await run.result).toMatchObject({ text: "Done", finishReason: "Stop" });
  });

  describe("chat names", () => {
    it("emits names separately from text and deduplicates the done summary", async () => {
      const { transport, emit } = fakeTransport();
      const run = runLlmStream(transport, params);
      const named = vi.fn();
      run.onChatName(named);

      emit({ type: "chatName", name: "First name" });
      emit({ type: "chunk", delta: "Hello" });
      emit({ type: "chatName", name: "Updated name" });
      emit({ type: "done", text: "Hello", stats, chatName: "Updated name" });

      expect(named.mock.calls).toEqual([["First name"], ["Updated name"]]);
      expect(await readAll(run.stream)).toBe("Hello");
      expect(await run.result).toMatchObject({ text: "Hello", chatName: "Updated name" });
    });

    it("notifies listeners when the name only arrives in the done summary", async () => {
      const { transport, emit } = fakeTransport();
      const run = runLlmStream(transport, params);
      const named = vi.fn();
      run.onChatName(named);

      emit({ type: "done", text: "Hello", stats, chatName: "A greeting" });

      expect(named).toHaveBeenCalledExactlyOnceWith("A greeting");
      expect(await run.result).toMatchObject({ chatName: "A greeting" });
    });

    it("replays a name emitted before the listener could subscribe", async () => {
      const { transport } = fakeTransport(async (_runId, _params, emit) => {
        emit({ type: "chatName", name: "A greeting" });
        emit({ type: "done", text: "Hello", stats, chatName: "A greeting" });
      });
      const run = runLlmStream(transport, params);
      const named = vi.fn();
      run.onChatName(named);

      expect(named).toHaveBeenCalledExactlyOnceWith("A greeting");
      expect(await run.result).toMatchObject({ chatName: "A greeting" });
    });

    it.each(["done", "error", "cancel"] as const)(
      "preserves the last name after %s and ignores later names",
      async (end) => {
        const { transport, emit } = fakeTransport();
        const run = runLlmStream(transport, params);
        const named = vi.fn();
        run.onChatName(named);
        emit({ type: "chatName", name: "Keep this name" });

        if (end === "cancel") {
          await run.cancel();
        } else if (end === "error") {
          emit({ type: "error", error: "generation failed" });
          await expect(readAll(run.stream)).rejects.toThrow("generation failed");
        } else {
          emit({ type: "done", text: "Hello", stats });
        }
        emit({ type: "chatName", name: "Ignore this name" });
        emit({ type: "done", text: "Late result", stats, chatName: "Ignore this summary" });

        expect(named).toHaveBeenCalledExactlyOnceWith("Keep this name");
        expect(await run.result).toMatchObject({ chatName: "Keep this name" });
      },
    );
  });
});
