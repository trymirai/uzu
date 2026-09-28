import { describe, expect, it, vi } from "vitest";
import type { LlmRunParams, SessionOutputStats } from "@/types/llm-stream";
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
});
