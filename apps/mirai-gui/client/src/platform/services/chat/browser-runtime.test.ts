import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BrowserRuntime, browserInferenceError } from "./browser-runtime";
import type { EngineRequest, EngineResponse } from "./browser-protocol";
import { emptyStats } from "./empty-stats";
import { webSettings } from "../settings/web";

class FakeWorker {
  onmessage?: (event: MessageEvent<EngineResponse>) => void;
  onerror?: (event: ErrorEvent) => void;
  onmessageerror?: () => void;
  requests: EngineRequest[] = [];
  postMessage = (request: EngineRequest) => this.requests.push(request);
  terminate = vi.fn();
  send(message: EngineResponse) {
    this.onmessage?.({ data: message } as MessageEvent<EngineResponse>);
  }
}

beforeEach(() => {
  vi.stubGlobal("isSecureContext", true);
  vi.stubGlobal("crossOriginIsolated", true);
  Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
  localStorage.clear();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

it("reports browser capability and isolation failures before creating an engine worker", async () => {
  vi.stubGlobal("crossOriginIsolated", false);
  expect(browserInferenceError()).toContain("cross-origin isolation");
  const createWorker = vi.fn();
  await expect(
    new BrowserRuntime(createWorker).transport.start("run", { repoId: "model", messages: [] }, vi.fn()),
  ).rejects.toThrow("cross-origin isolation");
  expect(createWorker).not.toHaveBeenCalled();
});

it("streams worker events and replays the active session to late subscribers", async () => {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const onEvent = vi.fn();
  const run = runtime.transport.start("run", { repoId: "model", messages: [] }, onEvent);
  worker.send({ type: "loading", state: { repoId: "model", status: "start" } });
  const loading = vi.fn();
  runtime.onSessionLoading(loading);
  expect(loading).toHaveBeenCalledWith({ repoId: "model", status: "start" });
  worker.send({ type: "session", state: { active: true, repoId: "model" } });
  const session = vi.fn();
  runtime.onSessionState(session);
  expect(session).toHaveBeenCalledWith({ active: true, repoId: "model" });
  worker.send({ id: "run", type: "event", event: { type: "chunk", delta: "Hello" } });
  worker.send({ id: "run", type: "event", event: { type: "done", text: "Hello", stats: emptyStats() } });
  worker.send({ id: "run", type: "result" });
  await run;
  expect(onEvent).toHaveBeenCalledTimes(2);
  const eject = runtime.eject("model");
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await eject;
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(session).toHaveBeenLastCalledWith({ active: false, repoId: "model", isEjecting: false });
});

it("rejects overlapping runs and routes cancellation to the active response", async () => {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const run = runtime.transport.start("first", { repoId: "model", messages: [] }, vi.fn());
  const loadRequest = worker.requests[0]!;
  if (loadRequest.type !== "run") throw new Error("Expected a run request");
  expect(Atomics.load(loadRequest.loadCancellation, 0)).toBe(0);
  vi.spyOn(worker, "postMessage").mockImplementation((request) => {
    if (request.type === "cancel") expect(Atomics.load(loadRequest.loadCancellation, 0)).toBe(1);
    return worker.requests.push(request);
  });
  await expect(runtime.transport.start("second", { repoId: "model", messages: [] }, vi.fn())).rejects.toThrow(
    "already running",
  );
  const cancellation = runtime.transport.cancel("first");
  expect(worker.requests.at(-1)).toMatchObject({ type: "cancel", runId: "first" });
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await cancellation;
  worker.send({ id: "first", type: "result" });
  await run;
});

it("terminates a stalled worker when cancellation cannot be acknowledged", async () => {
  vi.useFakeTimers();
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const run = runtime.transport.start("run", { repoId: "model", messages: [] }, vi.fn());
  const cancelled = runtime.transport.cancel("run");
  const results = Promise.allSettled([run, cancelled]);
  await vi.advanceTimersByTimeAsync(5000);
  expect(await results).toMatchObject([{ status: "rejected" }, { status: "rejected" }]);
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it("generates real worker titles and cancels title generation through the same session", async () => {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const title = runtime.generateTitle({ repoId: "model", userText: "Explain gravity" });
  const request = worker.requests[0]!;
  expect(request).toMatchObject({ type: "title", params: { userText: "Explain gravity" } });
  if (request.type !== "title") throw new Error("Expected a title request");
  await expect(runtime.transport.start("overlap", { repoId: "model", messages: [] }, vi.fn())).rejects.toThrow(
    "already running",
  );
  const cancellation = runtime.cancelTitleGen();
  expect(Atomics.load(request.loadCancellation, 0)).toBe(1);
  expect(worker.requests.at(-1)).toMatchObject({ type: "cancel", runId: request.id });
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await cancellation;
  worker.send({ id: request.id, type: "result", value: "" });
  expect(await title).toBe("");
  const retry = runtime.generateTitle({ repoId: "model", userText: "Explain gravity" });
  worker.send({ id: worker.requests.at(-1)!.id, type: "result", value: "Understanding gravity" });
  expect(await retry).toBe("Understanding gravity");
});

it("can eject a loading model and clears its target without waiting for a forced worker stop", async () => {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const onEvent = vi.fn();
  const run = runtime.transport.start("loading", { repoId: "model", messages: [] }, onEvent);
  const loadRequest = worker.requests[0]!;
  if (loadRequest.type !== "run") throw new Error("Expected a run request");
  worker.send({ type: "session", state: { active: false, repoId: "previous/model", isEjecting: false } });
  worker.send({ type: "loading", state: { status: "start", repoId: "model" } });
  const ejected = runtime.eject("model");
  expect(Atomics.load(loadRequest.loadCancellation, 0)).toBe(1);
  const cancel = worker.requests.at(-1)!;
  expect(cancel.type).toBe("cancel");
  worker.send({ type: "session", state: { active: false, repoId: "model", isEjecting: false } });
  worker.send({
    id: "loading",
    type: "event",
    event: { type: "done", text: "", stats: emptyStats(), finishReason: "Cancelled" },
  });
  worker.send({ id: "loading", type: "result" });
  worker.send({ id: cancel.id, type: "result" });
  await run;
  await vi.waitFor(() => expect(worker.requests.at(-1)?.type).toBe("eject"));
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await ejected;
  expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ finishReason: "Cancelled" }));
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it("waits for a cancelled asynchronous load to unwind before ejecting", async () => {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const run = runtime.transport.start("loading", { repoId: "model", messages: [] }, vi.fn());
  worker.send({ type: "loading", state: { status: "start", repoId: "model" } });
  const ejected = runtime.eject("model");
  const cancel = worker.requests.at(-1)!;
  expect(cancel.type).toBe("cancel");
  worker.send({ id: cancel.id, type: "result" });
  await Promise.resolve();
  await Promise.resolve();
  expect(worker.requests.some((request) => request.type === "eject")).toBe(false);
  worker.send({ type: "session", state: { active: false, repoId: "model", isEjecting: false } });
  worker.send({ id: "loading", type: "result" });
  await run;
  await vi.waitFor(() => expect(worker.requests.at(-1)?.type).toBe("eject"));
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await ejected;
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it("inspects model encodings without requiring a GPU or cross-origin isolation", async () => {
  vi.stubGlobal("crossOriginIsolated", false);
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const inspecting = runtime.inspectEncoding({ type: "hanashi", name: "qwen3.5" });
  expect(worker.requests[0]).toMatchObject({ type: "inspectEncoding" });
  const value = { supportsTools: true, reasoning: { kind: "toggle", defaultEffort: "default" } };
  worker.send({ id: worker.requests[0]!.id, type: "result", value });
  expect(await inspecting).toEqual(value);
  await expect(runtime.transport.start("run", { repoId: "model", messages: [] }, vi.fn())).rejects.toThrow(
    "cross-origin isolation",
  );
});

it("inspects config and releases a metadata-only worker without inference capabilities", async () => {
  vi.stubGlobal("crossOriginIsolated", false);
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const config = new TextEncoder().encode("{}");
  const inspecting = runtime.inspectModelConfig(config);
  expect(worker.requests[0]).toMatchObject({ type: "inspectModelConfig", config });
  worker.send({ id: worker.requests[0]!.id, type: "result", error: "Unsupported architecture" });
  await expect(inspecting).rejects.toThrow("Unsupported architecture");
  const ejected = runtime.eject("model");
  expect(worker.requests[1]).toMatchObject({ type: "eject" });
  worker.send({ id: worker.requests[1]!.id, type: "result" });
  await ejected;
  expect(worker.terminate).toHaveBeenCalled();
});

async function idleModel() {
  const worker = new FakeWorker();
  const runtime = new BrowserRuntime(() => worker as unknown as Worker);
  const run = runtime.transport.start("run", { repoId: "model", messages: [] }, vi.fn());
  worker.send({ type: "session", state: { active: true, repoId: "model" } });
  worker.send({ id: "run", type: "result" });
  await run;
  await vi.advanceTimersByTimeAsync(0);
  return { worker, runtime };
}

it("applies changed auto-eject intervals to the already idle model", async () => {
  vi.useFakeTimers();
  const { worker } = await idleModel();
  await vi.advanceTimersByTimeAsync(60_000);
  await webSettings.setAutoEjectMinutes(2);
  await vi.advanceTimersByTimeAsync(119_999);
  expect(worker.requests.at(-1)?.type).toBe("run");
  await vi.advanceTimersByTimeAsync(1);
  expect(worker.requests.at(-1)?.type).toBe("eject");
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await vi.advanceTimersByTimeAsync(0);
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it.each([false, true])("applies enabled/disabled auto-eject changes while idle (other tab: %s)", async (otherTab) => {
  vi.useFakeTimers();
  await webSettings.setAutoEjectMinutes(1);
  const { worker } = await idleModel();
  await webSettings.setAutoEjectEnabled(false);
  await vi.advanceTimersByTimeAsync(60_000);
  expect(worker.requests.at(-1)?.type).toBe("run");
  if (otherTab) {
    localStorage.setItem("mirai.web.settings", JSON.stringify({ autoEjectEnabled: true, autoEjectMinutes: 1 }));
    window.dispatchEvent(new StorageEvent("storage", { key: "mirai.web.settings" }));
  } else await webSettings.setAutoEjectEnabled(true);
  await vi.advanceTimersByTimeAsync(60_000);
  expect(worker.requests.at(-1)?.type).toBe("eject");
  worker.send({ id: worker.requests.at(-1)!.id, type: "result" });
  await vi.advanceTimersByTimeAsync(0);
  expect(worker.terminate).toHaveBeenCalledOnce();
});
