import { beforeEach, expect, it, vi } from "vitest";
import { tauriChat } from "./tauri";
import type { LlmRunParams } from "@/types/llm-stream";

const mocks = vi.hoisted(() => ({ invoke: vi.fn(async () => {}) }));
vi.mock("../shared/invoke", () => ({ invoke: mocks.invoke }));
vi.mock("@tauri-apps/api/core", () => ({ Channel: class {} }));

beforeEach(() => vi.clearAllMocks());

it.each([true, false])("sends chartToolEnabled=%s through the native run payload", async (chartToolEnabled) => {
  const params: LlmRunParams = {
    repoId: "model",
    messages: [{ role: "user", content: "Draw a chart" }],
    chartToolEnabled,
  };
  const run = tauriChat.runStream(params);
  expect(mocks.invoke).toHaveBeenCalledWith("run_stream", {
    payload: { runId: run.runId, ...params },
    onEvent: expect.any(Object),
  });
  await run.cancel();
});
