import type { LlmRunParams } from "@/types/llm-stream";
import type { BrowserEncodingSupport } from "./browser-protocol";
import type { ToolCall } from "./browser-tools";
import type { RunEvent } from "./run-stream";

export type WasmSession = {
  run(
    params: LlmRunParams & { maxOutputTokens?: number },
    onEvent: (event: RunEvent) => void,
    onTool?: (call: ToolCall) => unknown,
  ): Promise<void>;
  cancel(): void;
  dispose(): void;
};
export type WasmModule = {
  default(): Promise<unknown>;
  inspectEncoding(encoding: unknown): BrowserEncodingSupport;
  inspectModelConfig(config: Uint8Array): void;
  supportsBlobWeights(): boolean;
  createSession(options: {
    config: Uint8Array;
    tokenizer: Uint8Array;
    weights: Blob | Uint8Array;
    loadCancellation?: Int32Array;
    contextLength?: number;
    encoding?: unknown;
    seed?: number;
  }): Promise<WasmSession>;
};

const outdated = () =>
  new Error("The browser engine is out of date. Rebuild and deploy it with pnpm web:wasm, then reload the app.");

export async function initializeBrowserWasm(module: WasmModule): Promise<WasmModule> {
  if (
    ![
      module.default,
      module.createSession,
      module.inspectEncoding,
      module.inspectModelConfig,
      module.supportsBlobWeights,
    ].every((value) => typeof value === "function")
  )
    throw outdated();
  await module.default();
  // The marker is a WASM export and may only be called after initialization.
  if (!module.supportsBlobWeights()) throw outdated();
  return module;
}
