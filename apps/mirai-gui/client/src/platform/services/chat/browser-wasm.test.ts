import { expect, it, vi } from "vitest";
import { initializeBrowserWasm, type WasmModule } from "./browser-wasm";

const moduleWith = (overrides: Partial<WasmModule> = {}): WasmModule => ({
  default: vi.fn(async () => undefined),
  createSession: vi.fn(),
  inspectEncoding: vi.fn(),
  inspectModelConfig: vi.fn(),
  supportsBlobWeights: () => true,
  ...overrides,
});

it("checks the Blob capability only after WASM has initialized", async () => {
  let initialized = false;
  const module = moduleWith({
    default: async () => {
      initialized = true;
    },
    supportsBlobWeights: () => {
      if (!initialized) throw new Error("WASM has not initialized");
      return true;
    },
  });
  await expect(initializeBrowserWasm(module)).resolves.toBe(module);
});
it("reports stale engines before loading model data", async () => {
  const module = moduleWith({ supportsBlobWeights: undefined });
  await expect(initializeBrowserWasm(module)).rejects.toThrow("pnpm web:wasm");
  expect(module.default).not.toHaveBeenCalled();
  await expect(initializeBrowserWasm(moduleWith({ supportsBlobWeights: () => false }))).rejects.toThrow("out of date");
});
it("preserves an initialization failure for the caller to report and retry", async () => {
  const fail = new Error("WASM response was truncated");
  const capability = vi.fn(() => true);
  await expect(
    initializeBrowserWasm(
      moduleWith({
        default: async () => {
          throw fail;
        },
        supportsBlobWeights: capability,
      }),
    ),
  ).rejects.toBe(fail);
  expect(capability).not.toHaveBeenCalled();
});
