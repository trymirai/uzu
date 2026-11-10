import { afterEach, expect, it, vi } from "vitest";
import { readModelSamplingDefaults } from "./sampling-defaults";

type Entry = string | { [name: string]: Entry };

const directory = (entries: Record<string, Entry>): unknown => ({
  getDirectoryHandle: async (name: string) => {
    const entry = entries[name];
    if (!entry || typeof entry === "string") throw new DOMException(name, "NotFoundError");
    return directory(entry);
  },
  getFileHandle: async (name: string) => {
    const entry = entries[name];
    if (typeof entry !== "string") throw new DOMException(name, "NotFoundError");
    return { getFile: async () => ({ text: async () => entry }) };
  },
});

const storeModel = (config: unknown) =>
  vi.stubGlobal("navigator", {
    storage: {
      getDirectory: async () =>
        directory({ mirai: { models: { "vendor-model": { "config.json": JSON.stringify(config) } } } }),
    },
  });

afterEach(() => vi.unstubAllGlobals());

it("reads uzu's generation config from the model directory in OPFS", async () => {
  storeModel({ generation_config: { stop_token_ids: [1], temperature: 0.7, top_p: 0.9, repetition_penalty: 1.1 } });
  expect(await readModelSamplingDefaults("/mirai/models/vendor-model/")).toEqual({
    type: "Stochastic",
    temperature: 0.7,
    topK: null,
    topP: 0.9,
    minP: null,
    repetitionPenalty: 1.1,
    suffixRepetitionLength: null,
  });
});

it("reports no defaults for a model without a generation config and fails for a missing model", async () => {
  storeModel({ model_type: "classifier" });
  expect(await readModelSamplingDefaults("/mirai/models/vendor-model")).toBeNull();
  await expect(readModelSamplingDefaults("/mirai/models/other")).rejects.toThrow();
});
