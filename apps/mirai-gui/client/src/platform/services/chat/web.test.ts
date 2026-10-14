import { beforeEach, expect, it, vi } from "vitest";
import { webChat } from "./web";
import { findBrowserModel } from "../models/browser-models";
import { modelFileStatus, readModelFile } from "../models/browser-model-files";

vi.mock("../models/browser-models", () => ({ findBrowserModel: vi.fn() }));
vi.mock("../models/browser-model-files", () => ({ modelFileStatus: vi.fn(), readModelFile: vi.fn() }));
const config = { name: "config.json", url: "https://example.com/config.json", sizeBytes: 100, hashes: [] };
const model: Awaited<ReturnType<typeof findBrowserModel>> = {
  identifier: "model",
  name: "Model",
  vendor: "Vendor",
  reasoning: { kind: "unsupported" },
  supportsTools: false,
  files: [config],
};
beforeEach(() => vi.resetAllMocks());

it("reads sampling defaults from completed config chunks without loading model weights", async () => {
  vi.mocked(findBrowserModel).mockResolvedValue(model);
  vi.mocked(modelFileStatus).mockResolvedValue({ complete: true } as Awaited<ReturnType<typeof modelFileStatus>>);
  vi.mocked(readModelFile).mockResolvedValue(
    new TextEncoder().encode(
      JSON.stringify({
        generation_config: {
          temperature: 0.6,
          top_k: 20,
          top_p: 0.95,
          min_p: 0.1,
          repetition_penalty: 1.1,
          suffix_repetition_length: 1024,
        },
      }),
    ),
  );
  expect(await webChat.getSamplingDefaults("model")).toEqual({
    type: "Stochastic",
    temperature: 0.6,
    topK: 20,
    topP: 0.95,
    minP: 0.1,
    repetitionPenalty: 1.1,
    suffixRepetitionLength: 1024,
  });
  expect(readModelFile).toHaveBeenCalledExactlyOnceWith(config);
});

it("does not read a partially downloaded config", async () => {
  vi.mocked(findBrowserModel).mockResolvedValue(model);
  vi.mocked(modelFileStatus).mockResolvedValue({ complete: false } as Awaited<ReturnType<typeof modelFileStatus>>);
  expect(await webChat.getSamplingDefaults("model")).toBeNull();
  expect(readModelFile).not.toHaveBeenCalled();
});
