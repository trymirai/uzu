import type { ChatService } from ".";
import { runLlmStream } from "./run-stream";
import { browserRuntime } from "./browser-runtime";
import { findBrowserModel } from "../models/browser-models";
import { modelFileStatus, readModelFile } from "../models/browser-model-files";
import type { SamplingDefaults } from ".";

export const webChat: ChatService = {
  runStream: (params) => runLlmStream(browserRuntime.transport, params),
  cancelRun: (runId) => browserRuntime.transport.cancel(runId),
  generateTitle: (params) => browserRuntime.generateTitle(params),
  cancelTitleGen: () => browserRuntime.cancelTitleGen(),
  async getSamplingDefaults(repoId) {
    const model = await findBrowserModel(repoId);
    const file = model.files.find((file) => file.name === "config.json");
    if (!file) return null;
    if (!(await modelFileStatus(file)).complete) return null;
    const config = JSON.parse(new TextDecoder().decode(await readModelFile(file))) as {
      generation_config?: {
        temperature?: number;
        top_k?: number;
        top_p?: number;
        min_p?: number;
        repetition_penalty?: number;
        suffix_repetition_length?: number;
      };
    };
    const sampling = config.generation_config;
    if (!sampling) return null;
    if (sampling.temperature != null && sampling.temperature <= 0) return { type: "Greedy" };
    return {
      type: "Stochastic",
      temperature: sampling.temperature,
      topK: sampling.top_k,
      topP: sampling.top_p,
      minP: sampling.min_p,
      repetitionPenalty: sampling.repetition_penalty,
      suffixRepetitionLength: sampling.suffix_repetition_length,
    } satisfies SamplingDefaults;
  },
};
