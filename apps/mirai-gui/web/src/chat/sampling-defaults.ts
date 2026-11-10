import type { SamplingDefaults } from "@/platform/services/chat";

type GenerationConfig = {
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  min_p?: number | null;
  repetition_penalty?: number | null;
  suffix_repetition_length?: number | null;
};

// uzu stores models in OPFS under the path it reports, one directory per segment.
async function readOpfsText(path: string): Promise<string> {
  const segments = path.split("/").filter(Boolean);
  const name = segments.pop();
  if (!name) throw new Error(`Not a file path: ${path}`);
  let directory = await navigator.storage.getDirectory();
  for (const segment of segments) directory = await directory.getDirectoryHandle(segment);
  return (await (await directory.getFileHandle(name)).getFile()).text();
}

/**
 * The browser counterpart of Engine::model_sampling_defaults, which the WASM
 * bindings do not export: uzu's generation config in the model's config.json.
 */
export async function readModelSamplingDefaults(modelPath: string): Promise<SamplingDefaults | null> {
  const config = JSON.parse(await readOpfsText(`${modelPath.replace(/\/+$/, "")}/config.json`)) as {
    generation_config?: GenerationConfig;
  };
  const generation = config.generation_config;
  if (!generation) return null;
  return {
    type: "Stochastic",
    temperature: generation.temperature ?? null,
    topK: generation.top_k ?? null,
    topP: generation.top_p ?? null,
    minP: generation.min_p ?? null,
    repetitionPenalty: generation.repetition_penalty ?? null,
    suffixRepetitionLength: generation.suffix_repetition_length ?? null,
  };
}
