import type { Model, ThinkingLevelMap } from "@earendil-works/pi-ai";
import {
  DownloadPhaseDownloaded,
  ReasoningEffort,
  type ChatModelInfo,
  type Engine,
  type Model as UzuModel,
} from "@trymirai/uzu";
import type { ModelEntry } from "./stream.js";

export const providerId = "uzu";
export const api = "uzu-local";

export function modelDefinition(
  model: UzuModel,
  info: ChatModelInfo,
  downloaded: boolean,
): Model<typeof api> {
  // Zero denotes an unknown limit; downloaded models without one are unbounded.
  const contextWindow = info.contextLength ?? (downloaded ? Infinity : 0);
  const thinkingLevelMap: ThinkingLevelMap = { minimal: null, max: null };
  const efforts = info.capabilities.reasoningEfforts;
  const fallback = info.defaultReasoningEffort ?? ReasoningEffort.Default;
  const hasLevels = efforts.some((effort) =>
    [
      ReasoningEffort.Low,
      ReasoningEffort.Medium,
      ReasoningEffort.High,
      ReasoningEffort.XHigh,
    ].includes(effort),
  );
  const levels = {
    low: ReasoningEffort.Low,
    medium: ReasoningEffort.Medium,
    high: ReasoningEffort.High,
    xhigh: ReasoningEffort.XHigh,
  } as const;
  thinkingLevelMap.off =
    info.capabilities.supportsDisableReasoning ||
    !info.capabilities.supportsReasoning
      ? ReasoningEffort.Disabled
      : null;
  for (const [level, effort] of Object.entries(levels)) {
    thinkingLevelMap[level as keyof typeof levels] = !info.capabilities
      .supportsReasoning
      ? null
      : efforts.includes(effort)
        ? effort
        : !hasLevels && level === "medium"
          ? fallback
          : null;
  }
  return {
    id: model.repoIds[0] ?? model.identifier,
    name: downloaded ? model.name : `${model.name} (not downloaded)`,
    api,
    provider: providerId,
    baseUrl: "",
    input: ["text"],
    reasoning: info.capabilities.supportsReasoning,
    thinkingLevelMap,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: info.contextLength ?? 0,
  };
}

export async function discoverModels(
  engine: Pick<Engine, "modelsOnDevice" | "chatModelInfo" | "downloadStates">,
  signal?: AbortSignal,
) {
  const entries = new Map<string, ModelEntry>();
  const errors: string[] = [];
  const states = await engine.downloadStates;
  for (const model of await engine.modelsOnDevice) {
    signal?.throwIfAborted();
    if (!model.isChatCapable) continue;
    const size = model.properties?.size;
    if (size !== undefined && size < 9_000_000_000) continue;
    try {
      const info = await engine.chatModelInfo(model);
      if (info?.capabilities.supportsTools) {
        const state = states[model.identifier];
        const downloaded =
          !!model.filesystemPath ||
          state?.phase instanceof DownloadPhaseDownloaded;
        const definition = modelDefinition(model, info, downloaded);
        if (!downloaded && state)
          definition.name = `${model.name} (${state.name.toLowerCase()})`;
        entries.set(definition.id, {
          model,
          info,
          definition,
        });
      }
    } catch (error) {
      errors.push(
        `${model.name}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  signal?.throwIfAborted();
  return { entries, errors };
}
