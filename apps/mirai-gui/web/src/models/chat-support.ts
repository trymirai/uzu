import type { Engine, Model } from "@uzu/wasm";
import type { ReasoningEffort, ReasoningSupport } from "@/types/sampling";

export type ChatSupport = { reasoning: ReasoningSupport; supportsTools: boolean };

type LevelEffort = Exclude<ReasoningEffort, "default">;

export const UNSUPPORTED_CHAT: ChatSupport = { reasoning: { kind: "unsupported" }, supportsTools: false };

// uzu lists efforts in template-mapping order, which differs per model.
const EFFORT_ORDER: LevelEffort[] = ["disabled", "low", "medium", "high", "xhigh"];

// uzu names efforts `Disabled`, `Default`, `Low`, … `XHigh`; the client uses lowercase.
export const fromUzuEffort = (effort: string | null | undefined): ReasoningEffort | undefined =>
  effort ? (effort.toLowerCase() as ReasoningEffort) : undefined;

type Capabilities = {
  supportsReasoning: boolean;
  supportsDisableReasoning: boolean;
  reasoningEfforts: string[];
  supportsTools: boolean;
};

// Mirrors classify_reasoning_support in desktop/src/models.rs.
export function classifyReasoningSupport(
  capabilities: Capabilities,
  defaultEffort?: ReasoningEffort,
): ReasoningSupport {
  const efforts = capabilities.reasoningEfforts.map(fromUzuEffort);
  const levels = EFFORT_ORDER.filter((effort) => efforts.includes(effort));
  if (!capabilities.supportsReasoning) return { kind: "unsupported" };
  if (levels.some((effort) => effort !== "disabled")) {
    const preferred = levels.find((effort) => effort === defaultEffort);
    return { kind: "levels", efforts: levels, ...(preferred ? { defaultEffort: preferred } : {}) };
  }
  if (capabilities.supportsDisableReasoning) {
    // Identical or unresolved modes do not prove an enabled choice.
    if (defaultEffort !== "default" || !efforts.includes("default")) {
      return { kind: "levels", efforts: ["disabled"], ...(defaultEffort === "disabled" ? { defaultEffort } : {}) };
    }
    return { kind: "toggle", defaultEffort: "default" };
  }
  return { kind: "alwaysOn" };
}

const cache = new Map<string, ChatSupport>();

export const knownChatSupport = (identifier: string): ChatSupport | undefined => cache.get(identifier);

/** Reads chat capabilities from the catalog's encoding, without the model's weights. */
export async function chatSupport(engine: Engine, model: Model): Promise<ChatSupport> {
  const cached = cache.get(model.identifier);
  if (cached) return cached;
  let support = UNSUPPORTED_CHAT;
  try {
    const info = await engine.chatModelInfo(model);
    if (info) {
      const capabilities = info.capabilities;
      try {
        support = {
          reasoning: classifyReasoningSupport(capabilities, fromUzuEffort(info.defaultReasoningEffort)),
          supportsTools: capabilities.supportsTools,
        };
      } finally {
        capabilities.free();
        info.free();
      }
    }
  } catch (cause) {
    // As on desktop, a model without readable capabilities still chats, without tools or reasoning controls.
    console.warn("[models] chat capabilities unavailable", { identifier: model.identifier }, cause);
  }
  cache.set(model.identifier, support);
  return support;
}
