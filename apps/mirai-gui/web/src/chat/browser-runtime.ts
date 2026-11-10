import type { CancelToken, ChatInstance, ChatSession, Engine, Model } from "@uzu/wasm";
import type { SamplingDefaults, TitleGenParams } from "@/platform/services/chat";
import type { RunEvent, RunTransport } from "@/platform/services/chat/run-stream";
import { emptyStats } from "@/platform/services/chat/empty-stats";
import type { SessionLoadingPayload, SessionStatePayload } from "@/platform/services/session";
import type { SettingsService } from "@/platform/services/settings";
import { onBrowserAutoEjectChanged, webSettings } from "@/platform/services/settings/web";
import type { LlmRunParams } from "@/types/llm-stream";
import type { ReasoningEffort, ReasoningSupport } from "@/types/sampling";
import { browserInferenceError } from "@/utils/browser-inference";
import { errorText } from "@/utils/error-text";
import { getUzuRuntime, type UzuBindings, type UzuRuntime } from "../uzu-runtime";
import { chatSupport } from "../models/chat-support";
import { ReplyRun } from "./reply-run";
import { readModelSamplingDefaults } from "./sampling-defaults";
import { sameHistoryMessage, sanitize, type HistoryMessage } from "./text";
import { ChatNaming, NAMING_INSTRUCTION, nativeTools, normalizeName, titlePrompt, toolSpecs } from "./tools";
import type { Reply } from "./transcript";
import {
  addNamingInstruction,
  buildMessages,
  buildTailMessages,
  chatErrorMessage,
  readMessage,
  readReply,
  samplingPolicy,
} from "./uzu-values";

const MIN_IDLE_MS = 60_000;
const MAX_IDLE_MS = 24 * 60 * 60_000;
const TITLE_TOKEN_LIMIT = 40;
const TITLE_RUN_ID = "title-gen";

type ToolChoice = { naming: boolean; dateTime: boolean; chart: boolean };

type Resident = ToolChoice & {
  repoId: string;
  identifier: string;
  instance: ChatInstance;
  session: ChatSession;
  supportsTools: boolean;
  namingTool: ChatNaming | null;
  reasoning: ReasoningSupport;
};

// The client-visible prefix the resident session holds. The session also keeps
// internal tool calls and results that are absent from this projection.
type History = { repoId: string; messages: HistoryMessage[]; effort?: ReasoningEffort };

// A run is registered before its model loads: a cold load takes a while, and a
// Stop that arrives in that window has no stream token yet.
type RunCancel = { requested: boolean; token?: CancelToken };

export function effectiveEffort(support: ReasoningSupport, requested?: ReasoningEffort): ReasoningEffort | undefined {
  if (support.kind === "toggle") return requested === "disabled" ? requested : undefined;
  if (support.kind === "levels")
    // Leave default selection to the template, including mappings whose
    // default is not named as an explicit effort level.
    return requested && requested !== "default" && support.efforts.includes(requested) ? requested : undefined;
  return undefined;
}

export function cheapestEffort(support: ReasoningSupport): ReasoningEffort | undefined {
  if (support.kind === "toggle") return "disabled";
  if (support.kind === "levels") return support.efforts[0];
  return undefined;
}

// Without an explicit choice, tools stay off for small models that misuse them.
const toolEnabled = (requested: boolean | undefined, modelSize: number | undefined) =>
  requested ?? (modelSize === undefined || modelSize >= 2_000_000_000);

/**
 * The browser counterpart of desktop/src/chat: one resident chat session over
 * uzu's WASM engine, kept between runs and ejected after an idle period.
 */
export class BrowserChatRuntime {
  private resident?: Resident;
  private history?: History;
  private runQueue: Promise<unknown> = Promise.resolve();
  private readonly cancels = new Map<string, RunCancel>();
  private idleTimer?: ReturnType<typeof setTimeout>;
  private idleGeneration = 0;
  private stopSettingsListener?: () => void;
  private readonly stateListeners = new Set<(state: SessionStatePayload) => void>();
  private readonly loadingListeners = new Set<(state: SessionLoadingPayload) => void>();

  constructor(
    private readonly loadRuntime: () => Promise<UzuRuntime> = getUzuRuntime,
    private readonly settings: SettingsService = webSettings,
    private readonly unavailableReason: () => string | null = browserInferenceError,
    private readonly onSettingsChanged: (listener: () => void) => () => void = onBrowserAutoEjectChanged,
  ) {}

  readonly transport: RunTransport = {
    start: (runId, params, onEvent) => this.run(runId, params, onEvent),
    cancel: async (runId) => this.requestCancel(runId),
  };

  // ---- runs ---------------------------------------------------------------

  private async run(runId: string, params: LlmRunParams, onEvent: (event: RunEvent) => void) {
    this.cancelIdleTimer();
    this.cancels.set(runId, { requested: false });
    try {
      await this.exclusive(() => this.runInner(runId, params, onEvent));
    } catch (error) {
      console.error("[chat] run failed", { runId, repoId: params.repoId }, error);
      onEvent({ type: "error", error: errorText(error) });
    } finally {
      this.cancels.delete(runId);
      if (this.resident) this.armIdleTimer();
    }
  }

  private async runInner(runId: string, params: LlmRunParams, onEvent: (event: RunEvent) => void) {
    const { api } = await this.loadRuntime();
    const resident = await this.ensureSession(params.repoId, {
      naming: params.modelChatNamingEnabled,
      dateTime: params.dateTimeToolEnabled,
      chart: params.chartToolEnabled,
    });
    if (this.cancels.get(runId)?.requested) {
      onEvent({ type: "done", text: "", stats: emptyStats(), finishReason: "Cancelled", transcript: [] });
      return;
    }
    const effort = effectiveEffort(resident.reasoning, params.reasoningEffort);
    const tail = this.continuation(params.repoId, params.messages, effort);
    let messages;
    if (tail) messages = buildTailMessages(api, tail);
    else {
      // Full replay requires resetting uzu's accumulated role validator.
      await this.resetSession(resident);
      messages = buildMessages(api, params.messages, effort);
      if (resident.namingTool) addNamingInstruction(api, messages, NAMING_INSTRUCTION);
    }
    this.history = undefined;

    const naming = resident.namingTool?.begin(onEvent);
    const config = api.ChatReplyConfig.create().withSamplingPolicy(samplingPolicy(api, params.samplingPolicy));
    const toolsEnabled = resident.supportsTools && (resident.naming || resident.dateTime || resident.chart);
    const reply = new ReplyRun(onEvent, effort === "disabled", toolsEnabled);
    try {
      const stream = await resident.session.replyWithStream(messages, config);
      const token = stream.cancelToken;
      try {
        this.attachCancelToken(runId, token);
        await this.consume(api, stream, {
          replies: (value) => reply.reply(value),
          toolResults: (values) => reply.toolResults(values),
          error: (message) => {
            reply.fail();
            throw new Error(message);
          },
        });
      } finally {
        stream.free();
      }
      const remembered = reply.finish(token.isCancelled, naming?.name());
      if (remembered)
        this.history = {
          repoId: params.repoId,
          messages: [...params.messages, remembered],
          ...(effort ? { effort } : {}),
        };
    } finally {
      naming?.end();
    }
  }

  private async consume(
    api: UzuBindings,
    stream: Awaited<ReturnType<ChatSession["replyWithStream"]>>,
    on: { replies(reply: Reply): void; toolResults(messages: Reply["message"][]): void; error(message: string): void },
  ) {
    for (let chunk = await stream.next(); chunk; chunk = await stream.next()) {
      try {
        if (chunk instanceof api.ChatSessionStreamChunkReplies) {
          const [first, ...rest] = chunk.replies;
          for (const extra of rest) extra.free();
          if (first) on.replies(readReply(api, first));
        } else if (chunk instanceof api.ChatSessionStreamChunkToolResults) {
          on.toolResults(
            chunk.messages.map((message) => {
              try {
                return readMessage(api, message);
              } finally {
                message.free();
              }
            }),
          );
        } else if (chunk instanceof api.ChatSessionStreamChunkError) {
          on.error(chatErrorMessage(chunk.error));
        }
      } finally {
        (chunk as { free?: () => void }).free?.();
      }
    }
  }

  // Continue the resident session when the request extends what it already holds.
  private continuation(repoId: string, messages: HistoryMessage[], effort?: ReasoningEffort) {
    const history = this.history;
    if (
      history &&
      history.repoId === repoId &&
      history.effort === effort &&
      history.messages.length > 0 &&
      messages.length > history.messages.length &&
      history.messages.every((message, index) => sameHistoryMessage(message, messages[index]!))
    )
      return messages.slice(history.messages.length);
    return undefined;
  }

  async generateTitle(params: TitleGenParams): Promise<string> {
    this.cancelIdleTimer();
    this.cancels.set(TITLE_RUN_ID, { requested: false });
    try {
      return await this.exclusive(async () => {
        const { api } = await this.loadRuntime();
        const resident = await this.ensureSession(params.repoId, { naming: false, dateTime: false, chart: false });
        // Such a model spends the whole budget on reasoning; a bigger budget would delay the reply.
        if (this.cancels.get(TITLE_RUN_ID)?.requested || resident.reasoning.kind === "alwaysOn") return "";
        const messages = buildMessages(
          api,
          [{ role: "user", content: titlePrompt(params.userText) }],
          cheapestEffort(resident.reasoning),
        );
        const config = api.ChatReplyConfig.create()
          .withTokenLimit(TITLE_TOKEN_LIMIT)
          .withSamplingPolicy(new api.SamplingPolicyDefault());
        // reset() invalidates the KV prefix tracked for chat runs.
        this.history = undefined;
        await this.resetSession(resident);
        let text = "";
        const stream = await resident.session.replyWithStream(messages, config);
        try {
          this.attachCancelToken(TITLE_RUN_ID, stream.cancelToken);
          await this.consume(api, stream, {
            replies: (reply) => {
              text = reply.message.content.map((block) => (block.type === "text" ? block.value : "")).join("");
            },
            toolResults: () => {},
            error: (message) => {
              throw new Error(message);
            },
          });
        } finally {
          stream.free();
        }
        if (this.cancels.get(TITLE_RUN_ID)?.requested) return "";
        try {
          return normalizeName(sanitize(text.trim()));
        } catch {
          return "";
        }
      });
    } finally {
      this.cancels.delete(TITLE_RUN_ID);
      if (this.resident) this.armIdleTimer();
    }
  }

  cancelTitleGen(): Promise<void> {
    this.requestCancel(TITLE_RUN_ID);
    return Promise.resolve();
  }

  private requestCancel(runId: string) {
    const run = this.cancels.get(runId);
    if (!run) return;
    run.requested = true;
    run.token?.cancel();
  }

  private attachCancelToken(runId: string, token: CancelToken) {
    const run = this.cancels.get(runId);
    if (!run) return;
    if (run.requested) token.cancel();
    run.token = token;
  }

  // Runs, titles and ejection share one session: a concurrent reset() would
  // clobber another stream.
  private exclusive<T>(task: () => Promise<T>): Promise<T> {
    const result = this.runQueue.then(task, task);
    this.runQueue = result.catch(() => undefined);
    return result;
  }

  // ---- session ------------------------------------------------------------

  private async ensureSession(
    repoId: string,
    requested: { naming?: boolean; dateTime?: boolean; chart?: boolean },
  ): Promise<Resident> {
    const unavailable = this.unavailableReason();
    if (unavailable) throw new Error(unavailable);
    const { api, engine } = await this.loadRuntime();
    const model = await engine.modelByRepoId(repoId);
    if (!model) throw new Error(`Model not found: ${repoId}`);
    try {
      const size = this.modelSize(model);
      const tools: ToolChoice = {
        naming:
          toolEnabled(requested.naming, size) && (await this.settings.getModelChatNamingEnabled().catch(() => true)),
        dateTime: toolEnabled(requested.dateTime, size),
        chart: toolEnabled(requested.chart, size),
      };
      await this.requireDownloaded(engine, model);
      const resident = this.resident;
      if (resident?.identifier === model.identifier) {
        if (resident.naming === tools.naming && resident.dateTime === tools.dateTime && resident.chart === tools.chart)
          return resident;
        // Keep the loaded weights in the instance; replace only the session and its tools.
        this.resident = undefined;
        this.history = undefined;
        resident.session.free();
        try {
          this.resident = {
            ...resident,
            ...tools,
            ...(await this.createSession(api, engine, resident.instance, tools)),
          };
        } catch (error) {
          resident.instance.free();
          this.emitState({ active: false, repoId, isEjecting: false });
          throw error;
        }
        return this.resident;
      }
      this.dropResident();
      const support = await chatSupport(engine, model);
      // Loading events only on the real load path: emitting them on session
      // reuse would flash the spinner and the resident-model chip.
      this.emitLoading({ status: "start", repoId });
      let instance: ChatInstance | undefined;
      try {
        instance = await engine.chatInstance(model, api.ChatConfig.create());
        this.resident = {
          repoId,
          identifier: model.identifier,
          instance,
          reasoning: support.reasoning,
          ...tools,
          ...(await this.createSession(api, engine, instance, tools)),
        };
      } catch (error) {
        instance?.free();
        this.emitLoading({ status: "error", repoId });
        throw error;
      }
      this.emitState({ active: true, repoId, isEjecting: false });
      return this.resident;
    } finally {
      model.free();
    }
  }

  private async createSession(api: UzuBindings, engine: Engine, instance: ChatInstance, tools: ToolChoice) {
    const session = await engine.chatWithInstance(instance);
    try {
      const supportsTools = await session.supportsToolCalls;
      const namingTool = tools.naming && supportsTools ? new ChatNaming() : null;
      const specs = toolSpecs({
        dateTime: tools.dateTime && supportsTools,
        chart: tools.chart && supportsTools,
        naming: namingTool,
      });
      if (specs.length > 0) await session.addTools(nativeTools(api, specs));
      return { session, supportsTools, namingTool };
    } catch (error) {
      session.free();
      throw error;
    }
  }

  private async resetSession(resident: Resident) {
    try {
      await resident.session.reset();
    } catch (error) {
      this.evict(resident.repoId);
      throw error;
    }
  }

  private modelSize(model: Model): number | undefined {
    const properties = model.properties;
    try {
      return properties?.size ?? undefined;
    } finally {
      properties?.free();
    }
  }

  private async requireDownloaded(engine: Engine, model: Model) {
    if (!model.isDownloadable) return;
    const state = await engine.downloadState(model);
    try {
      if (state?.name !== "Downloaded") throw new Error("Model is not downloaded");
    } finally {
      state?.free();
    }
  }

  private dropResident() {
    const resident = this.resident;
    this.resident = undefined;
    this.history = undefined;
    resident?.session.free();
    resident?.instance.free();
    if (resident) this.emitState({ active: false, repoId: resident.repoId, isEjecting: false });
  }

  private evict(repoId: string): boolean {
    const hadSession = this.resident !== undefined;
    if (hadSession) this.emitState({ active: false, repoId, isEjecting: true });
    this.dropResident();
    return hadSession;
  }

  async eject(repoId: string): Promise<void> {
    this.cancelIdleTimer();
    // Freeing the session under a stream would break it: stop running replies first.
    for (const runId of this.cancels.keys()) this.requestCancel(runId);
    await this.exclusive(async () => {
      if (!this.evict(repoId)) this.emitState({ active: false, repoId, isEjecting: false });
    });
  }

  async samplingDefaults(repoId: string): Promise<SamplingDefaults | null> {
    const { engine } = await this.loadRuntime();
    const model = await engine.modelByRepoId(repoId);
    if (!model) return null;
    try {
      const path = await engine.modelPath(model);
      return path ? await readModelSamplingDefaults(path) : null;
    } catch (error) {
      console.warn("[chat] sampling defaults unavailable", { repoId }, error);
      return null;
    } finally {
      model.free();
    }
  }

  // ---- auto-eject ---------------------------------------------------------

  private cancelIdleTimer() {
    clearTimeout(this.idleTimer);
    this.idleTimer = undefined;
    this.idleGeneration += 1;
  }

  private armIdleTimer() {
    this.cancelIdleTimer();
    this.stopSettingsListener ??= this.onSettingsChanged(() => {
      // An idle resident session gets the new timer now; a run re-arms it when it finishes.
      if (this.resident && this.cancels.size === 0) this.armIdleTimer();
    });
    const generation = this.idleGeneration;
    void Promise.all([this.settings.getAutoEjectEnabled(), this.settings.getAutoEjectMinutes()])
      .then(([enabled, minutes]) => {
        const repoId = this.resident?.repoId;
        if (!enabled || !repoId || generation !== this.idleGeneration) return;
        const idleMs = Math.min(MAX_IDLE_MS, Math.max(MIN_IDLE_MS, minutes * 60_000));
        this.idleTimer = setTimeout(() => {
          if (generation === this.idleGeneration) void this.exclusive(async () => void this.evict(repoId));
        }, idleMs);
      })
      .catch((error: unknown) => console.warn("[chat] auto-eject settings unavailable", error));
  }

  // ---- events -------------------------------------------------------------

  private emitState(state: SessionStatePayload) {
    this.stateListeners.forEach((listener) => notify(listener, state));
  }

  private emitLoading(state: SessionLoadingPayload) {
    this.loadingListeners.forEach((listener) => notify(listener, state));
  }

  onSessionState(listener: (state: SessionStatePayload) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  onSessionLoading(listener: (state: SessionLoadingPayload) => void): () => void {
    this.loadingListeners.add(listener);
    return () => this.loadingListeners.delete(listener);
  }
}

function notify<T>(listener: (payload: T) => void, payload: T) {
  try {
    listener(payload);
  } catch (error) {
    // The other listeners still get the event.
    console.error("[session] listener failed", error);
  }
}
