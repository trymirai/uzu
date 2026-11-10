import { afterEach, expect, it, vi } from "vitest";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import type { SessionLoadingPayload, SessionStatePayload } from "@/platform/services/session";
import type { SettingsService } from "@/platform/services/settings";
import type { LlmRunParams } from "@/types/llm-stream";
import type { UzuRuntime } from "../uzu-runtime";
import { BrowserChatRuntime, cheapestEffort, effectiveEffort } from "./browser-runtime";

class Role {}
class ChatRoleSystem extends Role {}
class ChatRoleUser extends Role {}
class ChatRoleAssistant extends Role {}
class ChatRoleTool extends Role {}

// Built input messages record their blocks as readable strings.
class ChatMessage {
  constructor(
    readonly role: Role,
    readonly blocks: string[] = [],
  ) {}
  static system = () => new ChatMessage(new ChatRoleSystem());
  static user = () => new ChatMessage(new ChatRoleUser());
  static assistant = () => new ChatMessage(new ChatRoleAssistant());
  withText = (value: string) => new ChatMessage(this.role, [...this.blocks, `text:${value}`]);
  withReasoning = (value: string) => new ChatMessage(this.role, [...this.blocks, `reasoning:${value}`]);
  withReasoningEffort = (value: string) => new ChatMessage(this.role, [...this.blocks, `effort:${value}`]);
}

const free = () => {};
class ChatContentBlockText {
  constructor(readonly value: string) {}
  free = free;
}
class ChatContentBlockReasoning {
  constructor(readonly value: string) {}
  free = free;
}
class ChatSessionStreamChunkReplies {
  constructor(readonly replies: unknown[]) {}
  free = free;
}
class ChatSessionStreamChunkError {
  constructor(readonly error: unknown) {}
  free = free;
}
class ChatSessionStreamChunkToolResults {
  constructor(readonly messages: unknown[]) {}
  free = free;
}
class ReplyConfig {
  constructor(readonly fields: Record<string, unknown> = {}) {}
  withSamplingPolicy = (policy: unknown) => new ReplyConfig({ ...this.fields, policy });
  withTokenLimit = (tokenLimit: number) => new ReplyConfig({ ...this.fields, tokenLimit });
}

const api = {
  ChatMessage,
  ChatRoleSystem,
  ChatRoleAssistant,
  ChatRoleTool,
  ChatContentBlockText,
  ChatContentBlockReasoning,
  ChatContentBlockToolCall: class {},
  ChatContentBlockToolCallResult: class {},
  ChatSessionStreamChunkReplies,
  ChatSessionStreamChunkError,
  ChatSessionStreamChunkToolResults,
  ChatReplyConfig: { create: () => new ReplyConfig() },
  ChatConfig: { create: () => ({}) },
  SamplingPolicyDefault: class {
    kind = "default";
  },
  SamplingPolicyCustom: class {
    constructor(readonly method: unknown) {}
  },
  SamplingMethodGreedy: class {},
  SamplingMethodStochastic: class {},
  NativeTool: class {
    constructor(readonly definition: { name: string }) {}
  },
  ToolFunction: class {
    constructor(readonly name: string) {}
  },
  Value: class {
    constructor(readonly json: string) {}
  },
};

const reply = (blocks: (ChatContentBlockText | ChatContentBlockReasoning)[], finishReason: string | null = null) => ({
  message: { role: new ChatRoleAssistant(), metadata: { values: {}, free }, content: blocks, free },
  finishReason,
  stats: {
    duration: 2,
    timeToFirstToken: 0.5,
    prefillTokensPerSecond: 20,
    generateTokensPerSecond: 10,
    tokensCountInput: 10,
    tokensCountOutput: 4,
    speculatorStats: null,
    free,
  },
  free,
});
const replies = (...values: ReturnType<typeof reply>[]) => new ChatSessionStreamChunkReplies(values);

function fixture({ downloaded = true, reasoning = "toggle" as "toggle" | "alwaysOn", tools = true } = {}) {
  let script: unknown[][] = [];
  const inputs: { messages: ChatMessage[]; config: ReplyConfig }[] = [];
  const tokens: { cancel: ReturnType<typeof vi.fn>; isCancelled: boolean }[] = [];
  let release: (() => void) | undefined;
  const session = {
    supportsToolCalls: Promise.resolve(tools),
    addTools: vi.fn<(tools: unknown[]) => Promise<void>>(async () => {}),
    reset: vi.fn(async () => {}),
    replyWithStream: vi.fn(async (messages: ChatMessage[], config: ReplyConfig) => {
      inputs.push({ messages, config });
      const chunks = script.shift() ?? [];
      const token = {
        isCancelled: false,
        cancel: vi.fn(() => {
          token.isCancelled = true;
          release?.();
        }),
      };
      tokens.push(token);
      return {
        cancelToken: token,
        // `null` in a script holds the stream open until it is cancelled.
        next: vi.fn(async () => {
          const chunk = chunks.shift();
          if (chunk === null)
            await new Promise<void>((resolve) => {
              release = resolve;
            });
          return chunk ?? null;
        }),
        free: vi.fn(),
      };
    }),
    free: vi.fn(),
  };
  const instance = { free: vi.fn() };
  const model = { identifier: "vendor:model", isDownloadable: true, properties: { size: 4e9, free }, free };
  const engine = {
    modelByRepoId: vi.fn(async (repoId: string) => (repoId === "vendor/model" ? { ...model } : null)),
    downloadState: vi.fn(async () => ({ name: downloaded ? "Downloaded" : "Paused", free })),
    chatInstance: vi.fn(async () => instance),
    chatWithInstance: vi.fn(async () => session),
    chatModelInfo: vi.fn(async () => ({
      capabilities: {
        supportsReasoning: true,
        supportsDisableReasoning: reasoning === "toggle",
        reasoningEfforts: reasoning === "toggle" ? ["Disabled", "Default"] : ["Default"],
        supportsTools: tools,
        free,
      },
      defaultReasoningEffort: "Default",
      free,
    })),
    modelPath: vi.fn(async () => null),
  };
  const settings = {
    getModelChatNamingEnabled: vi.fn(async () => true),
    getAutoEjectEnabled: vi.fn(async () => true),
    getAutoEjectMinutes: vi.fn(async () => 1),
  } as unknown as SettingsService;
  const runtime = new BrowserChatRuntime(
    async () => ({ api, engine }) as unknown as UzuRuntime,
    settings,
    () => null,
    () => () => {},
  );
  const states: SessionStatePayload[] = [];
  const loading: SessionLoadingPayload[] = [];
  runtime.onSessionState((state) => states.push(state));
  runtime.onSessionLoading((state) => loading.push(state));
  return {
    runtime,
    engine,
    session,
    instance,
    inputs,
    tokens,
    states,
    loading,
    script: (...runs: unknown[][]) => {
      script = runs;
    },
    async run(params: Partial<LlmRunParams> = {}, runId = "run") {
      const events: RunEvent[] = [];
      await runtime.transport.start(
        runId,
        { repoId: "vendor/model", messages: [{ role: "user", content: "Hello" }], ...params },
        (event) => events.push(event),
      );
      return events;
    },
  };
}

// The model has a different identifier in every test, so the capability cache never leaks.
let modelCounter = 0;
const uniqueModel = (source: ReturnType<typeof fixture>) => {
  const identifier = `vendor:model:${++modelCounter}`;
  source.engine.modelByRepoId.mockImplementation(async (repoId: string) =>
    repoId === "vendor/model" ? { identifier, isDownloadable: true, properties: { size: 4e9, free }, free } : null,
  );
};

afterEach(() => {
  vi.useRealTimers();
});

it("loads a model once, streams the reply and continues the resident session with only new messages", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script(
    [replies(reply([new ChatContentBlockText("Hi")])), replies(reply([new ChatContentBlockText("Hi there")], "Stop"))],
    [replies(reply([new ChatContentBlockText("Fine")], "Stop"))],
  );
  const first = await source.run({ dateTimeToolEnabled: false, chartToolEnabled: false });
  expect(source.loading).toEqual([{ status: "start", repoId: "vendor/model" }]);
  expect(source.states).toEqual([{ active: true, repoId: "vendor/model", isEjecting: false }]);
  expect(first.at(-1)).toMatchObject({
    type: "done",
    text: "Hi there",
    finishReason: "Stop",
    parsed: { response: "Hi there" },
    stats: { prefillStats: { duration: 0.5, tokensCount: 10 }, generateStats: { duration: 1.5, tokensCount: 4 } },
  });
  // The naming tool is on by default for a 4 GB model and asks for a name in the system prompt.
  expect(source.inputs[0]!.messages[0]!.blocks[0]).toMatch(/^text:Use set_chat_name/);
  expect(
    source.session.addTools.mock.calls[0]![0].map((tool) => (tool as { definition: { name: string } }).definition.name),
  ).toEqual(["set_chat_name"]);

  const second = await source.run({
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
    messages: [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
      { role: "user", content: "How are you?" },
    ],
  });
  expect(second.at(-1)).toMatchObject({ type: "done", text: "Fine" });
  expect(source.engine.chatInstance).toHaveBeenCalledOnce();
  expect(source.session.reset).toHaveBeenCalledOnce();
  expect(source.inputs[1]!.messages.map((message) => message.blocks)).toEqual([["text:How are you?"]]);
});

it("replays the whole conversation after an edit and passes the chosen reasoning effort", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script(
    [replies(reply([new ChatContentBlockText("A")], "Stop"))],
    [replies(reply([new ChatContentBlockText("B")], "Stop"))],
  );
  await source.run({ modelChatNamingEnabled: false, dateTimeToolEnabled: false, chartToolEnabled: false });
  await source.run({
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
    reasoningEffort: "disabled",
    messages: [
      { role: "system", content: "Be brief" },
      { role: "user", content: "Edited" },
    ],
  });
  expect(source.session.reset).toHaveBeenCalledTimes(2);
  expect(source.inputs[1]!.messages.map((message) => message.blocks)).toEqual([
    ["text:Be brief", "effort:Disabled"],
    ["text:Edited"],
  ]);
});

it("shows reasoning as the answer when reasoning was disabled but the model still reasoned", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script([replies(reply([new ChatContentBlockReasoning("Just the answer")], "Stop"))]);
  const events = await source.run({
    reasoningEffort: "disabled",
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
  });
  expect(events.at(-1)).toMatchObject({
    type: "done",
    text: "Just the answer",
    parsed: { response: "Just the answer", chainOfThought: "" },
  });
});

it("finishes a run stopped while its model was loading without generating", async () => {
  const source = fixture();
  uniqueModel(source);
  let loaded!: () => void;
  source.engine.chatInstance.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        loaded = () => resolve(source.instance);
      }),
  );
  const run = source.run();
  await vi.waitFor(() => expect(source.engine.chatInstance).toHaveBeenCalled());
  await source.runtime.transport.cancel("run");
  loaded();
  expect((await run).at(-1)).toMatchObject({ type: "done", text: "", finishReason: "Cancelled" });
  expect(source.session.replyWithStream).not.toHaveBeenCalled();
});

it("cancels a streaming reply through uzu's cancel token and does not reuse its history", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script(
    [replies(reply([new ChatContentBlockText("Partial")])), null],
    [replies(reply([new ChatContentBlockText("Again")], "Stop"))],
  );
  const run = source.run({ modelChatNamingEnabled: false });
  await vi.waitFor(() => expect(source.tokens).toHaveLength(1));
  await source.runtime.transport.cancel("run");
  expect(source.tokens[0]!.cancel).toHaveBeenCalledOnce();
  expect((await run).at(-1)).toMatchObject({ type: "done", text: "Partial", finishReason: "Cancelled" });
  await source.run({
    modelChatNamingEnabled: false,
    messages: [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Partial" },
      { role: "user", content: "Continue" },
    ],
  });
  expect(source.session.reset).toHaveBeenCalledTimes(2);
});

it("reports stream errors with a readable message and the transcript so far", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script([
    replies(reply([new ChatContentBlockText("Long")])),
    new ChatSessionStreamChunkError({ type: "Backend", error: { type: "ContextOverflow" } }),
  ]);
  vi.spyOn(console, "error").mockImplementation(() => {});
  const events = await source.run({ modelChatNamingEnabled: false });
  expect(events.at(-2)).toEqual({ type: "transcript", items: [{ type: "text", text: "Long" }] });
  expect(events.at(-1)).toEqual({ type: "error", error: "Conversation is too long for this model's context window." });
});

it("refuses to load a model that is not fully downloaded", async () => {
  const source = fixture({ downloaded: false });
  uniqueModel(source);
  vi.spyOn(console, "error").mockImplementation(() => {});
  expect(await source.run()).toEqual([{ type: "error", error: "Model is not downloaded" }]);
  expect(source.engine.chatInstance).not.toHaveBeenCalled();
});

it("generates a title with the cheapest reasoning and a small token budget", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script([replies(reply([new ChatContentBlockText('  "Trip   to Rome"  ')], "Stop"))]);
  expect(await source.runtime.generateTitle({ repoId: "vendor/model", userText: "Plan a trip" })).toBe("Trip to Rome");
  const { messages, config } = source.inputs[0]!;
  expect(messages[0]!.blocks).toEqual(["effort:Disabled"]);
  expect(messages[1]!.blocks[0]).toContain('Message: "Plan a trip"');
  expect(config.fields).toMatchObject({ tokenLimit: 40 });
});

it("skips title generation for a model that always reasons", async () => {
  const source = fixture({ reasoning: "alwaysOn" });
  uniqueModel(source);
  expect(await source.runtime.generateTitle({ repoId: "vendor/model", userText: "Plan a trip" })).toBe("");
  expect(source.session.replyWithStream).not.toHaveBeenCalled();
});

it("ejects the resident model and loads it again for the next run", async () => {
  const source = fixture();
  uniqueModel(source);
  source.script(
    [replies(reply([new ChatContentBlockText("A")], "Stop"))],
    [replies(reply([new ChatContentBlockText("B")], "Stop"))],
  );
  await source.run({ modelChatNamingEnabled: false });
  await source.runtime.eject("vendor/model");
  expect(source.states.slice(1)).toEqual([
    { active: false, repoId: "vendor/model", isEjecting: true },
    { active: false, repoId: "vendor/model", isEjecting: false },
  ]);
  expect(source.session.free).toHaveBeenCalledOnce();
  expect(source.instance.free).toHaveBeenCalledOnce();
  await source.run({ modelChatNamingEnabled: false });
  expect(source.engine.chatInstance).toHaveBeenCalledTimes(2);
});

it.each(["chatInstance", "chatWithInstance"] as const)(
  "publishes eviction before a model switch that fails in %s",
  async (failurePoint) => {
    const source = fixture();
    uniqueModel(source);
    await source.run({ modelChatNamingEnabled: false });
    source.engine.modelByRepoId.mockResolvedValueOnce({
      identifier: `vendor:other:${++modelCounter}`,
      isDownloadable: true,
      properties: { size: 4e9, free },
      free,
    });
    source.engine[failurePoint].mockImplementationOnce(async () => {
      expect(source.states.at(-1)).toEqual({ active: false, repoId: "vendor/model", isEjecting: false });
      expect(source.session.free).toHaveBeenCalledOnce();
      expect(source.instance.free).toHaveBeenCalledOnce();
      throw new Error("Load failed");
    });
    vi.spyOn(console, "error").mockImplementation(() => {});

    expect(await source.run({ repoId: "vendor/other" })).toEqual([{ type: "error", error: "Load failed" }]);
    expect(source.states).toEqual([
      { active: true, repoId: "vendor/model", isEjecting: false },
      { active: false, repoId: "vendor/model", isEjecting: false },
    ]);
    expect(source.loading.slice(1)).toEqual([
      { status: "start", repoId: "vendor/other" },
      { status: "error", repoId: "vendor/other" },
    ]);
  },
);

it("ejects an idle model after the configured interval", async () => {
  vi.useFakeTimers();
  const source = fixture();
  uniqueModel(source);
  source.script([replies(reply([new ChatContentBlockText("A")], "Stop"))]);
  await source.run({ modelChatNamingEnabled: false });
  await vi.advanceTimersByTimeAsync(59_000);
  expect(source.session.free).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1_000);
  expect(source.session.free).toHaveBeenCalledOnce();
  expect(source.states.at(-1)).toEqual({ active: false, repoId: "vendor/model", isEjecting: false });
});

it("keeps the requested reasoning effort only when the model offers that choice", () => {
  expect(effectiveEffort({ kind: "toggle", defaultEffort: "default" }, "disabled")).toBe("disabled");
  expect(effectiveEffort({ kind: "toggle", defaultEffort: "default" }, "high")).toBeUndefined();
  expect(effectiveEffort({ kind: "levels", efforts: ["low", "high"] }, "high")).toBe("high");
  expect(effectiveEffort({ kind: "levels", efforts: ["low", "high"] }, "default")).toBeUndefined();
  expect(effectiveEffort({ kind: "alwaysOn" }, "disabled")).toBeUndefined();
  expect(cheapestEffort({ kind: "levels", efforts: ["low", "high"] })).toBe("low");
  expect(cheapestEffort({ kind: "unsupported" })).toBeUndefined();
});
