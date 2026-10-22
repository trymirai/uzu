import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clampThinkingLevel,
  createAssistantMessageEventStream,
  getSupportedThinkingLevels,
  isContextOverflow,
  normalizeContext,
  Type,
  type AssistantMessageEvent,
  type ModelThinkingLevel,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import {
  CancelToken,
  ChatConfig,
  ChatContentBlockToolCallCandidate,
  ChatMessage,
  ChatModelCapabilities,
  ChatModelInfo,
  ChatReply,
  ChatReplyFinishReason,
  ChatReplyStats,
  ChatReplyConfig,
  ChatRoleDeveloper,
  ChatRoleSystem,
  ChatRoleUser,
  ChatSessionState,
  ChatSessionStreamChunkError,
  ChatSessionStreamChunkReplies,
  ContextLengthMaximal,
  DownloadPhaseDownloaded,
  DownloadPhaseNotDownloaded,
  DownloadState,
  Metadata,
  Model,
  ModelAccessibilityOnDevice,
  ModelProperties,
  ModelSourceFilesystem,
  ModelSourceRegistry,
  ModelSpecializationChat,
  ReasoningEffort,
  Repository,
  ToolCall,
  ToolDescriptionFunction,
  Value,
  type ChatInstance,
  type ChatSession,
  type ChatSessionStream,
  type ChatSessionStreamChunk,
} from "@trymirai/uzu";
import {
  AssistantMessageFrameEncoder,
  reduceAssistantMessageFrames,
  type AssistantMessageFrame,
} from "@earendil-works/pi-ai/utils/assistant-message-frame";
import { SessionCache } from "../src/cache.js";
import { prepareInput } from "../src/messages.js";
import { discoverModels, modelDefinition } from "../src/models.js";
import { createStream, emptyMessage } from "../src/stream.js";

const model = Model.external(
  "test",
  "local",
  "Local",
  "uzu",
  "Uzu",
  "1",
  [new ModelSpecializationChat()],
  new ModelAccessibilityOnDevice(new ModelSourceFilesystem("/test/model")),
);
const info = new ChatModelInfo(
  65536,
  new ChatModelCapabilities(
    true,
    true,
    [ReasoningEffort.Default, ReasoningEffort.Disabled],
    true,
    true,
    false,
  ),
  ReasoningEffort.Default,
);
const definition = modelDefinition(model, info, true);
const context = (): TranscriptContext =>
  normalizeContext({
    systemPrompt: "Help",
    tools: [
      {
        name: "read",
        description: "Read",
        parameters: Type.Object({ path: Type.String() }),
      },
    ],
    messages: [{ role: "user", content: "hello", timestamp: 1 }],
  });
const reply = (message: ChatMessage, finish?: ChatReplyFinishReason) =>
  new ChatReply(
    message,
    new ChatReplyStats(1, undefined, undefined, undefined, 8, 32, 5),
    finish,
  );
const candidate = (id: string, raw: string) =>
  new ChatContentBlockToolCallCandidate(new Value(JSON.stringify(raw)), id);
const call = (id: string, name: string, args: unknown) =>
  new ToolCall(id, name, new Value(JSON.stringify(args)));

async function collect(
  stream: ReturnType<typeof createAssistantMessageEventStream>,
) {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(structuredClone(event));
  return { events, message: await stream.result() };
}

function fixture(getOwnerSessionId?: () => string | undefined) {
  let resolutions = 0;
  let loads = 0;
  let resets = 0;
  let sessions = 0;
  const inputs: ChatMessage[][] = [];
  const instanceConfigs: ChatConfig[] = [];
  const replyConfigs: ChatReplyConfig[] = [];
  let chunks: ChatSessionStreamChunk[] = [];
  let onNext: (() => Promise<void>) | undefined;
  let token: CancelToken;
  const engine = {
    async chatInstance(
      _model: Model,
      config: ChatConfig,
    ): Promise<ChatInstance> {
      instanceConfigs.push(config);
      loads++;
      return { model };
    },
    async chatWithInstance(): Promise<ChatSession> {
      sessions++;
      return {
        state: Promise.resolve(ChatSessionState.Idle),
        messages: Promise.resolve([]),
        supportsToolCalls: Promise.resolve(true),
        samplingDefaults: Promise.resolve(null),
        async addTool() {
          return undefined;
        },
        async addTools() {
          return undefined;
        },
        async reset() {
          resets++;
        },
        async reply() {
          throw new Error("unused");
        },
        async replyWithStream(messages, config): Promise<ChatSessionStream> {
          replyConfigs.push(config);
          inputs.push(messages);
          token = CancelToken.create();
          const next = async () => {
            await new Promise<void>((resolve) => setImmediate(resolve));
            await onNext?.();
            return token.isCancelled ? null : (chunks.shift() ?? null);
          };
          return {
            cancelToken: token,
            next,
            async *[Symbol.asyncIterator]() {
              let chunk;
              while ((chunk = await next()) !== null) yield chunk;
            },
          };
        },
      };
    },
  };
  const cache = new SessionCache(engine, getOwnerSessionId);
  const stream = createStream(cache, async () => {
    resolutions++;
    return { model, info, definition };
  });
  return {
    cache,
    stream,
    inputs,
    instanceConfigs,
    replyConfigs,
    get resolutions() {
      return resolutions;
    },
    get loads() {
      return loads;
    },
    get resets() {
      return resets;
    },
    get sessions() {
      return sessions;
    },
    setReplies(replies: ChatReply[]) {
      chunks = replies.map(
        (value) => new ChatSessionStreamChunkReplies([value]),
      );
    },
    setChunks(value: ChatSessionStreamChunk[]) {
      chunks = value;
    },
    setOnNext(callback: () => Promise<void>) {
      onNext = callback;
    },
    get token() {
      return token;
    },
  };
}

test("streams reasoning, Unicode text and tool arguments through Pi frames", async () => {
  const f = fixture();
  f.setReplies([
    reply(
      ChatMessage.assistant()
        .withReasoning("think")
        .withText("🙂")
        .withBlock(candidate("id", "<function=read>\n<parameter=path>\n/")),
    ),
    reply(
      ChatMessage.assistant()
        .withReasoning("thinking")
        .withText("🙂 ok")
        .withToolCall(call("id", "read", { path: "/tmp" })),
      ChatReplyFinishReason.ToolCalls,
    ),
  ]);
  const result = await collect(
    f.stream(definition, context(), { sessionId: "s" }),
  );
  assert.equal(result.message.stopReason, "toolUse");
  assert.deepEqual(
    result.events
      .filter((e) => e.type === "toolcall_delta")
      .map((e) => e.delta),
    ['{"path":"/', 'tmp"}'],
  );
  const encoder = new AssistantMessageFrameEncoder();
  const frames: AssistantMessageFrame[] = [];
  let liveArguments: unknown;
  for (const event of result.events) {
    const frame = encoder.encode(event);
    if (frame) frames.push(frame);
    if (event.type === "toolcall_delta" && liveArguments === undefined) {
      liveArguments = reduceAssistantMessageFrames(frames)?.content.find(
        (block) => block.type === "toolCall",
      )?.arguments;
    }
  }
  assert.deepEqual(liveArguments, { path: "/" });
  const snapshot = result.events.find((e) => e.type === "toolcall_delta")!;
  assert.equal(snapshot.type, "toolcall_delta");
  if (snapshot.type === "toolcall_delta")
    assert.deepEqual(snapshot.partial.content[2], {
      type: "toolCall",
      id: "id",
      name: "read",
      arguments: { path: "/" },
    });
  assert.equal(
    result.events.filter((e) => e.type === "toolcall_start").length,
    1,
  );
  assert.equal(
    result.events.filter((e) => e.type === "toolcall_end").length,
    1,
  );
  assert.equal(result.events.filter((e) => e.type === "done").length, 1);
  assert.deepEqual(result.message.usage, {
    input: 8,
    cacheRead: 32,
    output: 5,
    cacheWrite: 0,
    totalTokens: 45,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  });
});

test("tool declarations use the developer role while instructions and reasoning stay in system", async () => {
  const f = fixture();
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { sessionId: "s" }).result();
  const messages = f.inputs[0]!;
  assert.equal(messages.length, 3);
  assert.ok(messages[0]!.role instanceof ChatRoleSystem);
  assert.equal(messages[0]!.text, "Help");
  assert.equal(messages[0]!.reasoningEffort, ReasoningEffort.Disabled);
  assert.deepEqual(messages[0]!.toolNamespaces, []);
  assert.ok(messages[1]!.role instanceof ChatRoleDeveloper);
  assert.equal(messages[1]!.toolNamespaces[0]!.name, "functions");
  const tool = messages[1]!.toolNamespaces[0]!.tools[0]!;
  assert.ok(tool instanceof ToolDescriptionFunction);
  assert.equal(tool.toolFunction.name, "read");
  assert.deepEqual(JSON.parse(tool.toolFunction.parameters.json), {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
  });
  assert.ok(messages[2]!.role instanceof ChatRoleUser);

  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f
    .stream(definition, context(), { sessionId: "s", toolChoice: "none" })
    .result();
  assert.equal(f.resets, 1);
  assert.equal(f.inputs[1]!.length, 2);
  assert.ok(f.inputs[1]![0]!.role instanceof ChatRoleSystem);
  assert.ok(f.inputs[1]![1]!.role instanceof ChatRoleUser);
  assert.deepEqual(
    f.inputs[1]!.flatMap((message) => message.toolNamespaces),
    [],
  );
});

test("tool round trips append only results, keep weights and KV, and reset on history or settings changes", async () => {
  const f = fixture();
  const firstContext = context();
  f.setReplies([
    reply(
      ChatMessage.assistant().withToolCall(call("id", "read", { path: "a" })),
      ChatReplyFinishReason.ToolCalls,
    ),
  ]);
  const first = await f
    .stream(definition, firstContext, { sessionId: "s" })
    .result();
  const next = normalizeContext({
    messages: [
      ...firstContext.messages,
      first,
      {
        role: "toolResult",
        toolCallId: "id",
        toolName: "read",
        content: [{ type: "text", text: "file" }],
        isError: false,
        timestamp: 2,
      },
    ],
  });
  f.setReplies([
    reply(ChatMessage.assistant().withText("done"), ChatReplyFinishReason.Stop),
  ]);
  const second = await f.stream(definition, next, { sessionId: "s" }).result();
  assert.equal(second.stopReason, "stop");
  assert.equal(f.inputs[0]?.length, 3);
  assert.equal(f.inputs[1]?.length, 1);
  assert.equal(
    f.inputs[1]?.[0]?.content[0]?.constructor.name,
    "ChatContentBlockToolCallResult",
  );
  assert.equal(f.loads, 1);
  assert.equal(f.resets, 0);
  f.setReplies([
    reply(
      ChatMessage.assistant().withText("branch"),
      ChatReplyFinishReason.Stop,
    ),
  ]);
  await f.stream(definition, firstContext, { sessionId: "s" }).result();
  assert.equal(f.resets, 1);
  f.setReplies([
    reply(
      ChatMessage.assistant().withText("reason"),
      ChatReplyFinishReason.Stop,
    ),
  ]);
  await f
    .stream(definition, firstContext, { sessionId: "s", reasoning: "low" })
    .result();
  assert.equal(f.resets, 2);
  assert.equal(f.loads, 1);
  await f.cache.release();
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, firstContext, { sessionId: "other" }).result();
  assert.equal(f.loads, 2);
});

test("a different session, changed tools and cacheRetention none never reuse prior KV state", async () => {
  const f = fixture();
  for (const sessionId of ["one", "two", "two"]) {
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    const input = context();
    if (f.inputs.length === 2)
      input.messages.unshift({
        role: "system",
        content: "changed",
        timestamp: 2,
      });
    await f.stream(definition, input, { sessionId }).result();
  }
  assert.equal(f.resets, 2);
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f
    .stream(definition, context(), { sessionId: "two", cacheRetention: "none" })
    .result();
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { sessionId: "two" }).result();
  assert.equal(f.sessions, 2);
  assert.equal(f.loads, 1);
});

test("malformed, unterminated and overflowing streams fail instead of executing partial calls", async () => {
  for (const value of [
    reply(
      ChatMessage.assistant().withBlock(
        new ChatContentBlockToolCallCandidate(new Value('"unfinished"')),
      ),
      ChatReplyFinishReason.ToolCalls,
    ),
    reply(
      ChatMessage.assistant().withToolCall(
        call("id", "read", { __uzu_unparsed_arguments: "bad" }),
      ),
      ChatReplyFinishReason.ToolCalls,
    ),
    reply(ChatMessage.assistant().withText("unfinished")),
    reply(ChatMessage.assistant(), ChatReplyFinishReason.ContextLimitReached),
    reply(
      ChatMessage.assistant().withToolCall(
        call("id", "read", { __uzu_unparsed_arguments: "partial" }),
      ),
      ChatReplyFinishReason.ContextLimitReached,
    ),
  ]) {
    const f = fixture();
    f.setReplies([value]);
    const { events, message } = await collect(
      f.stream(definition, context(), { sessionId: "s" }),
    );
    assert.equal(message.stopReason, "error");
    assert.equal(events.filter((e) => e.type === "error").length, 1);
    assert.equal(
      events.filter((e) => e.type === "done" || e.type === "toolcall_end")
        .length,
      0,
    );
    if (value.finishReason === ChatReplyFinishReason.ContextLimitReached)
      assert.match(message.errorMessage!, /context_length_exceeded/);
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f.stream(definition, context(), { sessionId: "s" }).result();
    assert.equal(f.sessions, 2);
  }
});

test("native input overflow triggers Pi's compaction detector without relabeling other errors", async () => {
  for (const error of [
    { type: "Backend", message: "Context overflow" },
    { type: "Backend", message: "Unable to allocate buffer" },
    { type: "Loading", message: "Context overflow" },
  ] as const) {
    const f = fixture();
    f.setChunks([new ChatSessionStreamChunkError(error)]);
    const { message, events } = await collect(
      f.stream(definition, context(), { sessionId: "s" }),
    );
    const overflow =
      error.type === "Backend" && error.message === "Context overflow";
    assert.equal(message.stopReason, "error");
    assert.equal(isContextOverflow(message), overflow);
    if (!overflow) assert.equal(message.errorMessage, JSON.stringify(error));
    assert.equal(events.filter((event) => event.type === "error").length, 1);
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f.stream(definition, context(), { sessionId: "s" }).result();
    assert.equal(f.sessions, 2);
  }
});

test("unsupported required tool constraints fail before download or inference", async () => {
  for (const choice of ["any", { type: "tool", name: "read" }]) {
    const f = fixture();
    f.setReplies([
      reply(
        ChatMessage.assistant().withText("hello"),
        ChatReplyFinishReason.Stop,
      ),
    ]);
    // Runtime callers can supply values outside Pi 1.1's auto/none TypeScript union.
    const options = { toolChoice: choice } as SimpleStreamOptions;
    const result = await f.stream(definition, context(), options).result();
    assert.equal(result.stopReason, "error");
    assert.match(result.errorMessage!, /toolChoice.*auto.*none/);
    assert.equal(f.resolutions, 0);
    assert.equal(f.loads, 0);
  }

  for (const strict of ["require", "prefer"] as const) {
    const f = fixture();
    const input = normalizeContext({
      tools: [
        {
          name: "read",
          description: "Read",
          parameters: Type.Object({ path: Type.String() }),
          constrainedSampling: { type: "json_schema", strict },
        },
      ],
      messages: [{ role: "user", content: "hello", timestamp: 1 }],
    });
    f.setReplies([
      reply(
        ChatMessage.assistant().withText("hello"),
        ChatReplyFinishReason.Stop,
      ),
    ]);
    const result = await f.stream(definition, input).result();
    if (strict === "require") {
      assert.equal(result.stopReason, "error");
      assert.match(
        result.errorMessage!,
        /read.*requires JSON-schema constrained sampling/,
      );
      assert.equal(f.resolutions, 0);
      assert.equal(f.loads, 0);
      const withoutTools = await f
        .stream(definition, input, { toolChoice: "none" })
        .result();
      assert.equal(withoutTools.stopReason, "stop");
    } else {
      assert.equal(result.stopReason, "stop");
      assert.equal(f.loads, 1);
    }
  }
});

test("session cleanup releases weights after failure, cancellation, or disabled KV retention", async () => {
  for (const outcome of ["failure", "cancel", "no-cache"] as const) {
    const f = fixture();
    const controller = new AbortController();
    if (outcome === "failure")
      f.setChunks([
        new ChatSessionStreamChunkError({ type: "Backend", message: "failed" }),
      ]);
    else
      f.setReplies([
        reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop),
      ]);
    if (outcome === "cancel") f.setOnNext(async () => controller.abort());
    const result = await f
      .stream(definition, context(), {
        sessionId: "owner",
        signal: controller.signal,
        cacheRetention: outcome === "no-cache" ? "none" : undefined,
      })
      .result();
    assert.equal(
      result.stopReason,
      outcome === "failure"
        ? "error"
        : outcome === "cancel"
          ? "aborted"
          : "stop",
    );
    await f.cache.release("unrelated");
    await f.cache.release("owner");
    f.setOnNext(async () => {});
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f.stream(definition, context(), { sessionId: "next" }).result();
    assert.equal(f.loads, 2, outcome);
  }
});

test("cleanup from a previous owner does not release another session's retained weights", async () => {
  const f = fixture();
  for (const sessionId of ["first", "second"]) {
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f
      .stream(definition, context(), { sessionId, cacheRetention: "none" })
      .result();
  }
  await f.cache.release("first");
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { sessionId: "second" }).result();
  assert.equal(f.loads, 1);
  await f.cache.release("second");
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { sessionId: "third" }).result();
  assert.equal(f.loads, 2);
});

test("temporary requests keep their owning Pi session for cleanup", async () => {
  for (const sessionId of [undefined, "summary-request"]) {
    let owner = "parent";
    const f = fixture(() => owner);
    for (const options of [
      { sessionId: owner },
      { sessionId, cacheRetention: "none" as const },
    ]) {
      f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
      assert.equal(
        (await f.stream(definition, context(), options).result()).stopReason,
        "stop",
      );
    }
    await f.cache.release("summary-request");
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f
      .stream(definition, context(), { sessionId, cacheRetention: "none" })
      .result();
    assert.equal(f.loads, 1);
    await f.cache.release(owner);
    owner = "next";
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f.stream(definition, context(), { sessionId: owner }).result();
    assert.equal(f.loads, 2);
  }
});

test("an anonymous request does not erase an existing cache owner", async () => {
  const f = fixture();
  for (const sessionId of ["owner", undefined]) {
    f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
    await f.stream(definition, context(), { sessionId }).result();
  }
  await f.cache.release("owner");
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { sessionId: "next" }).result();
  assert.equal(f.loads, 2);
});

test("cancellation drains the native stream, releases the lock and discards the session", async () => {
  const f = fixture();
  const abort = new AbortController();
  f.setOnNext(async () => {
    abort.abort();
  });
  f.setReplies([
    reply(
      ChatMessage.assistant().withText("ignored"),
      ChatReplyFinishReason.Stop,
    ),
  ]);
  const result = await collect(
    f.stream(definition, context(), { signal: abort.signal, sessionId: "s" }),
  );
  assert.equal(result.message.stopReason, "aborted");
  assert.equal(f.token.isCancelled, true);
  f.setOnNext(async () => {});
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  assert.equal(
    (await f.stream(definition, context(), { sessionId: "s" }).result())
      .stopReason,
    "stop",
  );
  assert.equal(f.sessions, 2);
});

test("a queued request aborts immediately without letting later requests overtake the active request", async () => {
  const f = fixture();
  let release!: () => void;
  let entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const held = f.cache.exclusive(
    undefined,
    () =>
      new Promise<void>((resolve) => {
        release = resolve;
        entered();
      }),
  );
  await started;
  const controller = new AbortController();
  const waiting = f.stream(definition, context(), {
    signal: controller.signal,
  });
  controller.abort();
  let followingRan = false;
  const following = f.cache.exclusive(undefined, async () => {
    followingRan = true;
  });
  try {
    const result = await Promise.race([
      waiting.result(),
      new Promise<undefined>((resolve) =>
        setImmediate(() => resolve(undefined)),
      ),
    ]);
    assert.equal(
      result?.stopReason,
      "aborted",
      "cancellation must finish while the first request still holds the lock",
    );
    assert.equal(followingRan, false);
    assert.equal(f.loads, 0);
  } finally {
    release();
    await held;
    await following;
  }
  assert.equal(followingRan, true);
  assert.equal((await waiting.result()).stopReason, "aborted");
  const cancelled = await f
    .stream(definition, context(), { signal: AbortSignal.abort() })
    .result();
  assert.equal(cancelled.stopReason, "aborted");
  assert.equal(f.loads, 0);
});

test("cancellation after the final chunk still discards the session", async () => {
  const f = fixture();
  const controller = new AbortController();
  let reads = 0;
  f.setOnNext(async () => {
    if (++reads === 2) controller.abort();
  });
  f.setReplies([
    reply(ChatMessage.assistant().withText("done"), ChatReplyFinishReason.Stop),
  ]);
  const result = await collect(
    f.stream(definition, context(), {
      sessionId: "s",
      signal: controller.signal,
    }),
  );
  assert.equal(result.message.stopReason, "aborted");
  assert.equal(
    result.events.filter((event) => event.type === "done").length,
    0,
  );
  assert.equal(f.token.isCancelled, true);
});

test("partial tool arguments can become valid, and multiple calls retain separate IDs", async () => {
  const f = fixture();
  f.setReplies([
    reply(
      ChatMessage.assistant().withBlock(
        candidate("first", "<function=read>\n<parameter=path>\na"),
      ),
    ),
    reply(
      ChatMessage.assistant()
        .withToolCall(call("first", "read", { path: "a" }))
        .withToolCall(call("second", "read", { path: "b" })),
      ChatReplyFinishReason.ToolCalls,
    ),
  ]);
  const result = await collect(f.stream(definition, context()));
  assert.equal(result.message.stopReason, "toolUse");
  assert.deepEqual(
    result.events
      .filter((event) => event.type === "toolcall_end")
      .map((event) => event.toolCall.id),
    ["first", "second"],
  );
  assert.deepEqual(result.message.content, [
    { type: "toolCall", id: "first", name: "read", arguments: { path: "a" } },
    { type: "toolCall", id: "second", name: "read", arguments: { path: "b" } },
  ]);
});

for (const format of ["Qwen XML", "Muse Glimmer ATEM"]) {
  test(`${format} streams escaped strings through Pi frames and preserves declared argument types`, async () => {
    const f = fixture();
    const text = 'London — 🙂\n"quoted"\\path\n</parametric>\n';
    const raw = format === "Qwen XML"
      ? `<function=write>\n<parameter=path>\n/tmp/a\n</parameter>\n<parameter=content>\n${text}\n</parameter>\n<parameter=count>\n3\n</parameter>\n</function>`
      : `<atem:function_calls>\n<atem:invoke name="write">\n<atem:parameter name="path">/tmp/a</atem:parameter>\n<atem:parameter name="content">${text}</atem:parameter>\n<atem:parameter name="count">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>`;
    let prefix = "";
    const replies: ChatReply[] = [];
    for (const character of raw) {
      prefix += character;
      replies.push(
        reply(ChatMessage.assistant().withBlock(candidate("write-id", prefix))),
      );
    }
    replies.push(
      reply(
        ChatMessage.assistant().withToolCall(
          call("write-id", "write", {
            path: "/tmp/a",
            content: text,
            count: "3",
          }),
        ),
        ChatReplyFinishReason.ToolCalls,
      ),
    );
    f.setReplies(replies);
    const input = normalizeContext({
      tools: [
        {
          name: "write",
          description: "Write",
          parameters: Type.Object({
            path: Type.String(),
            content: Type.String(),
            count: Type.Integer(),
          }),
        },
      ],
      messages: [{ role: "user", content: "Write", timestamp: 1 }],
    });
    const encoder = new AssistantMessageFrameEncoder();
    const frames: AssistantMessageFrame[] = [];
    const liveContents: string[] = [];
    const stream = f.stream(definition, input);
    for await (const event of stream) {
      const frame = encoder.encode(event);
      if (frame) frames.push(frame);
      if (event.type === "toolcall_delta") {
        const block = reduceAssistantMessageFrames(frames)?.content[0];
        if (
          block?.type === "toolCall" &&
          typeof block.arguments.content === "string"
        )
          liveContents.push(block.arguments.content);
      }
    }
    const result = await stream.result();
    assert.equal(result.stopReason, "toolUse", result.errorMessage);
    assert.ok(liveContents.includes("London"));
    assert.ok(liveContents.every((value) => text.startsWith(value)));
    assert.deepEqual(result.content, [
      {
        type: "toolCall",
        id: "write-id",
        name: "write",
        arguments: { path: "/tmp/a", content: text, count: 3 },
      },
    ]);
    assert.deepEqual(
      reduceAssistantMessageFrames(frames)?.content,
      result.content,
    );
  });
}

test("observer errors cancel generation and payload hooks can replace the native request", async () => {
  const f = fixture();
  f.setReplies([
    reply(ChatMessage.assistant().withText("x"), ChatReplyFinishReason.Stop),
  ]);
  const failed = await f
    .stream(definition, context(), {
      onProviderStreamEvent() {
        throw new Error("observer failed");
      },
    })
    .result();
  assert.equal(failed.errorMessage, "observer failed");
  assert.equal(f.token.isCancelled, true);
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  const success = await f
    .stream(definition, context(), {
      onPayload(payload) {
        assert.ok(
          payload &&
            typeof payload === "object" &&
            "input" in payload &&
            "config" in payload,
        );
        return {
          ...payload,
          input: {
            ...prepareInput(context(), info, {}),
            messages: [
              { role: "user", content: [{ type: "text", text: "changed" }] },
            ],
          },
        };
      },
    })
    .result();
  assert.equal(success.stopReason, "stop");
  assert.equal(f.inputs.at(-1)?.at(-1)?.text, "changed");
});

test("advertised thinking levels match native efforts without duplicate aliases", () => {
  const profiles: {
    name: string;
    capabilities: ChatModelCapabilities;
    defaultEffort?: ReasoningEffort;
    expected: [ModelThinkingLevel, ReasoningEffort | undefined][];
  }[] = [
    {
      name: "Qwen3.8",
      capabilities: new ChatModelCapabilities(true, true, [
        ReasoningEffort.Disabled, ReasoningEffort.Default,
        ReasoningEffort.Low, ReasoningEffort.Medium, ReasoningEffort.XHigh,
      ], true, true, false),
      defaultEffort: ReasoningEffort.XHigh,
      expected: [
        ["off", ReasoningEffort.Disabled],
        ["low", ReasoningEffort.Low],
        ["medium", ReasoningEffort.Medium],
        ["xhigh", ReasoningEffort.XHigh],
      ],
    },
    {
      name: "Muse Glimmer",
      capabilities: new ChatModelCapabilities(true, false, [
        ReasoningEffort.Default, ReasoningEffort.Low, ReasoningEffort.Medium,
        ReasoningEffort.High, ReasoningEffort.XHigh,
      ], true, true, false),
      defaultEffort: ReasoningEffort.High,
      expected: [
        ["low", ReasoningEffort.Low],
        ["medium", ReasoningEffort.Medium],
        ["high", ReasoningEffort.High],
        ["xhigh", ReasoningEffort.XHigh],
      ],
    },
    {
      name: "GPT-OSS",
      capabilities: new ChatModelCapabilities(true, false, [
        ReasoningEffort.Low, ReasoningEffort.Medium, ReasoningEffort.High,
      ], true, false, false),
      defaultEffort: ReasoningEffort.Medium,
      expected: [
        ["low", ReasoningEffort.Low],
        ["medium", ReasoningEffort.Medium],
        ["high", ReasoningEffort.High],
      ],
    },
    {
      name: "thinking toggle",
      capabilities: info.capabilities,
      defaultEffort: ReasoningEffort.Default,
      expected: [["off", ReasoningEffort.Disabled], ["medium", ReasoningEffort.Default]],
    },
    {
      name: "always thinking",
      capabilities: new ChatModelCapabilities(true, false, [], true, true, false),
      expected: [["medium", ReasoningEffort.Default]],
    },
    {
      name: "no thinking",
      capabilities: new ChatModelCapabilities(false, false, [], true, true, false),
      expected: [["off", undefined]],
    },
  ];
  for (const profile of profiles) {
    const metadata = new ChatModelInfo(65536, profile.capabilities, profile.defaultEffort);
    const definition = modelDefinition(model, metadata, true);
    assert.equal(definition.reasoning, profile.capabilities.supportsReasoning, profile.name);
    assert.deepEqual(
      getSupportedThinkingLevels(definition),
      profile.expected.map(([level]) => level),
      profile.name,
    );
    assert.equal(definition.thinkingLevelMap?.minimal, null, profile.name);
    assert.equal(definition.thinkingLevelMap?.max, null, profile.name);
    for (const [level, effort] of profile.expected) {
      const options = { reasoning: level === "off" ? undefined : level };
      assert.equal(prepareInput(context(), metadata, options).reasoning, effort, `${profile.name}: ${level}`);
    }
    for (const requested of ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const) {
      const clamped = clampThinkingLevel(definition, requested);
      assert.ok(profile.expected.some(([level]) => level === clamped), `${profile.name}: ${requested}`);
    }
  }
});

test("model metadata keeps the model's full context and lists models before download", async () => {
  assert.equal(definition.contextWindow, 65536);
  assert.equal(definition.maxTokens, definition.contextWindow);
  assert.equal(definition.reasoning, true);
  assert.deepEqual(getSupportedThinkingLevels(definition), ["off", "medium"]);
  assert.deepEqual(definition.input, ["text"]);
  assert.equal(definition.id, model.identifier);
  const downloadable = Model.external(
    "alibaba:qwen3.5:9b:mirai:mirai-m:4",
    "test",
    "Test",
    "uzu",
    "Uzu",
    "1",
    [new ModelSpecializationChat()],
    new ModelAccessibilityOnDevice(
      new ModelSourceRegistry(
        "1",
        new Repository("trymirai/Qwen3.5-9B-M"),
        new Repository("Qwen/Qwen3.5-9B"),
        [],
      ),
    ),
  );
  const pendingInfo = new ChatModelInfo(
    undefined,
    info.capabilities,
    info.defaultReasoningEffort,
  );
  const result = await discoverModels({
    modelsOnDevice: Promise.resolve([model, downloadable]),
    downloadStates: Promise.resolve({
      [downloadable.identifier]: new DownloadState(
        100,
        0,
        new DownloadPhaseNotDownloaded(),
      ),
    }),
    async chatModelInfo(value) {
      if (value.identifier === downloadable.identifier) return pendingInfo;
      return info;
    },
  });
  assert.equal(result.entries.size, 2);
  assert.equal(
    result.entries.get(model.identifier)?.definition.contextWindow,
    65536,
  );
  const entry = result.entries.get("trymirai/Qwen3.5-9B-M")!;
  assert.equal(entry.model.identifier, downloadable.identifier);
  const pending = entry.definition;
  assert.equal(pending.id, "trymirai/Qwen3.5-9B-M");
  assert.match(pending.name, /not downloaded/);
  assert.equal(pending.contextWindow, 0);
  assert.equal(pending.reasoning, true);
  assert.equal(
    modelDefinition(model, pendingInfo, true).contextWindow,
    Infinity,
  );
  assert.equal(modelDefinition(model, pendingInfo, true).maxTokens, 0);
  const downloaded = await discoverModels({
    modelsOnDevice: Promise.resolve([downloadable]),
    downloadStates: Promise.resolve({
      [downloadable.identifier]: new DownloadState(
        100,
        100,
        new DownloadPhaseDownloaded(),
      ),
    }),
    async chatModelInfo() {
      return info;
    },
  });
  assert.equal(
    downloaded.entries.get(pending.id)?.definition.name,
    downloadable.name,
  );
  assert.equal(
    downloaded.entries.get(pending.id)?.definition.contextWindow,
    65536,
  );
  const invalid = await discoverModels({
    modelsOnDevice: Promise.resolve([model]),
    downloadStates: Promise.resolve({}),
    async chatModelInfo() {
      throw new Error("bad config");
    },
  });
  assert.match(invalid.errors[0]!, /bad config/);
});

test("model discovery filters tool support and size for downloaded and missing models", async () => {
  const candidates = [
    { id: "small", size: 8_999_999_999, supportsTools: true },
    { id: "minimum", size: 9_000_000_000, supportsTools: true },
    { id: "large", size: 27_000_000_000, supportsTools: true },
    { id: "no-tools", size: 9_000_000_000, supportsTools: false },
    { id: "unknown-size", size: undefined, supportsTools: true },
    { id: "unknown-size-no-tools", size: undefined, supportsTools: false },
  ];
  for (const downloaded of [false, true]) {
    const models = candidates.map(({ id, size }) => {
      const candidate = Model.external(
        id,
        "test",
        "Test",
        "uzu",
        "Uzu",
        "1",
        [new ModelSpecializationChat()],
        new ModelAccessibilityOnDevice(
          new ModelSourceRegistry(
            "1", new Repository(`test/${id}`), undefined, [],
          ),
        ),
      );
      if (size !== undefined)
        candidate.properties = new ModelProperties(
          id,
          size,
          undefined,
          Metadata.external(id),
        );
      return candidate;
    });
    const catalog = await discoverModels({
      modelsOnDevice: Promise.resolve(models),
      downloadStates: Promise.resolve(
        Object.fromEntries(
          models.map((model) => [
            model.identifier,
            new DownloadState(
              100,
              downloaded ? 100 : 0,
              downloaded
                ? new DownloadPhaseDownloaded()
                : new DownloadPhaseNotDownloaded(),
            ),
          ]),
        ),
      ),
      async chatModelInfo(model) {
        const capabilities = info.capabilities;
        capabilities.supportsTools = candidates.find(
          ({ id }) => id === model.identifier,
        )!.supportsTools;
        return new ChatModelInfo(65536, capabilities, info.defaultReasoningEffort);
      },
    });
    assert.deepEqual([...catalog.entries.keys()], [
      "test/minimum",
      "test/large",
      "test/unknown-size",
    ]);
    assert.deepEqual(catalog.errors, []);
  }
});

test("sessions use Uzu's maximal context and only explicit requests limit output", async () => {
  const f = fixture();
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context()).result();
  assert.ok(
    f.instanceConfigs[0]?.contextLength instanceof ContextLengthMaximal,
  );
  assert.equal(f.replyConfigs[0]?.tokenLimit, null);
  f.setReplies([reply(ChatMessage.assistant(), ChatReplyFinishReason.Stop)]);
  await f.stream(definition, context(), { maxTokens: 12345 }).result();
  assert.equal(f.replyConfigs[1]?.tokenLimit, 12345);
});

test("images fail explicitly and foreign provider tool calls retain their IDs and arguments", () => {
  assert.throws(
    () =>
      prepareInput(
        normalizeContext({
          messages: [
            {
              role: "user",
              timestamp: 1,
              content: [{ type: "image", mimeType: "image/png", data: "AA==" }],
            },
          ],
        }),
        info,
        {},
      ),
    /text input only/,
  );
  const foreign = emptyMessage({ ...definition, provider: "foreign" });
  foreign.stopReason = "toolUse";
  foreign.content = [
    {
      type: "toolCall",
      id: "foreign-id",
      name: "read",
      arguments: { path: "x" },
    },
  ];
  assert.deepEqual(
    prepareInput(normalizeContext({ messages: [foreign] }), info, {})
      .messages[0]?.content,
    foreign.content,
  );
  foreign.stopReason = "aborted";
  assert.deepEqual(
    prepareInput(
      normalizeContext({
        messages: [
          foreign,
          {
            role: "toolResult",
            toolCallId: "foreign-id",
            toolName: "read",
            content: [{ type: "text", text: "Cancelled" }],
            isError: true,
            timestamp: 1,
          },
        ],
      }),
      info,
      {},
    ).messages,
    [],
  );
});
