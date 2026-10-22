import assert from "node:assert/strict";
import { test } from "node:test";
import {
  cleanupSessionResources,
  getSupportedThinkingLevels,
  normalizeContext,
  type Provider,
} from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ExtensionContext,
  ExtensionEvent,
} from "@earendil-works/pi-coding-agent";
import {
  ChatModelCapabilities,
  ChatModelInfo,
  Model,
  ModelAccessibilityOnDevice,
  ModelSourceFilesystem,
  ModelSpecializationChat,
  ReasoningEffort,
  type Engine,
} from "@trymirai/uzu";
import { SessionCache } from "../src/cache.js";
import { registerUzuProvider } from "../src/index.js";

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
  new ChatModelCapabilities(false, true, [], true, true, false),
  ReasoningEffort.Default,
);

function fixture() {
  type Handler = (event: ExtensionEvent, ctx: ExtensionContext) => unknown;
  const handlers = new Map<string, Handler>();
  let provider: Provider;
  let loads = 0;
  const pi = {
    registerProvider(value: Provider) {
      provider = value;
    },
    on(event: string, handler: Handler) {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    },
  } as unknown as ExtensionAPI;
  const engine = {
    modelsOnDevice: Promise.resolve([model]),
    downloadStates: Promise.resolve({}),
    async chatModelInfo() {
      return info;
    },
    async modelPath() {
      return "/test/model";
    },
    async chatInstance() {
      loads++;
      return {};
    },
    async chatWithInstance() {
      throw new Error("Stop before inference");
    },
    // A registry that never responds must not delay cached-model registration.
    refreshModels: () => new Promise<Model[]>(() => {}),
  } as unknown as Engine;
  return {
    pi,
    engine,
    get provider() {
      return provider;
    },
    get loads() {
      return loads;
    },
    async emit(event: ExtensionEvent, sessionId = "session") {
      const ctx = {
        sessionManager: { getSessionId: () => sessionId },
        ui: { notify() {} },
      } as unknown as ExtensionContext;
      await handlers.get(event.type)?.(event, ctx);
    },
  };
}

test("cached models register without waiting for registry refresh", {
  timeout: 2000,
}, async (t) => {
  const f = fixture();
  const refresh = t.mock.method(f.engine, "refreshModels");
  await registerUzuProvider(f.pi, f.engine);
  t.after(() => f.emit({ type: "session_shutdown", reason: "quit" }));
  assert.equal(refresh.mock.callCount(), 0);
  assert.equal(f.provider.getModels().length, 1);
  assert.equal(f.provider.getModels()[0]?.name, model.name);

  // Pi's offline refresh also remains usable while networking is unavailable.
  let publications = 0;
  await f.provider.refreshModels!({
    allowNetwork: false,
    signal: new AbortController().signal,
    publish: async ({ update }) => {
      update?.();
      publications++;
      return true;
    },
  });
  assert.equal(publications, 1);
  assert.equal(refresh.mock.callCount(), 0);
});

test("provider cleanup uses the Pi session even before session_start", async (t) => {
  const f = fixture();
  await registerUzuProvider(f.pi, f.engine);
  t.after(() => f.emit({ type: "session_shutdown", reason: "quit" }));
  const definition = f.provider.getModels()[0]!;
  const context = normalizeContext({
    messages: [{ role: "user", content: "hello", timestamp: 1 }],
  });
  for (const owner of ["first", "second"]) {
    const response = await f.provider.streamSimple(definition, context, {
      sessionId: "temporary-summary-id",
      cacheRetention: "none",
      onPayload: async (payload) => {
        await f.emit({ type: "before_provider_request", payload }, owner);
      },
    }).result();
    assert.equal(response.errorMessage, "Stop before inference");
    cleanupSessionResources(owner);
  }
  assert.equal(f.loads, 2);
});

test("capability changes refresh Pi's model even when its name and context stay the same", async (t) => {
  const f = fixture();
  const registration = t.mock.method(f.pi, "registerProvider");
  await registerUzuProvider(f.pi, f.engine);
  t.after(() => f.emit({ type: "session_shutdown", reason: "quit" }));
  const definition = f.provider.getModels()[0]!;
  assert.deepEqual(getSupportedThinkingLevels(definition), ["off"]);
  t.mock.method(f.engine, "chatModelInfo", async () => new ChatModelInfo(
    info.contextLength,
    new ChatModelCapabilities(true, true, [ReasoningEffort.Disabled, ReasoningEffort.Default], true, true, false),
    ReasoningEffort.Default,
  ));
  await f.provider.streamSimple(definition, normalizeContext({
    messages: [{ role: "user", content: "hello", timestamp: 1 }],
  })).result();
  assert.equal(registration.mock.callCount(), 2);
  const refreshed = f.provider.getModels()[0]!;
  assert.equal(refreshed.name, definition.name);
  assert.equal(refreshed.contextWindow, definition.contextWindow);
  assert.deepEqual(getSupportedThinkingLevels(refreshed), ["off", "medium"]);
});

test("session cleanup survives replacements and unregisters on quit or reload", async (t) => {
  const releases: (string | undefined)[] = [];
  t.mock.method(SessionCache.prototype, "release", async (sessionId?: string) => {
    releases.push(sessionId);
  });
  for (const finalReason of ["quit", "reload"] as const) {
    const f = fixture();
    await registerUzuProvider(f.pi, f.engine);
    try {
      await f.emit({ type: "session_start", reason: "startup" });
      for (const reason of ["new", "resume", "fork"] as const) {
        releases.length = 0;
        await f.emit({ type: "session_shutdown", reason });
        cleanupSessionResources("old");
        await f.emit({ type: "session_start", reason }, "next");
        cleanupSessionResources("next");
        assert.deepEqual(releases, [undefined, "old", "next"]);
      }
      await f.emit({ type: "session_shutdown", reason: finalReason });
      releases.length = 0;
      cleanupSessionResources("disposed");
      assert.deepEqual(releases, []);
    } finally {
      await f.emit({ type: "session_shutdown", reason: "quit" });
    }
  }
});
