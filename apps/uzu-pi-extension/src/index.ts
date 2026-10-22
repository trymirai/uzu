import { isDeepStrictEqual } from "node:util";
import {
  registerSessionResourceCleanup,
  type Provider,
} from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ExtensionUIContext,
} from "@earendil-works/pi-coding-agent";
import { Engine, EngineConfig } from "@trymirai/uzu";
import { SessionCache } from "./cache.js";
import { abortable } from "./abort.js";
import { prepareModel } from "./download.js";
import { api, discoverModels, modelDefinition, providerId } from "./models.js";
import { createStream } from "./stream.js";

export default async function uzuExtension(pi: ExtensionAPI) {
  const config = EngineConfig.create()
    .withAllowOllamaUsage(false)
    .withAllowLmstudioUsage(false);
  // This provider owns local inference; Pi already owns cloud provider credentials.
  config.openaiApiKey =
    config.anthropicApiKey =
    config.geminiApiKey =
    config.xaiApiKey =
      undefined;
  config.basetenApiKey = config.openrouterApiKey = undefined;
  const engine = await Engine.create(config);
  await registerUzuProvider(pi, engine);
}

export async function registerUzuProvider(pi: ExtensionAPI, engine: Engine) {
  let ownerSessionId: string | undefined;
  const cache = new SessionCache(engine, () => ownerSessionId);
  // Engine creation exposes the cached catalog while its refresh continues.
  let catalog = await discoverModels(engine);
  let ui: ExtensionUIContext | undefined;
  const stream = createStream(cache, async (id, signal) => {
    const entry = catalog.entries.get(id);
    if (!entry) return undefined;
    const info = await prepareModel(engine, entry.model, signal, (message) =>
      ui?.setStatus("uzu", message),
    );
    const definition = modelDefinition(entry.model, info, true);
    const next = { model: entry.model, info, definition };
    catalog.entries.set(id, next);
    // Pi refreshes its active model through the registry, preserving models.json overrides.
    if (!isDeepStrictEqual(definition, entry.definition))
      pi.registerProvider(provider);
    return next;
  });
  const provider: Provider<typeof api> = {
    id: providerId,
    name: "Uzu",
    auth: {
      apiKey: {
        name: "On-device Uzu",
        async resolve() {
          return { auth: {}, source: "On-device Uzu" };
        },
      },
    },
    getModels: () =>
      [...catalog.entries.values()].map((entry) => entry.definition),
    async refreshModels(context) {
      if (context.allowNetwork)
        await abortable(engine.refreshModels(), context.signal);
      const next = await discoverModels(engine, context.signal);
      await context.publish({
        update: () => {
          catalog = next;
        },
      });
    },
    stream,
    streamSimple: stream,
  };
  pi.registerProvider(provider);
  const unregisterCleanup = registerSessionResourceCleanup((sessionId) => {
    void cache.release(sessionId);
  });
  pi.on("session_shutdown", async (event) => {
    if (event.reason === "quit" || event.reason === "reload")
      unregisterCleanup();
    await cache.release();
    ownerSessionId = undefined;
    ui = undefined;
  });
  pi.on("before_provider_request", (_event, ctx) => {
    ownerSessionId = ctx.sessionManager.getSessionId();
  });
  pi.on("model_select", async (event, ctx) => {
    ui = ctx.ui;
    if (
      event.previousModel?.provider === providerId &&
      (event.model.provider !== providerId ||
        event.model.id !== event.previousModel.id)
    )
      await cache.release();
  });
  pi.on("session_start", async (_event, ctx) => {
    ownerSessionId = ctx.sessionManager.getSessionId();
    ui = ctx.ui;
    if (catalog.errors.length)
      ctx.ui.notify(
        `Some Uzu models could not be read:\n${catalog.errors.join("\n")}`,
        "warning",
      );
  });
}
