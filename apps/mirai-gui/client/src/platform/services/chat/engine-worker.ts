import { cachedModelFiles, findBrowserModel } from "../models/browser-models";
import { restrictCheckpointCapabilities } from "../models/browser-catalog";
import { emptyStats } from "./empty-stats";
import { browserTools } from "./browser-tools";
import { generateBrowserTitle } from "./browser-title";
import type { BrowserEncodingSupport, EngineRequest, EngineResponse } from "./browser-protocol";
import { initializeBrowserWasm, type WasmModule, type WasmSession } from "./browser-wasm";

let wasm: Promise<WasmModule> | undefined;
let session: WasmSession | undefined;
let sessionRepo: string | undefined;
let loadingRepo: string | undefined;
let activeRun: string | undefined;
let support: BrowserEncodingSupport | undefined;
let cancelled = false;

const send = (message: EngineResponse) => self.postMessage(message);

function loadWasm(): Promise<WasmModule> {
  wasm ??= (async () => {
    const url = new URL(`${import.meta.env.BASE_URL}wasm/uzu_web.js`, self.location.origin).href;
    let module: WasmModule;
    try {
      module = (await import(/* @vite-ignore */ url)) as WasmModule;
    } catch {
      throw new Error("The browser engine is missing. Run pnpm web:wasm before starting the web app.");
    }
    return initializeBrowserWasm(module);
  })().catch((error: unknown) => {
    wasm = undefined;
    throw error;
  });
  return wasm;
}

function disposeSession() {
  const repoId = sessionRepo ?? loadingRepo;
  session?.dispose();
  session = undefined;
  sessionRepo = undefined;
  loadingRepo = undefined;
  support = undefined;
  send({ type: "session", state: { active: false, repoId, isEjecting: false } });
}

async function getSession(repoId: string, loadCancellation: Int32Array): Promise<WasmSession> {
  const isCancelled = () => cancelled || Atomics.load(loadCancellation, 0) !== 0;
  const checkCancelled = () => {
    if (isCancelled()) throw new Error("Model loading cancelled");
  };
  checkCancelled();
  if (session && sessionRepo === repoId) return session;
  disposeSession();
  loadingRepo = repoId;
  send({ type: "loading", state: { status: "start", repoId } });
  try {
    const model = await findBrowserModel(repoId);
    const module = await loadWasm();
    const files = await cachedModelFiles(model, checkCancelled);
    checkCancelled();
    const config = files["config.json"];
    const tokenizer = files["tokenizer.json"];
    const weights = files["model.safetensors"];
    if (!config || !tokenizer || !weights)
      throw new Error("The model requires config.json, tokenizer.json and model.safetensors");
    const encoding = (model as typeof model & { encoding?: unknown }).encoding;
    support = encoding
      ? restrictCheckpointCapabilities(module.inspectEncoding(encoding), model.checkpointCapabilities)
      : undefined;
    session = await module.createSession({
      config,
      tokenizer,
      weights,
      loadCancellation,
      encoding,
    });
    checkCancelled();
    sessionRepo = repoId;
    loadingRepo = undefined;
    send({ type: "session", state: { active: true, repoId, isEjecting: false } });
    return session;
  } catch (error) {
    disposeSession();
    if (!isCancelled()) send({ type: "loading", state: { status: "error", repoId } });
    throw error;
  }
}

async function handle(request: EngineRequest): Promise<unknown> {
  switch (request.type) {
    case "inspectEncoding":
      return (await loadWasm()).inspectEncoding(request.encoding);
    case "inspectModelConfig":
      return (await loadWasm()).inspectModelConfig(request.config);
    case "title":
    case "run": {
      if (activeRun) throw new Error("Another response is already running");
      activeRun = request.id;
      cancelled = false;
      const isCancelled = () => cancelled || Atomics.load(request.loadCancellation, 0) !== 0;
      const finishCancelled = () => {
        if (request.type === "title") return "";
        send({
          type: "event",
          id: request.id,
          event: { type: "done", text: "", stats: emptyStats(), finishReason: "Cancelled" },
        });
        return undefined;
      };
      try {
        const engine = await getSession(request.params.repoId, request.loadCancellation);
        if (isCancelled()) return finishCancelled();
        if (request.type === "title") return await generateBrowserTitle(engine, request.params, support, isCancelled);
        const tools = browserTools(request.params, support?.supportsTools === true);
        await engine.run(tools.params, (event) => send({ type: "event", id: request.id, event }), tools.execute);
      } catch (error) {
        if (isCancelled()) return finishCancelled();
        throw error;
      } finally {
        activeRun = undefined;
      }
      return;
    }
    case "cancel":
      if (activeRun === request.runId) {
        cancelled = true;
        session?.cancel();
      }
      return;
    case "eject":
      if (activeRun) throw new Error("Stop the response before ejecting its model");
      if (!sessionRepo || sessionRepo === request.repoId) disposeSession();
      return;
  }
}

self.onmessage = (event: MessageEvent<EngineRequest>) => {
  const request = event.data;
  void handle(request).then(
    (value) => send({ type: "result", id: request.id, value }),
    (error: unknown) =>
      send({ type: "result", id: request.id, error: error instanceof Error ? error.message : String(error) }),
  );
};
