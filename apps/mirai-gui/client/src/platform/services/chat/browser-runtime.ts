import type { TitleGenParams } from ".";
import type { BrowserEncodingSupport } from "./browser-protocol";
import type { RunEvent, RunTransport } from "./run-stream";
import type { EngineRequest, EngineResponse } from "./browser-protocol";
import type { SessionLoadingPayload, SessionStatePayload } from "../session";
import { onBrowserAutoEjectChanged, webSettings } from "../settings/web";

type PendingRequest = {
  resolve(value: unknown): void;
  reject(error: Error): void;
  onEvent?: (event: RunEvent) => void;
};

export function browserInferenceError(): string | null {
  if (!globalThis.isSecureContext) return "Browser inference requires HTTPS or localhost.";
  if (!("gpu" in navigator)) return "WebGPU is unavailable. Use a browser and device with WebGPU enabled.";
  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
    return "Browser inference requires cross-origin isolation. Serve Mirai with Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.";
  }
  return null;
}

export class BrowserRuntime {
  private worker?: Worker;
  private pending = new Map<string, PendingRequest>();
  private state: SessionStatePayload = { active: false };
  private loading?: SessionLoadingPayload;
  private stateListeners = new Set<(state: SessionStatePayload) => void>();
  private loadingListeners = new Set<(state: SessionLoadingPayload) => void>();
  private idleTimer?: ReturnType<typeof setTimeout>;
  private idleGeneration = 0;
  private stopSettingsListener?: () => void;
  private activeRun?: string;
  private activeCompletion?: Promise<void>;
  private activeTitle?: string;
  private loadCancellation?: Int32Array;

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL("./engine-worker.ts", import.meta.url), { type: "module" }),
  ) {}

  private updateState(state: SessionStatePayload) {
    this.state = state;
    this.loading = undefined;
    this.stateListeners.forEach((listener) => listener(state));
  }

  private reset(error: Error) {
    const repoId = this.loading?.repoId ?? this.state.repoId;
    this.worker?.terminate();
    this.worker = undefined;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.activeRun = undefined;
    this.activeCompletion = undefined;
    this.activeTitle = undefined;
    this.loadCancellation = undefined;
    this.cancelEject();
    this.stopSettingsListener?.();
    this.stopSettingsListener = undefined;
    this.updateState({ active: false, repoId, isEjecting: false });
  }

  private getWorker(inference: boolean): Worker {
    const unavailable = inference ? browserInferenceError() : null;
    if (unavailable) throw new Error(unavailable);
    if (this.worker) return this.worker;
    const worker = this.createWorker();
    worker.onmessage = (event: MessageEvent<EngineResponse>) => {
      const message = event.data;
      if (message.type === "session") this.updateState(message.state);
      else if (message.type === "loading") {
        this.loading = message.state;
        this.loadingListeners.forEach((listener) => listener(message.state));
      } else {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        if (message.type === "event") pending.onEvent?.(message.event);
        else {
          this.pending.delete(message.id);
          if (message.error) pending.reject(new Error(message.error));
          else pending.resolve(message.value);
        }
      }
    };
    worker.onerror = (event) =>
      this.reset(new Error(event.message || "The browser inference worker stopped unexpectedly"));
    worker.onmessageerror = () => this.reset(new Error("Could not read the browser engine response"));
    this.worker = worker;
    this.stopSettingsListener = onBrowserAutoEjectChanged(() => {
      void this.scheduleEject().catch(() => undefined);
    });
    return worker;
  }

  private request<T = void>(request: EngineRequest, onEvent?: (event: RunEvent) => void): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const worker = this.getWorker(request.type === "run" || request.type === "title");
      this.pending.set(request.id, { resolve: (value) => resolve(value as T), reject, onEvent });
      try {
        worker.postMessage(request);
      } catch (error) {
        this.pending.delete(request.id);
        reject(error);
      }
    });
  }

  private beginLoadCancellation(): Int32Array {
    const unavailable = browserInferenceError();
    if (unavailable) throw new Error(unavailable);
    const flag = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
    this.loadCancellation = flag;
    return flag;
  }

  readonly transport: RunTransport = {
    start: async (runId, params, onEvent) => {
      if (this.activeRun) throw new Error("Another response is already running");
      this.activeRun = runId;
      this.cancelEject();
      try {
        const loadCancellation = this.beginLoadCancellation();
        const operation = this.request({ id: runId, type: "run", params, loadCancellation }, onEvent);
        this.activeCompletion = operation.then(
          () => undefined,
          () => undefined,
        );
        await operation;
      } finally {
        if (this.activeRun === runId) {
          this.activeRun = undefined;
          this.activeCompletion = undefined;
          this.loadCancellation = undefined;
        }
        void this.scheduleEject();
      }
    },
    cancel: async (runId) => {
      if (!this.worker || this.activeRun !== runId) return;
      // Synchronous model reads can observe this immediately, while the engine
      // worker cannot process a queued cancellation message until they finish.
      if (this.loadCancellation) Atomics.store(this.loadCancellation, 0, 1);
      // A long GPU submission cannot receive worker messages until it completes.
      // Termination provides a bounded stop even if the device or bridge stalls.
      const timeout = setTimeout(() => this.reset(new Error("Response cancelled")), 5000);
      try {
        await this.request({ id: crypto.randomUUID(), type: "cancel", runId });
      } finally {
        clearTimeout(timeout);
      }
    },
  };

  inspectEncoding(encoding: unknown): Promise<BrowserEncodingSupport> {
    return this.request({ id: crypto.randomUUID(), type: "inspectEncoding", encoding });
  }

  inspectModelConfig(config: Uint8Array): Promise<void> {
    return this.request({ id: crypto.randomUUID(), type: "inspectModelConfig", config });
  }

  async generateTitle(params: TitleGenParams): Promise<string> {
    if (this.activeRun) throw new Error("Another response is already running");
    const id = crypto.randomUUID();
    this.activeRun = id;
    this.activeTitle = id;
    this.cancelEject();
    try {
      const loadCancellation = this.beginLoadCancellation();
      const operation = this.request<string>({ id, type: "title", params, loadCancellation });
      this.activeCompletion = operation.then(
        () => undefined,
        () => undefined,
      );
      return await operation;
    } finally {
      if (this.activeRun === id) {
        this.activeRun = undefined;
        this.activeCompletion = undefined;
        this.loadCancellation = undefined;
      }
      if (this.activeTitle === id) this.activeTitle = undefined;
      void this.scheduleEject();
    }
  }

  cancelTitleGen(): Promise<void> {
    return this.activeTitle ? this.transport.cancel(this.activeTitle) : Promise.resolve();
  }

  private cancelEject() {
    clearTimeout(this.idleTimer);
    this.idleGeneration += 1;
  }

  private async scheduleEject() {
    this.cancelEject();
    const generation = this.idleGeneration;
    if (!(await webSettings.getAutoEjectEnabled()) || this.activeRun || !this.state.active || !this.state.repoId)
      return;
    const repoId = this.state.repoId;
    const minutes = await webSettings.getAutoEjectMinutes();
    if (generation !== this.idleGeneration || this.activeRun || this.state.repoId !== repoId) return;
    this.idleTimer = setTimeout(
      () => {
        // Respect a preference change made after this idle period began.
        void webSettings
          .getAutoEjectEnabled()
          .then((enabled) =>
            enabled && generation === this.idleGeneration && !this.activeRun && this.state.repoId === repoId
              ? this.eject(repoId)
              : undefined,
          )
          .catch(() => undefined);
      },
      Math.min(24 * 60 * 60_000, Math.max(60_000, minutes * 60_000)),
    );
  }

  async eject(repoId: string): Promise<void> {
    this.cancelEject();
    const target = this.loading?.repoId ?? this.state.repoId;
    if (!this.worker || (target && target !== repoId)) return;
    if (this.activeRun) {
      if (this.loading?.status !== "start" || this.loading.repoId !== repoId)
        throw new Error("Stop the response before ejecting its model");
      const completion = this.activeCompletion;
      await this.transport.cancel(this.activeRun);
      // The worker can acknowledge cancel while an asynchronous cache read is
      // still unwinding. Wait for that load to finish before sending eject.
      await completion;
      if (!this.worker) return;
    }
    this.updateState({ ...this.state, repoId, isEjecting: true });
    try {
      await this.request({ id: crypto.randomUUID(), type: "eject", repoId });
      // Release WASM linear memory and the nested GPU worker, too.
      this.reset(new Error("Model ejected"));
    } catch (error) {
      this.updateState({ ...this.state, isEjecting: false });
      throw error;
    }
  }

  onSessionState(listener: (state: SessionStatePayload) => void) {
    this.stateListeners.add(listener);
    listener(this.state);
    return () => this.stateListeners.delete(listener);
  }

  onSessionLoading(listener: (state: SessionLoadingPayload) => void) {
    this.loadingListeners.add(listener);
    if (this.loading) listener(this.loading);
    return () => this.loadingListeners.delete(listener);
  }
}

export const browserRuntime = new BrowserRuntime();

export const inspectBrowserEncoding = (encoding: unknown) => browserRuntime.inspectEncoding(encoding);
export const inspectBrowserModelConfig = (config: Uint8Array) => browserRuntime.inspectModelConfig(config);
