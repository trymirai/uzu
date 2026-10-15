import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ChatModelCapabilities,
  ChatModelInfo,
  DownloadPhaseDownloaded,
  DownloadPhaseDownloading,
  DownloadPhaseError,
  DownloadPhaseLocked,
  DownloadPhaseNotDownloaded,
  DownloadPhasePaused,
  DownloadState,
  Model,
  ModelAccessibilityOnDevice,
  ModelSourceRegistry,
  ModelSpecializationChat,
  type DownloaderStream,
} from "@trymirai/uzu";
import { abortable } from "../src/abort.js";
import { prepareModel } from "../src/download.js";

const model = Model.external(
  "downloadable",
  "test",
  "Test",
  "uzu",
  "Uzu",
  "1",
  [new ModelSpecializationChat()],
  new ModelAccessibilityOnDevice(
    new ModelSourceRegistry("1", undefined, undefined, []),
  ),
);
const info = new ChatModelInfo(
  65536,
  new ChatModelCapabilities(false, false, [], true, true, false),
);

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function fixture() {
  let state = new DownloadState(100, 0, new DownloadPhaseNotDownloaded());
  let updates: DownloadState[] = [];
  const events: string[] = [];
  const progress: (string | undefined)[] = [];
  let start = async () => {};
  let next = async () => updates.shift() ?? null;
  const stream: DownloaderStream = {
    async next() {
      const update = await next();
      if (update) state = update;
      return update;
    },
    async *[Symbol.asyncIterator]() {
      let update;
      while ((update = await this.next()) !== null) yield update;
    },
  };
  const downloader = {
    get state() {
      return Promise.resolve(state);
    },
    async resume() {
      throw new Error("unused");
    },
    async pause() {
      events.push("pause");
      state = new DownloadState(
        100,
        state.downloadedBytes,
        new DownloadPhasePaused(),
      );
    },
    async delete() {
      throw new Error("unused");
    },
    async progress() {
      return stream;
    },
  };
  const engine = {
    async modelPath() {
      return state.phase instanceof DownloadPhaseDownloaded ? "/model" : null;
    },
    downloader() {
      return downloader;
    },
    async download() {
      events.push("start");
      await start();
      events.push("started");
      return stream;
    },
    async chatModelInfo() {
      events.push("metadata");
      assert.ok(state.phase instanceof DownloadPhaseDownloaded);
      return info;
    },
  };
  return {
    engine,
    events,
    progress,
    run: (signal?: AbortSignal) =>
      prepareModel(engine, model, signal, (message) => progress.push(message)),
    setState(value: DownloadState) {
      state = value;
    },
    setUpdates(value: DownloadState[]) {
      updates = value;
    },
    setStart(value: typeof start) {
      start = value;
    },
    setNext(value: typeof next) {
      next = value;
    },
  };
}

test("downloads through the library and reads metadata only after completion", async () => {
  const f = fixture();
  f.setUpdates([
    new DownloadState(100, 25, new DownloadPhaseDownloading()),
    new DownloadState(100, 100, new DownloadPhaseDownloaded()),
  ]);
  assert.equal(await f.run(), info);
  assert.deepEqual(f.events, ["start", "started", "metadata"]);
  assert.deepEqual(f.progress, [
    `Downloading ${model.name}`,
    `Downloading ${model.name}: 25%`,
    `Downloading ${model.name}: 100%`,
    undefined,
  ]);
  f.events.length = 0;
  assert.equal(await f.run(), info);
  assert.deepEqual(f.events, ["metadata"]);
});

test("a download that finishes during startup still succeeds with an empty progress stream", async () => {
  const f = fixture();
  f.setStart(async () =>
    f.setState(new DownloadState(100, 100, new DownloadPhaseDownloaded())),
  );
  assert.equal(await f.run(), info);
});

test("failed and incomplete downloads never proceed to metadata or inference", async () => {
  for (const failure of [
    new DownloadPhaseError("network failed"),
    new DownloadPhasePaused(),
    new DownloadPhaseNotDownloaded(),
  ]) {
    const f = fixture();
    f.setUpdates([new DownloadState(100, 25, failure)]);
    await assert.rejects(
      f.run(),
      /network failed|Download paused|Download did not complete/,
    );
    assert.deepEqual(f.events, ["start", "started", "pause"]);
    assert.equal(f.progress.at(-1), undefined);
  }
  const f = fixture();
  f.setStart(async () =>
    f.setState(
      new DownloadState(100, 0, new DownloadPhaseError("startup failed")),
    ),
  );
  await assert.rejects(f.run(), /startup failed/);
});

test("cancellation pauses a pending download and a later request resumes it", async () => {
  const f = fixture();
  const waiting = deferred<void>();
  const pending = deferred<DownloadState | null>();
  f.setNext(async () => {
    waiting.resolve();
    return pending.promise;
  });
  const controller = new AbortController();
  const request = f.run(controller.signal);
  await waiting.promise;
  controller.abort();
  await assert.rejects(request, { name: "AbortError" });
  assert.deepEqual(f.events, ["start", "started", "pause"]);
  assert.equal(f.progress.at(-1), undefined);
  pending.resolve(null);
  let done = false;
  f.setNext(async () => {
    if (done) return null;
    done = true;
    return new DownloadState(100, 100, new DownloadPhaseDownloaded());
  });
  assert.equal(await f.run(), info);
  assert.deepEqual(f.events, [
    "start",
    "started",
    "pause",
    "start",
    "started",
    "metadata",
  ]);
});

test("cancellation waits for startup before pausing, so resume cannot restart a cancelled download", async () => {
  const f = fixture();
  const starting = deferred<void>();
  const started = deferred<void>();
  f.setStart(async () => {
    starting.resolve();
    await started.promise;
  });
  const controller = new AbortController();
  const request = f.run(controller.signal);
  await starting.promise;
  controller.abort();
  assert.deepEqual(f.events, ["start"]);
  started.resolve();
  await assert.rejects(request, { name: "AbortError" });
  assert.deepEqual(f.events, ["start", "started", "pause"]);
});

test("a download owned by another process is never paused", async () => {
  const locked = new DownloadState(100, 0, new DownloadPhaseLocked("other"));
  const f = fixture();
  f.setState(locked);
  await assert.rejects(f.run(), /Another process/);
  assert.deepEqual(f.events, []);

  const racing = fixture();
  racing.setUpdates([locked]);
  await assert.rejects(racing.run(), /Another process/);
  assert.deepEqual(racing.events, ["start", "started"]);
});

test("an already cancelled request starts no download and still observes pending rejection", async () => {
  const signal = AbortSignal.abort();
  const f = fixture();
  await assert.rejects(f.run(signal), { name: "AbortError" });
  assert.deepEqual(f.events, []);
  await assert.rejects(
    abortable(Promise.reject(new Error("late failure")), signal),
    { name: "AbortError" },
  );
  await assert.rejects(abortable(Promise.resolve(), signal), {
    name: "AbortError",
  });
});
