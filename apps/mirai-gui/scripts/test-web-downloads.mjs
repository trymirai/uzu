/* global process, console, setTimeout, clearTimeout */
// Opt-in integration test: downloads the real registry's LFM230M model into an
// isolated Chrome profile, then deletes it. No inference or GPU work is submitted.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("../", import.meta.url));
const wasmDirectory = process.env.UZU_WASM_DIR ? resolve(process.env.UZU_WASM_DIR) : undefined;
const inspectOnly = process.env.UZU_INSPECT_ONLY === "1";
if (inspectOnly && !wasmDirectory) throw new Error("UZU_INSPECT_ONLY requires UZU_WASM_DIR with the browser engine");
const profile = await mkdtemp(join(tmpdir(), "mirai-download-test-"));
let settle;
const result = new Promise((resolve) => {
  settle = resolve;
});
const page = `<!doctype html><title>Mirai remote model integration test</title><script type="module">
import '/@vite/client';
const nativeFetch = globalThis.fetch;
const requests = JSON.parse(sessionStorage.getItem('requests') || '[]');
globalThis.fetch = async (input, init) => {
  const response = await nativeFetch(input, init);
  if (String(input).includes('/resolve/')) {
    requests.push({url:String(input), range:new Headers(init?.headers).get('range'), status:response.status, type:response.type});
  }
  return response;
};
const check = (value, message) => { if (!value) throw new Error(message); };
try {
  const {browserModelManager, findBrowserModel, cachedModelFiles, setBrowserModelConfigInspector} = await import('/src/platform/services/models/browser-models.ts');
  const manager = ${Boolean(wasmDirectory)} ? (await import('/src/platform/services/models/web.ts')).webModels : browserModelManager;
  const catalog = await manager.getModels();
  check(catalog.models.length > 2, 'The live registry did not return a complete model list');
  const model = catalog.models.find(model => model.repoId === 'trymirai/LFM2.5-230M-M');
  check(model, 'The live registry is missing the LFM230M test model');
  check(model.vendorIcons?.light?.startsWith('https://'), 'Vendor metadata was not preserved');
  const icon = new Image();
  icon.crossOrigin = 'anonymous';
  icon.src = model.vendorIcons.light;
  await icon.decode();
  check(icon.naturalWidth > 0, 'Vendor icon was blocked by cross-origin isolation');
  if (${inspectOnly}) {
    check(catalog.complete && !catalog.error && !model.browserUnsupportedReason, 'Fresh browser capability inspection failed: ' + catalog.error);
    const {inspectBrowserModelConfig} = await import('/src/platform/services/chat/browser-runtime.ts');
    let invalidRejected = false;
    try { await inspectBrowserModelConfig(new TextEncoder().encode('{}')); } catch { invalidRejected = true; }
    check(invalidRejected, 'Invalid model configuration passed preflight');
    const marker = 'Intentional preflight interruption';
    let verified = false, finish;
    const failed = new Promise(resolve => { finish = resolve; });
    setBrowserModelConfigInspector(async config => {
      await inspectBrowserModelConfig(config);
      verified = true;
      throw new Error(marker);
    });
    manager.onDownloadEvent(event => { if (event.identifier === model.repoId && event.kind === 'error') finish(event.error); });
    await manager.startDownload(model.repoId);
    check(await failed === marker && verified, 'Valid config preflight or error propagation failed');
    const state = (await manager.getModels()).models.find(entry => entry.repoId === model.repoId).state;
    check(state.phase === 'Error' && state.error === marker, 'Preflight error was not reflected in model state');
    check(requests.length === 1 && requests[0].url.endsWith('/config.json'), 'Preflight downloaded files beyond config.json');
    await manager.deleteModel(model.repoId);
    await nativeFetch('/__download_result', {method:'POST',body:JSON.stringify({ok:true, catalogModels:catalog.models.length, freshWasmInspection:true, invalidConfigRejected:true, preflightError:true, filesRequested:requests.map(request => request.url), deletion:true, crossOriginIsolated})});
  } else if (!sessionStorage.getItem('resuming')) {
    let pause, rejectPause;
    const paused = new Promise((resolve,reject) => { pause = resolve; rejectPause = reject; });
    let requested = false;
    manager.onDownloadEvent(event => {
      if (event.identifier !== model.repoId) return;
      if (event.kind === 'error') rejectPause(new Error(event.error));
      if (event.kind === 'progress' && event.completedBytes > 6 * 1024 * 1024 && !requested) {
        requested = true;
        manager.pauseDownload(model.repoId).then(pause);
      }
    });
    await manager.startDownload(model.repoId);
    await paused;
    const state = (await manager.getModels()).models.find(entry => entry.repoId === model.repoId).state;
    check(state.phase === 'Paused' && state.downloadedKbytes > 1024, 'Pause did not persist a partial weight file');
    sessionStorage.setItem('pausedBytes', String(state.downloadedKbytes * 1024));
    sessionStorage.setItem('requests', JSON.stringify(requests));
    sessionStorage.setItem('resuming', '1');
    location.reload();
  } else {
    const restored = model.state;
    check(restored.phase === 'Paused' && restored.downloadedKbytes * 1024 === Number(sessionStorage.getItem('pausedBytes')), 'Reload lost partial download progress');
    let finish, fail;
    const finished = new Promise((resolve,reject) => { finish=resolve; fail=reject; });
    manager.onDownloadEvent(event => {
      if (event.identifier !== model.repoId) return;
      if (event.kind === 'done') finish();
      if (event.kind === 'error') fail(new Error(event.error));
    });
    await manager.startDownload(model.repoId);
    await finished;
    const descriptor = await findBrowserModel(model.repoId);
    const files = await cachedModelFiles(descriptor);
    check(files['model.safetensors'] instanceof Blob && files['model.safetensors'].size > 100_000_000, 'Real model weights were not opened as a cached Blob');
    check(requests.some(request => request.range?.startsWith('bytes=') && request.status === 206), 'No successful remote byte-range resume was observed');
    check(requests.every(request => request.type === 'cors'), 'Remote downloads did not use CORS');
    const lengths = Object.fromEntries(Object.entries(files).map(([name,value]) => [name,value instanceof Blob ? value.size : value.byteLength]));
    globalThis.fetch = async () => { throw new TypeError('Deliberately offline'); };
    await manager.refreshModels();
    const offline = await manager.getModels();
    check(offline.models.find(entry => entry.repoId === model.repoId).state.phase === 'Downloaded', 'Offline catalog lost the installed model');
    check((await cachedModelFiles(descriptor))['model.safetensors'].size === lengths['model.safetensors'], 'Offline file reuse failed');
    await manager.deleteModel(model.repoId);
    check((await manager.getModels()).models.find(entry => entry.repoId === model.repoId).state.phase === 'NotDownloaded', 'Deletion did not clear the model');
    const outcome = {ok:true, catalogModels:catalog.models.length, vendorIcon:true, repoId:model.repoId, pausedBytes:Number(sessionStorage.getItem('pausedBytes')), files:lengths, blobWeights:true, resumed:requests.filter(request => request.range), offlineReuse:true, deletion:true, crossOriginIsolated};
    await nativeFetch('/__download_result', {method:'POST',body:JSON.stringify(outcome)});
  }
} catch (error) {
  await nativeFetch('/__download_result', {method:'POST',body:JSON.stringify({ok:false,error:String(error?.stack ?? error),requests})});
}
</script>`;
const server = await createServer({
  configFile: join(root, "vite.config.ts"),
  mode: "web",
  logLevel: "warn",
  server: { host: "127.0.0.1", port: 0, strictPort: false },
  plugins: [
    {
      name: "model-download-test",
      configureServer(server) {
        server.middlewares.use(async (request, response, next) => {
          if (wasmDirectory && request.url?.startsWith("/wasm/")) {
            const path = resolve(
              wasmDirectory,
              decodeURIComponent(new URL(request.url, "http://localhost").pathname.slice(6)),
            );
            if (!path.startsWith(wasmDirectory + "/")) {
              response.statusCode = 403;
              response.end();
              return;
            }
            try {
              response.setHeader("Content-Type", path.endsWith(".wasm") ? "application/wasm" : "text/javascript");
              response.end(await readFile(path));
            } catch {
              response.statusCode = 404;
              response.end();
            }
          } else if (request.url === "/__download_result") {
            let body = "";
            for await (const chunk of request) body += chunk;
            settle(JSON.parse(body));
            response.end("ok");
          } else if (request.url === "/__download_test") {
            response.setHeader("Content-Type", "text/html");
            response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            response.end(page);
          } else next();
        });
      },
    },
  ],
});
await server.listen();
const port = server.httpServer.address().port;
const browser = spawn(
  process.env.CHROME_BIN ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  [
    "--headless=new",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    `--user-data-dir=${profile}`,
    `http://127.0.0.1:${port}/__download_test`,
  ],
  { stdio: ["ignore", "ignore", "pipe"] },
);
let diagnostics = "";
browser.stderr.on("data", (data) => {
  diagnostics = (diagnostics + data).slice(-4096);
});
browser.on("error", (error) => settle({ ok: false, error: String(error) }));
const closed = new Promise((resolve) => browser.once("close", resolve));
const timeout = setTimeout(() => settle({ ok: false, error: "Remote download test timed out", diagnostics }), 180_000);
try {
  const outcome = await result;
  console.log(JSON.stringify(outcome, null, 2));
  assert.equal(outcome.ok, true, outcome.error);
} finally {
  clearTimeout(timeout);
  browser.kill();
  await closed;
  await server.close();
  await rm(profile, { recursive: true, force: true });
}
