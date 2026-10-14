/** Real-browser worker bridge integration test. Run: node crates/uzu-engine/tests/webgpu-bridge.mjs */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const chrome = process.env.CHROME_BIN ?? (process.platform === "darwin"
    ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "google-chrome");
const unsafeWebGpu = process.env.UZU_UNSAFE_WEBGPU === "1";
const forceNoSubgroups = process.env.UZU_FORCE_NO_SUBGROUPS === "1";
function bridgeCapabilities(source) {
    if (!forceNoSubgroups) return source;
    const disabled = source.replace(/supportsSubgroups:\s*device\.features\.has\("subgroups"\)/, "supportsSubgroups: false");
    assert.notEqual(disabled, source, "Test bridge subgroup override no longer matches the capability response");
    return disabled;
}
const bridge = bridgeCapabilities(await readFile(fileURLToPath(new URL("../src/backends/webgpu/bridge.js", import.meta.url)), "utf8"));
const profile = await mkdtemp(join(tmpdir(), "uzu-webgpu-test-"));
let settle;
const requests = [];
const wasmDirectory = process.env.UZU_WASM_DIR;
const modelDirectory = process.env.UZU_MODEL_DIR;
const prompt = process.env.UZU_PROMPT ?? "Say hello in one sentence.";
const tokenLimit = Number(process.env.UZU_MAX_OUTPUT_TOKENS || 8);
const profileDispatches = process.env.UZU_WEBGPU_PROFILE_DISPATCHES === "1";
const profiling = process.env.UZU_WEBGPU_PROFILE === "1" || profileDispatches;
const profileTrials = Number(process.env.UZU_BENCH_TRIALS || (profiling ? 3 : 1));
const samplingPolicy = JSON.parse(process.env.UZU_SAMPLING_POLICY ?? '{"type":"Greedy"}');
assert(samplingPolicy === null || (typeof samplingPolicy === "object" && !Array.isArray(samplingPolicy)
    && ["Default", "Greedy", "Argmax", "Stochastic"].includes(samplingPolicy.type)),
"UZU_SAMPLING_POLICY must be null or a sampling policy object");
const expectedReference = process.env.UZU_EXPECTED_TOKENS ? JSON.parse(await readFile(process.env.UZU_EXPECTED_TOKENS,"utf8")) : null;
const expectedTokens = expectedReference?.tokenIds ?? null;
const referenceBackend = expectedReference?.backend ?? "CPU";
// Some model templates rewrite previous assistant turns, requiring a fresh
// state. Assert reuse only when the caller knows the model's template behavior.
const expectedReusedTokens = process.env.UZU_EXPECT_REUSED_TOKENS ?? null;
const result = new Promise(resolve => { settle = resolve; });
const bridgeWorker = `
import * as gpu from '/bridge.js';
function equal(actual, expected, label) {
  if (JSON.stringify(Array.from(actual)) !== JSON.stringify(Array.from(expected)))
    throw new Error(label + ': ' + JSON.stringify(Array.from(actual)) + ' != ' + JSON.stringify(Array.from(expected)));
}
let context;
try {
  await gpu.prepareContext();
  context = JSON.parse(gpu.createContext());
  const id = context.id;
  const count = 257;
  gpu.createBuffer(id, 1, count * 4);
  gpu.createBuffer(id, 2, count * 4);
  const input = new Float32Array(Array.from({length: count}, (_,i) => i / 4));
  gpu.uploadBuffer(id, 1, new Uint8Array(input.buffer));
  gpu.registerPipeline(id, 1, \`
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    struct Params { count: u32, pad: vec3<u32> }
    @group(0) @binding(2) var<uniform> p: Params;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < p.count) { output[id.x] = input[id.x] * 2.0 + 1.0; }
    }
  \`);
  const dispatch = {op:'dispatch',pipeline:1,bindings:[1,2],parameters:[count,0,0,0,0,0,0,0],workgroups:[5,1,1]};
  gpu.wait(id, gpu.submit(id, JSON.stringify([dispatch])));
  const output = new Float32Array(count);
  gpu.readBuffer(id, 2, new Uint8Array(output.buffer));
  equal(output, input.map(v => v*2+1), 'compute');
  // Reuse the compiled pipeline, persistent buffers and packed uniform pool.
  gpu.wait(id, gpu.submit(id, JSON.stringify([dispatch])));
  gpu.readBuffer(id, 2, new Uint8Array(output.buffer));
  equal(output, input.map(v => v*2+1), 'cached compute');
  gpu.wait(id, gpu.submit(id, new Uint32Array([1,1,2,8,5,1,1,1,2,count,0,0,0,0,0,0,0])));
  gpu.readBuffer(id, 2, new Uint8Array(output.buffer));
  equal(output, input.map(v => v*2+1), 'packed compute');
  gpu.createBuffer(id, 3, 20);
  gpu.createBuffer(id, 4, 20);
  const bytes = Uint8Array.from({length:20}, (_,i) => i);
  gpu.uploadBuffer(id, 3, bytes);
  gpu.uploadBuffer(id, 4, new Uint8Array(20).fill(99));
  gpu.wait(id, gpu.submit(id, JSON.stringify([
    {op:'copy',source:3,source_offset:3,destination:4,destination_offset:1,size:13},
    {op:'fill',destination:4,offset:4,size:5,value:171},
    {op:'fill',destination:4,offset:16,size:4,value:0},
  ])));
  const copied = new Uint8Array(20);
  gpu.readBuffer(id, 4, copied);
  const expected = new Uint8Array(20).fill(99);
  expected.set(bytes.subarray(3,16),1); expected.fill(171,4,9); expected.fill(0,16,20);
  equal(copied, expected, 'unaligned byte copy/fill');
  gpu.wait(id, gpu.submit(id, JSON.stringify([
    {op:'copy',source:4,source_offset:1,destination:4,destination_offset:3,size:13},
  ])));
  expected.copyWithin(3,1,14);
  gpu.readBuffer(id, 4, copied);
  equal(copied, expected, 'overlapping copy');
  // A logical allocation may be released while earlier queue commands still
  // reference its physical buffer. Recycling must preserve those reads and
  // initialize the next owner independently of the previous contents.
  gpu.createBuffer(id, 20, 20);
  gpu.createBuffer(id, 22, 20);
  gpu.uploadBuffer(id, 20, new Uint8Array(20).fill(91));
  const first = gpu.submit(id, JSON.stringify([{op:'fill',destination:20,offset:0,size:20,value:17}]));
  const second = gpu.submit(id, JSON.stringify([{op:'copy',source:20,source_offset:0,destination:22,destination_offset:0,size:20}]));
  gpu.destroyBuffer(id, 20);
  gpu.createBuffer(id, 21, 20);
  const third = gpu.submit(id, '[]');
  gpu.wait(id, first); gpu.wait(id, second); gpu.wait(id, third);
  const recycled = new Uint8Array(20);
  gpu.readBuffer(id, 22, recycled);
  equal(recycled, new Uint8Array(20).fill(17), 'queued read before recycle');
  gpu.readBuffer(id, 21, recycled);
  equal(recycled, new Uint8Array(20), 'recycled allocation initialized');
  gpu.destroyBuffer(id, 21);
  gpu.createBuffer(id, 23, 20);
  gpu.uploadBuffer(id, 23, new Uint8Array(8).fill(42));
  gpu.wait(id, gpu.submit(id, '[]'));
  gpu.readBuffer(id, 23, recycled);
  const partial = new Uint8Array(20); partial.fill(42,0,8);
  equal(recycled, partial, 'partial upload initializes remaining bytes');
  gpu.wait(id, gpu.submit(id, new Uint32Array([3,23,4,4,11,2,23,0,22,0,20])));
  gpu.readBuffer(id, 22, recycled);
  partial.fill(11,4,8);
  equal(recycled, partial, 'packed copy/fill');
  let malformed = false;
  try { gpu.wait(id, gpu.submit(id, new Uint32Array([1,1,2,8,5,1,1,1]))); }
  catch { malformed = true; }
  if (!malformed) throw new Error('Truncated packed command was not rejected');
  // Validation errors must wake a blocked engine worker and preserve the device.
  gpu.registerPipeline(id, 99, '@group(0) @binding(0) var<storage, read> invalid: array<u32>; broken WGSL');
  let rejected = false;
  try { gpu.wait(id, gpu.submit(id, JSON.stringify([{op:'dispatch',pipeline:99,bindings:[3],parameters:[0,0,0,0],workgroups:[1,1,1]}]))); }
  catch { rejected = true; }
  if (!rejected) throw new Error('invalid WGSL was not reported');
  gpu.wait(id, gpu.submit(id, JSON.stringify([dispatch])));
  rejected = false;
  try { gpu.readBuffer(id, 999, new Uint8Array(4)); } catch { rejected = true; }
  if (!rejected) throw new Error('Unknown readback buffer was not rejected');
  gpu.readBuffer(id, 2, new Uint8Array(output.buffer));
  equal(output, input.map(v => v*2+1), 'readback after invalid buffer');
  postMessage({ok:true, device:context.name, capabilities:{subgroups:context.supportsSubgroups,minSize:context.subgroupMinSize,maxSize:context.subgroupMaxSize,subgroupId:context.subgroupIdSupported}, checks:['compute','pipeline cache','packed command transport','byte copy/fill','overlapping copy','queued buffer recycling','partial upload initialization','validation recovery','readback error recovery']});
} catch (error) { postMessage({ok:false,error:String(error?.stack ?? error)}); }
finally { if(context) gpu.destroyContext(context.id); }
`;

const wasmWorker = `
import init, * as wasm from '/wasm/uzu_web.js';
const checks = [];
globalThis.UZU_WEBGPU_PROFILE = ${profiling};
globalThis.UZU_WEBGPU_PROFILE_DISPATCHES = ${profileDispatches};
try {
  const wasmExports = await init();
  if (wasm.testWebGpuKernels && !${profiling}) {
    checks.push(JSON.parse(await wasm.testWebGpuKernels()));
  }
  if (${Boolean(modelDirectory)}) {
    const read = async name => new Uint8Array(await (await fetch('/model/'+name)).arrayBuffer());
    const config = await read('config.json');
    const tokenizer = await read('tokenizer.json');
    const downloadedWeights = await (await fetch('/model/model.safetensors')).blob();
    const weightParts = [];
    for (let offset = 0; offset < downloadedWeights.size; offset += 4 * 1024 * 1024)
      weightParts.push(downloadedWeights.slice(offset, offset + 4 * 1024 * 1024));
    const weights = new Blob(weightParts);
    const memoryBeforeLoad = wasmExports.memory.buffer.byteLength;
    const started = performance.now();
    const session = await wasm.createSession({config, tokenizer, weights, contextLength:256, seed:42});
    const loading = {weightsSource:'composite Blob',weightsBytes:weights.size,wasmMemoryBefore:memoryBeforeLoad,wasmMemoryAfter:wasmExports.memory.buffer.byteLength};
    const params = {repoId:'test', messages:[{role:'user',content:${JSON.stringify(prompt)}}],...(${JSON.stringify(samplingPolicy)} === null ? {} : {samplingPolicy:${JSON.stringify(samplingPolicy)}}),maxOutputTokens:${tokenLimit}};
    let done;
    await session.run({...params,maxOutputTokens:1},()=>{});
    const profiles = [];
    for (let trial = 0; trial < ${profileTrials}; trial++) {
      if (${profiling}) wasm.webGpuProfile(true);
      const runStarted = performance.now();
      await session.run(params, event => {if (event.type === 'done') done=event;});
      if (${profiling || profileTrials > 1}) profiles.push({trial:trial+1, elapsedMs:performance.now()-runStarted, stats:done?.stats, counters:${profiling} ? JSON.parse(wasm.webGpuProfile(false)) : undefined});
    }
    if (!done?.text?.trim()) throw new Error('Model returned no response: '+JSON.stringify(done));
    const tokens = session.debugTokenIds ? Array.from(session.debugTokenIds()) : null;
    const expected = ${JSON.stringify(expectedTokens)};
    const tokensMatch = !expected || JSON.stringify(tokens) === JSON.stringify(expected);
    checks.push({name:'warm model generation',loading,samplingPolicy:${JSON.stringify(samplingPolicy)},text:done.text,stats:done.stats,elapsedSeconds:(performance.now()-started)/1000,tokenIds:tokens,parity:expected ? (tokensMatch ? 'all tokens match ' : 'token mismatch against ')+${JSON.stringify(referenceBackend)} : undefined,profiles:profiles.length ? profiles : undefined});
    if (!tokensMatch) throw new Error(${JSON.stringify(referenceBackend)}+'/WebGPU token mismatch: '+JSON.stringify({actual:tokens,expected}));
    const nextTurn = {...params,messages:[...params.messages,{role:'assistant',content:done.text},{role:'user',content:'What makes sunsets red?'}],maxOutputTokens:4};
    await session.run(nextTurn,event=>{if(event.type==='done')done=event;});
    if (!done?.text?.trim()) throw new Error('Second turn returned no text');
    const reused = session.debugReusedTokens?.();
    const expectedReuse = ${JSON.stringify(expectedReusedTokens)};
    if (expectedReuse !== null && (reused === undefined || (expectedReuse === 'positive' ? reused === 0 : reused !== Number(expectedReuse))))
      throw new Error('Unexpected prompt cache reuse: '+JSON.stringify({actual:reused,expected:expectedReuse}));
    checks.push({name:'second turn',reusedTokens:reused,text:done.text});
    let scheduled = false;
    let concurrent;
    await session.run({...params,maxOutputTokens:100}, event => {
      if ((event.type === 'chunk' || event.type === 'transcriptDelta') && !scheduled) {
        scheduled=true;
        concurrent=session.run(params,()=>{}).then(()=>false,error=>String(error).includes('already running'));
        setTimeout(()=>session.cancel(),0);
      }
      if (event.type === 'done') done=event;
    });
    if (done?.finishReason !== 'Cancelled') throw new Error('Cancellation did not interrupt decoding: '+JSON.stringify(done));
    if (!await concurrent) throw new Error('Concurrent generation was not rejected');
    checks.push({name:'cooperative cancellation',tokens:done.stats.totalStats.tokensCountOutput});
    checks.push('concurrent generation rejected');
    let callbackRejected = false;
    try {
      await session.run(params, event => {if (event.type === 'chunk' || event.type === 'transcriptDelta') throw new Error('intentional callback failure');});
    } catch (error) {callbackRejected=String(error).includes('intentional callback failure');}
    if (!callbackRejected) throw new Error('Callback error was not propagated');
    await session.run({...params,maxOutputTokens:4},event=>{if(event.type==='done')done=event;});
    if (done?.finishReason === 'Cancelled' || !done?.text?.trim()) throw new Error('Session did not recover after cancellation');
    checks.push('cancellation and callback-error recovery');
    scheduled=false;
    await session.run({...params,maxOutputTokens:100}, event => {
      if ((event.type === 'chunk' || event.type === 'transcriptDelta') && !scheduled) {scheduled=true;setTimeout(()=>session.dispose(),0);}
      if (event.type === 'done') done=event;
    });
    if (done?.finishReason !== 'Cancelled') throw new Error('Disposal did not cancel active generation');
    let rejected = false;
    try {await session.run(params,()=>{});} catch {rejected=true;}
    if (!rejected) throw new Error('Disposed session accepted a generation');
    session.dispose();
    checks.push('dispose during generation');
  }
  if (!checks.length) throw new Error('Build with --features kernel-tests or set UZU_MODEL_DIR');
  postMessage({ok:true,checks});
} catch(error) {postMessage({ok:false,checks,error:String(error?.stack ?? error)});}
`;
const worker = process.env.UZU_WORKER_SCRIPT ? await readFile(process.env.UZU_WORKER_SCRIPT, "utf8") : wasmDirectory ? wasmWorker : bridgeWorker;

const server = createServer(async (request, response) => {
    requests.push(request.url);
    response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    if ((request.url.startsWith('/wasm/') && wasmDirectory) || (request.url.startsWith('/model/') && modelDirectory)) {
        const root = request.url.startsWith('/wasm/') ? wasmDirectory : modelDirectory;
        const relative = request.url.split('/').slice(2).join('/');
        if (relative.split('/').includes('..')) {response.writeHead(400);response.end();return;}
        response.setHeader('Content-Type',relative.endsWith('.wasm') ? 'application/wasm' : relative.endsWith('.js') ? 'application/javascript' : 'application/octet-stream');
        if (forceNoSubgroups && relative.endsWith('/backends/webgpu/bridge.js')) {
            response.end(bridgeCapabilities(await readFile(join(root, relative), 'utf8')));
            return;
        }
        const stream = createReadStream(join(root, relative));
        stream.on('error', error => {response.writeHead(404);response.end(String(error));});
        stream.pipe(response);
    } else if (request.url === "/result") {
        let text = "";
        for await (const chunk of request) text += chunk;
        settle(JSON.parse(text));
        response.end("ok");
    } else if (request.url === "/bridge.js" || request.url === "/worker.js") {
        response.setHeader("Content-Type", "application/javascript");
        response.end(request.url === "/bridge.js" ? bridge : worker);
    } else {
        response.setHeader("Content-Type", "text/html");
        response.end(`<script>const w=new Worker('/worker.js',{type:'module'});w.onmessage=e=>fetch('/result',{method:'POST',body:JSON.stringify(e.data)});w.onerror=e=>fetch('/result',{method:'POST',body:JSON.stringify({ok:false,error:e.message})});</script>`);
    }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = spawn(chrome, [
    "--headless=new", ...(unsafeWebGpu ? ["--enable-unsafe-webgpu"] : []),
    "--disable-extensions", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
    `--user-data-dir=${profile}`, `http://127.0.0.1:${server.address().port}/`,
], { stdio: ["ignore", "ignore", "pipe"] });
const browserClosed = new Promise(resolve => browser.once("close", resolve));
let diagnostics = "";
browser.stderr.on("data", data => { diagnostics = (diagnostics + data).slice(-8192); });
browser.on("error", error => settle({ok:false, error:String(error)}));
const timeout = setTimeout(() => settle({ok:false, error:"Browser test timed out", requests, diagnostics}), Number(process.env.UZU_TEST_TIMEOUT_MS || (modelDirectory ? 180_000 : 60_000)));
try {
    const outcome = await result;
    console.log(JSON.stringify({...outcome, browser: {unsafeWebGpu}}, null, 2));
    assert.equal(outcome.ok, true, outcome.error);
} finally {
    clearTimeout(timeout);
    browser.kill();
    server.close();
    await browserClosed;
    await rm(profile, {recursive:true, force:true});
}
