/**
 * Worker-to-worker WebGPU bridge. Only the inference worker blocks; this
 * module's GPU worker always keeps its event loop available to WebGPU.
 *
 * Buffers and compiled pipelines persist across submissions. CPU data is sent
 * only after CPU access, uniforms are packed once per batch, and GPU results
 * are read back on demand. Small written buffers also receive a bounded snapshot
 * in their producing submission, before later dependent work reaches the queue.
 */
const contexts = new Map();
const preparedContexts = [];
let nextContext = 1;
const HEADER_BYTES = 16;
const ERROR_BYTES = 8192;
const WAIT_TIMEOUT_MS = 120_000;
const SNAPSHOT_MAX_BYTES = 64;
const SNAPSHOT_MAX_BUFFERS = 128;
const WEIGHT_CHUNK_BYTES = 4 * 1024 * 1024;
let weightReader;

export function checkLoadCancellation(flag) {
    if (flag === undefined || flag === null) return;
    if (!(flag instanceof Int32Array) || !(flag.buffer instanceof SharedArrayBuffer) || flag.length < 1) {
        throw new Error("loadCancellation must be an Int32Array backed by SharedArrayBuffer");
    }
    if (Atomics.load(flag, 0) !== 0) throw new Error("Model loading cancelled");
}

export function weightSourceSize(source) {
    const size = source instanceof Blob ? source.size : source instanceof Uint8Array ? source.byteLength : NaN;
    if (!Number.isSafeInteger(size) || size <= 0) {
        throw new Error("weights must be a nonempty Blob or Uint8Array with a safe integer byte length");
    }
    return size;
}

function weightRange(source, offset, size) {
    const length = weightSourceSize(source);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 ||
        !Number.isSafeInteger(offset + size) || offset + size > length) {
        throw new Error("Weights read extends beyond the source");
    }
    const bytes = source instanceof Blob
        ? new Uint8Array((weightReader ??= new FileReaderSync()).readAsArrayBuffer(source.slice(offset, offset + size)))
        : source.subarray(offset, offset + size);
    if (bytes.byteLength !== size) throw new Error("Truncated weights read");
    return bytes;
}

export function readWeightSource(source, offset, destination, cancellation) {
    checkLoadCancellation(cancellation);
    // Header/CPU tensor reads also stay bounded; no whole checkpoint enters WASM.
    for (let position = 0; position < destination.byteLength; position += WEIGHT_CHUNK_BYTES) {
        checkLoadCancellation(cancellation);
        const size = Math.min(WEIGHT_CHUNK_BYTES, destination.byteLength - position);
        destination.set(weightRange(source, offset + position, size), position);
        checkLoadCancellation(cancellation);
    }
}

function signalBuffer(payloadBytes = ERROR_BYTES) {
    return new SharedArrayBuffer(HEADER_BYTES + Math.max(ERROR_BYTES, payloadBytes));
}

function waitSignal(signal) {
    const state = new Int32Array(signal, 0, 2);
    while (Atomics.load(state, 0) === 0) {
        if (Atomics.wait(state, 0, 0, WAIT_TIMEOUT_MS) === "timed-out") {
            throw new Error("WebGPU worker did not respond within 120 seconds. The device may have been lost.");
        }
    }
    if (Atomics.load(state, 0) !== 1) {
        throw new Error(new TextDecoder().decode(new Uint8Array(signal, HEADER_BYTES, state[1]).slice()));
    }
    return new Float64Array(signal, 8, 1)[0];
}

function contextFor(id) {
    const context = contexts.get(id);
    if (!context) throw new Error("WebGPU context has been disposed");
    return context;
}

function record(context, name, value) {
    if (context.profile) context.stats[name] = (context.stats[name] ?? 0) + value;
}

export function profileNow() { return performance.now(); }
export function recordHostAllocation(contextId, size) {
    const context = contextFor(contextId);
    record(context, "hostMirrorBytes", size);
    record(context, "hostMirrors", 1);
}

export function profileElapsed(contextId, name, started) {
    record(contextFor(contextId), name, performance.now() - started);
}

export function profileSnapshot(reset = false) {
    const result = [];
    for (const [id, context] of contexts) {
        if (!context.profile) continue;
        result.push({ id, ...context.stats,
            ...(context.profileDispatches ? { pipelineTimings: Array.from(context.pipelineStats, ([pipeline, values]) => ({
                pipeline,
                label: (context.pipelineLabels.get(Number(String(pipeline).split(":")[0])) ?? `pipeline-${pipeline}`)
                    + (String(pipeline).includes(":") ? ` [${String(pipeline).split(":").slice(1).join(", ")}]` : ""),
                ...values,
            })) } : {}),
        });
        if (reset) {
            context.stats = {};
            context.pipelineStats.clear();
        }
    }
    return JSON.stringify(result);
}

export async function prepareContext() {
    if (typeof document !== "undefined" || typeof WorkerGlobalScope === "undefined") {
        throw new Error("Uzu WebGPU must run in a dedicated Web Worker");
    }
    if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
        throw new Error("Uzu WebGPU requires cross-origin isolation (COOP: same-origin and COEP: require-corp)");
    }
    const id = nextContext++;
    const source = new Blob([`(${gpuWorkerMain.toString()})();`], { type: "text/javascript" });
    const url = URL.createObjectURL(source);
    const worker = new Worker(url, { type: "module", name: "uzu-webgpu" });
    const signal = signalBuffer();
    const context = {
        worker, url, nextSubmission: 1, submissions: new Map(),
        creates: new Map(), uploads: new Map(), pipelines: new Map(),
        bufferSizes: new Map(), readCaches: new Map(),
        profile: Boolean(globalThis.UZU_WEBGPU_PROFILE || globalThis.UZU_WEBGPU_PROFILE_DISPATCHES), stats: {},
        profileDispatches: Boolean(globalThis.UZU_WEBGPU_PROFILE_DISPATCHES),
        pipelineStats: new Map(), pipelineLabels: new Map(),
    };
    contexts.set(id, context);
    worker.postMessage({ type: "initialize", signal, profile: context.profile, profileDispatches: context.profileDispatches });
    try {
        // Creating a nested module worker needs an event-loop turn in Chromium.
        // Do not block the inference worker until initialization has finished.
        const started = performance.now();
        while (Atomics.load(new Int32Array(signal, 0, 2), 0) === 0) {
            if (performance.now() - started > WAIT_TIMEOUT_MS) throw new Error("WebGPU worker initialization timed out");
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        waitSignal(signal);
        const length = new Int32Array(signal, 0, 2)[1];
        const info = JSON.parse(new TextDecoder().decode(new Uint8Array(signal, HEADER_BYTES, length).slice()));
        preparedContexts.push(JSON.stringify({ id, ...info, profiling: context.profile }));
    } catch (error) {
        destroyContext(id);
        throw error;
    }
}

export function createContext() {
    const prepared = preparedContexts.shift();
    if (!prepared) throw new Error("Await WebGPU preparation before creating an engine context");
    return prepared;
}

export function destroyContext(id) {
    const context = contexts.get(id);
    if (!context) return;
    context.worker.terminate();
    URL.revokeObjectURL(context.url);
    contexts.delete(id);
}

export function createBuffer(contextId, id, size) {
    const context = contextFor(contextId);
    context.creates.set(id, size);
    context.bufferSizes.set(id, size);
    context.readCaches.delete(id);
}

export function destroyBuffer(contextId, id) {
    const context = contexts.get(contextId);
    if (!context) return;
    context.bufferSizes.delete(id);
    context.readCaches.delete(id);
    if (context.creates.delete(id)) {
        context.uploads.delete(id);
        return;
    }
    context.worker.postMessage({ type: "destroyBuffer", id });
}

export function uploadBuffer(contextId, id, bytes) {
    // Never transfer WASM memory. Copy once into a transferable upload buffer.
    const data = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
    data.set(bytes);
    const context = contextFor(contextId);
    context.uploads.set(id, data.buffer);
    context.readCaches.delete(id);
}

export function initializeBufferFromSource(contextId, id, source, sourceOffset, size, cancellation) {
    checkLoadCancellation(cancellation);
    const context = contextFor(contextId);
    const sourceSize = weightSourceSize(source);
    if (!Number.isSafeInteger(size) || size <= 0 || !Number.isSafeInteger(sourceOffset) || sourceOffset < 0 ||
        !Number.isSafeInteger(sourceOffset + size) || sourceOffset + size > sourceSize) {
        throw new Error("Invalid tensor range in weights source");
    }
    if (context.bufferSizes.has(id) || context.creates.has(id)) throw new Error("Weights buffer ID already exists");
    const physicalSize = Math.ceil(size / 4) * 4;
    const request = (message, transfers = []) => {
        const signal = signalBuffer();
        context.worker.postMessage({ ...message, signal }, transfers);
        waitSignal(signal);
    };
    try {
        checkLoadCancellation(cancellation);
        // This allocation is immediate: it must never enter pending creates,
        // where the first normal submit could recreate and erase its weights.
        request({ type: "initializeBuffer", id, size: physicalSize });
        context.bufferSizes.set(id, physicalSize);
        context.readCaches.delete(id);
        record(context, "buffersCreated", 1);
        for (let offset = 0; offset < size; offset += WEIGHT_CHUNK_BYTES) {
            checkLoadCancellation(cancellation);
            const length = Math.min(WEIGHT_CHUNK_BYTES, size - offset);
            const bytes = weightRange(source, sourceOffset + offset, length);
            checkLoadCancellation(cancellation);
            let data;
            if (source instanceof Blob && length % 4 === 0) data = bytes;
            else {
                data = new Uint8Array(Math.ceil(length / 4) * 4);
                data.set(bytes);
            }
            record(context, "streamUploadChunks", 1);
            record(context, "streamUploadBytes", data.byteLength);
            if (context.profile) context.stats.streamMaxChunkBytes = Math.max(context.stats.streamMaxChunkBytes ?? 0, data.byteLength);
            request({ type: "uploadBufferRange", id, offset, data: data.buffer }, [data.buffer]);
            checkLoadCancellation(cancellation);
            // Acknowledgement includes queue completion and error scopes. This
            // bounds transferred data and driver staging, not only JS arrays.
        }
    } catch (error) {
        context.bufferSizes.delete(id);
        context.readCaches.delete(id);
        context.worker.postMessage({ type: "destroyBuffer", id });
        throw error;
    }
}

export function readBuffer(contextId, id, destination) {
    const context = contextFor(contextId);
    const started = context.profile ? performance.now() : 0;
    const cached = context.readCaches.get(id);
    const hit = cached && destination.byteLength <= cached.size;
    const signal = hit ? cached.signal : signalBuffer(destination.byteLength);
    if (!hit) context.worker.postMessage({ type: "readBuffer", id, size: destination.byteLength, signal });
    record(context, "prefetchedReads", hit ? 1 : 0);
    waitSignal(signal);
    destination.set(new Uint8Array(signal, HEADER_BYTES, destination.byteLength));
    record(context, "readbackMs", performance.now() - started);
    record(context, "readbackBytes", destination.byteLength);
    record(context, "readbacks", 1);
}

export function registerPipeline(contextId, id, source) {
    const context = contextFor(contextId);
    context.pipelines.set(id, source);
    if (context.profileDispatches) {
        context.pipelineLabels.set(id, source.match(/\/\/\s*uzu-kernel:\s*([^\n]+)/)?.[1] ?? `pipeline-${id}`);
    }
}

// A snapshot belongs to the last submission that WRITES this buffer. A later
// read-only submission must not move its result behind unrelated queue work.
function submissionReadbacks(context, writtenBuffers) {
    // Raw bridge callers may omit access metadata. They retain demand readback
    // semantics, and must invalidate every snapshot rather than risk stale data.
    if (writtenBuffers === undefined) {
        context.readCaches.clear();
        return [];
    }
    const eligible = new Map();
    for (const id of writtenBuffers) {
        context.readCaches.delete(id);
        const size = context.bufferSizes.get(id);
        if (!(size > 0 && size <= SNAPSHOT_MAX_BYTES)) continue;
        eligible.delete(id);
        eligible.set(id, size);
        if (eligible.size > SNAPSHOT_MAX_BUFFERS) eligible.delete(eligible.keys().next().value);
    }
    // Bound allocation before creating shared signals as well as limiting the
    // retained cache and GPU copies. Evicted buffers use demand readback.
    const requests = Array.from(eligible, ([id, size]) => {
        const request = { id, size, signal: signalBuffer() };
        context.readCaches.set(id, request);
        if (context.readCaches.size > SNAPSHOT_MAX_BUFFERS) {
            context.readCaches.delete(context.readCaches.keys().next().value);
        }
        return request;
    });
    record(context, "prefetchCopies", requests.length);
    return requests;
}

export function submit(contextId, commands, writtenBuffers) {
    const context = contextFor(contextId);
    const started = context.profile ? performance.now() : 0;
    const id = context.nextSubmission++;
    const signal = signalBuffer(context.profileDispatches ? 512 * 1024 : ERROR_BYTES);
    const uploads = Array.from(context.uploads, ([id, data]) => ({ id, data }));
    // Copy borrowed WASM memory into a transferable command buffer. The JSON
    // form remains useful for the standalone bridge integration harness.
    const packed = typeof commands === "string" ? JSON.parse(commands) : new Uint32Array(commands).buffer;
    const readbacks = submissionReadbacks(context, writtenBuffers);
    const transfers = uploads.map(upload => upload.data);
    if (packed instanceof ArrayBuffer) transfers.push(packed);
    record(context, "submissions", 1);
    record(context, "buffersCreated", context.creates.size);
    record(context, "uploadBytes", uploads.reduce((sum, upload) => sum + upload.data.byteLength, 0));
    record(context, "pipelinesRegistered", context.pipelines.size);
    context.worker.postMessage({
        type: "submit", commands: packed, signal,
        creates: Array.from(context.creates, ([id, size]) => ({ id, size })),
        pipelines: Array.from(context.pipelines, ([id, source]) => ({ id, source })),
        uploads, readbacks,
    }, transfers);
    context.creates.clear();
    context.uploads.clear();
    context.pipelines.clear();
    context.submissions.set(id, signal);
    record(context, "submitParsePostMs", performance.now() - started);
    return id;
}

export function wait(contextId, submission) {
    const context = contextFor(contextId);
    const signal = context.submissions.get(submission);
    if (!signal) throw new Error("Unknown WebGPU submission");
    const started = context.profile ? performance.now() : 0;
    try {
        const duration = waitSignal(signal);
        if (context.profile) {
            record(context, "waitMs", performance.now() - started);
            const length = new Int32Array(signal, 0, 2)[1];
            if (length) {
                const values = JSON.parse(new TextDecoder().decode(new Uint8Array(signal, HEADER_BYTES, length).slice()));
                for (const [name, value] of Object.entries(values)) {
                    if (name === "pipelineTimings") {
                        for (const [pipeline, count, gpuMs] of value) {
                            const current = context.pipelineStats.get(pipeline) ?? { count: 0, gpuMs: 0 };
                            current.count += count;
                            current.gpuMs += gpuMs;
                            context.pipelineStats.set(pipeline, current);
                        }
                    } else record(context, name, value);
                }
            }
        }
        return duration;
    } finally {
        context.submissions.delete(submission);
    }
}

// This function is deliberately self-contained: wasm-bindgen copies one local
// JS module, and the nested worker needs neither a bundler path nor imports.
function gpuWorkerMain() {
    const HEADER_BYTES = 16;
    const buffers = new Map();
    const sources = new Map();
    const pipelines = new Map();
    const bindGroups = new Map();
    const uniformPool = [];
    const readbackPool = [];
    const bufferPool = new Map();
    const bufferIdentities = new WeakMap();
    const MAX_POOLED_BYTES = 32 * 1024 * 1024;
    let pooledBytes = 0;
    let nextBufferIdentity = 1;
    const activeSignals = new Set();
    let nextUniformId = 1;
    let nextTemporaryId = -10;
    let device;
    let fatalError;
    let initialization;
    let profiling = false;
    let profileDispatches = false;
    let timestampQueries = false;

    function finish(signal, error, duration = 0, payload) {
        activeSignals.delete(signal);
        const state = new Int32Array(signal, 0, 2);
        if (Atomics.load(state, 0) !== 0) return;
        if (error || payload !== undefined) {
            const text = error ? String(error?.message ?? error) : JSON.stringify(payload);
            const encoded = new TextEncoder().encode(text);
            const target = new Uint8Array(signal, HEADER_BYTES);
            const length = Math.min(encoded.byteLength, target.byteLength);
            target.set(encoded.subarray(0, length));
            state[1] = length;
        }
        new Float64Array(signal, 8, 1)[0] = duration;
        Atomics.store(state, 0, error ? -1 : 1);
        Atomics.notify(state, 0);
    }

    function failDevice(error) {
        fatalError = error;
        for (const signal of Array.from(activeSignals)) finish(signal, error);
    }

    async function initialize(signal, profile, dispatchProfile) {
        profiling = profile;
        profileDispatches = dispatchProfile;
        activeSignals.add(signal);
        try {
            if (!navigator.gpu) throw new Error("This browser does not expose WebGPU in workers");
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
            if (!adapter) throw new Error("No WebGPU adapter is available");
            const limitNames = [
                "maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage",
                "maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY",
                "maxComputeWorkgroupStorageSize", "maxUniformBufferBindingSize",
            ];
            const requiredLimits = Object.fromEntries(limitNames.map(name => [name, adapter.limits[name]]));
            const requiredFeatures = ["shader-f16", "subgroups"].filter(feature => adapter.features.has(feature));
            timestampQueries = profiling && adapter.features.has("timestamp-query");
            if (timestampQueries) requiredFeatures.push("timestamp-query");
            device = await adapter.requestDevice({ requiredLimits, requiredFeatures, label: "uzu" });
            device.lost.then(info => failDevice(new Error(`WebGPU device lost: ${info.message || info.reason}`)));
            device.addEventListener("uncapturederror", event => failDevice(event.error));
            const info = adapter.info;
            finish(signal, null, 0, {
                name: info?.description || [info?.vendor, info?.architecture].filter(Boolean).join(" ") || "WebGPU",
                maxBufferSize: Math.min(device.limits.maxBufferSize, 0xffff_fffc),
                maxStorageBufferBindingSize: Math.min(device.limits.maxStorageBufferBindingSize, 0xffff_fffc),
                supportsSubgroups: device.features.has("subgroups"),
                subgroupMinSize: info?.subgroupMinSize || 0,
                subgroupMaxSize: info?.subgroupMaxSize || 0,
                subgroupIdSupported: navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") || false,
            });
        } catch (error) {
            failDevice(error);
            finish(signal, error);
        }
    }

    function bufferFor(id) {
        const buffer = buffers.get(id);
        if (!buffer) throw new Error(`Unknown WebGPU buffer ${id}`);
        return buffer;
    }

    function bufferIdentity(buffer) {
        let identity = bufferIdentities.get(buffer);
        if (identity === undefined) {
            identity = nextBufferIdentity++;
            bufferIdentities.set(buffer, identity);
        }
        return identity;
    }

    function allocateBuffer(id, size, uploadSize, clears, metrics) {
        const available = bufferPool.get(size);
        // Partial uploads require zeroes before queue.writeBuffer. They are
        // unusual for engine buffers; a fresh allocation preserves that order
        // without an extra queue submission just to initialize pooled memory.
        if (available?.length && (uploadSize === undefined || uploadSize === size)) {
            const buffer = available.shift();
            if (!available.length) bufferPool.delete(size);
            pooledBytes -= size;
            if (uploadSize === undefined) clears.push(buffer);
            if (metrics) metrics.buffersReused++;
            return buffer;
        }
        return device.createBuffer({
            label: `uzu-buffer-${id}`, size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    function releaseBuffer(buffer) {
        if (!buffer) return;
        // Commands on the device queue retain their resources. Reuse is safe
        // immediately: the next upload/clear executes after previously queued
        // reads, even if their completion notification has not arrived yet.
        if (!fatalError && buffer.size <= 2 * 1024 * 1024 && pooledBytes + buffer.size <= MAX_POOLED_BYTES) {
            let available = bufferPool.get(buffer.size);
            if (!available) bufferPool.set(buffer.size, available = []);
            available.push(buffer);
            available.sort((a, b) => bufferIdentity(a) - bufferIdentity(b));
            pooledBytes += buffer.size;
        } else {
            forgetBuffer(buffer);
            device.queue.onSubmittedWorkDone().then(() => buffer.destroy());
        }
    }

    function pipelineFor(id) {
        let cached = pipelines.get(id);
        if (cached) return cached;
        const source = sources.get(id);
        if (!source) throw new Error(`Unknown WebGPU pipeline ${id}`);
        // Explicit layouts retain declared bindings even when the WGSL compiler
        // removes unused branches or arguments in a specialized kernel.
        const entries = [];
        const bindings = /((?:@\w+\s*(?:\([^)]*\))?\s*)+)var\s*<\s*(storage|uniform)\s*(?:,\s*(read|read_write)\s*)?>/g;
        for (const match of source.matchAll(bindings)) {
            const binding = match[1].match(/@binding\s*\(\s*(\d+)\s*\)/);
            if (!binding) continue;
            entries.push({
                binding: Number(binding[1]), visibility: GPUShaderStage.COMPUTE,
                buffer: { type: match[2] === "uniform" ? "uniform" : match[3] === "read" ? "read-only-storage" : "storage" },
            });
        }
        if (!entries.length) throw new Error("WebGPU kernel has no recognized bindings");
        const layout = device.createBindGroupLayout({ entries });
        const module = device.createShaderModule({ code: source, label: `uzu-kernel-${id}` });
        const pipeline = device.createComputePipeline({
            label: `uzu-pipeline-${id}`, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint: "main" },
        });
        const shapeWords = [];
        if (profileDispatches) {
            // Compiled Slang parameter accessors identify dynamic matrix
            // dimensions without exposing a kernel-specific dispatch API.
            for (const name of ["m", "n", "k"]) {
                const match = source.match(new RegExp(`fn\\s+uzu_param_${name}\\(\\)[^{]*\\{\\s*return\\s+p\\.data\\[(\\d+)\\]\\.([xyzw])`));
                if (match) shapeWords.push([name, Number(match[1]) * 4 + "xyzw".indexOf(match[2])]);
            }
        }
        cached = { pipeline, layout, shapeWords };
        pipelines.set(id, cached);
        return cached;
    }

    function uniformAllocation(size) {
        const index = uniformPool.findIndex(value => value.size >= size);
        if (index >= 0) return uniformPool.splice(index, 1)[0];
        const allocationSize = Math.max(256, 2 ** Math.ceil(Math.log2(size)));
        return {
            id: nextUniformId++, size: allocationSize,
            buffer: device.createBuffer({ size: allocationSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        };
    }

    function readbackAllocation(size) {
        const index = readbackPool.findIndex(buffer => buffer.size >= size);
        if (index >= 0) return readbackPool.splice(index, 1)[0];
        return device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }

    function releaseReadback(buffer) {
        if (!buffer) return;
        // Validation/device errors can reject while a concurrent map is still
        // pending. Such a buffer cannot re-enter the reusable staging pool.
        if (buffer.mapState === "pending") {
            buffer.destroy();
            return;
        }
        if (buffer.mapState === "mapped") buffer.unmap();
        if (!fatalError && readbackPool.length < 4) readbackPool.push(buffer);
        else buffer.destroy();
    }

    function bindGroupFor(pipelineId, bindingIds, uniforms, offset, size, metrics) {
        const physicalIds = bindingIds.map(id => bufferIdentity(bufferFor(id)));
        const key = `${pipelineId}:${physicalIds.join(",")}:${uniforms.id}:${offset}:${size}`;
        const cached = bindGroups.get(key);
        if (cached) {
            if (metrics) metrics.bindGroupHits++;
            return cached.group;
        }
        if (metrics) metrics.bindGroupsCreated++;
        const { layout } = pipelineFor(pipelineId);
        const group = device.createBindGroup({
            layout,
            entries: [
                ...Array.from(bindingIds, (id, binding) => ({ binding, resource: { buffer: bufferFor(id) } })),
                { binding: bindingIds.length, resource: { buffer: uniforms.buffer, offset, size } },
            ],
        });
        if (bindGroups.size >= 4096) bindGroups.delete(bindGroups.keys().next().value);
        bindGroups.set(key, { group, physicalIds, uniformId: uniforms.id });
        return group;
    }

    function forgetBuffer(buffer) {
        const id = bufferIdentity(buffer);
        for (const [key, entry] of bindGroups) {
            if (entry.physicalIds.includes(id)) bindGroups.delete(key);
        }
    }

    function forgetUniform(id) {
        for (const [key, entry] of bindGroups) {
            if (entry.uniformId === id) bindGroups.delete(key);
        }
    }

    const COPY_PIPELINE = -1;
    const FILL_PIPELINE = -2;
    sources.set(COPY_PIPELINE, `
        @group(0) @binding(0) var<storage, read> source: array<u32>;
        @group(0) @binding(1) var<storage, read_write> destination: array<u32>;
        struct Parameters { sourceOffset: u32, destinationOffset: u32, size: u32, pad: u32 }
        @group(0) @binding(2) var<uniform> p: Parameters;
        @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let word = p.destinationOffset / 4u + gid.x;
            if (word * 4u >= p.destinationOffset + p.size) { return; }
            var value = destination[word];
            for (var lane = 0u; lane < 4u; lane++) {
                let outputByte = word * 4u + lane;
                if (outputByte >= p.destinationOffset && outputByte < p.destinationOffset + p.size) {
                    let inputByte = p.sourceOffset + outputByte - p.destinationOffset;
                    let byte = (source[inputByte / 4u] >> ((inputByte % 4u) * 8u)) & 255u;
                    let shift = lane * 8u;
                    value = (value & ~(255u << shift)) | (byte << shift);
                }
            }
            destination[word] = value;
        }
    `);
    sources.set(FILL_PIPELINE, `
        @group(0) @binding(0) var<storage, read_write> destination: array<u32>;
        struct Parameters { offset: u32, size: u32, value: u32, pad: u32 }
        @group(0) @binding(1) var<uniform> p: Parameters;
        @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let word = p.offset / 4u + gid.x;
            if (word * 4u >= p.offset + p.size) { return; }
            var value = destination[word];
            for (var lane = 0u; lane < 4u; lane++) {
                let byte = word * 4u + lane;
                if (byte >= p.offset && byte < p.offset + p.size) {
                    let shift = lane * 8u;
                    value = (value & ~(255u << shift)) | (p.value << shift);
                }
            }
            destination[word] = value;
        }
    `);

    function decodeCommands(data) {
        if (!(data instanceof ArrayBuffer)) return data;
        const words = new Uint32Array(data);
        const commands = [];
        let offset = 0;
        const take = count => {
            if (offset + count > words.length) throw new Error("Truncated WebGPU command buffer");
            const result = words.subarray(offset, offset + count);
            offset += count;
            return result;
        };
        while (offset < words.length) {
            const operation = take(1)[0];
            if (operation === 1) {
                const [pipeline, bindingCount, parameterCount, x, y, z] = take(6);
                const bindings = take(bindingCount);
                const parameters = take(parameterCount);
                commands.push({ op: "dispatch", pipeline, bindings, parameters, workgroups: [x, y, z] });
            } else if (operation === 2) {
                const [source, source_offset, destination, destination_offset, size] = take(5);
                commands.push({ op: "copy", source, source_offset, destination, destination_offset, size });
            } else if (operation === 3) {
                const [destination, offset, size, value] = take(4);
                commands.push({ op: "fill", destination, offset, size, value });
            } else throw new Error(`Unknown packed WebGPU command ${operation}`);
        }
        return commands;
    }

    function submitBatch(message) {
        const { signal } = message;
        activeSignals.add(signal);
        for (const request of message.readbacks ?? []) activeSignals.add(request.signal);
        const cachedReadbacks = [];
        const start = performance.now();
        let uniforms;
        let querySet;
        let queryResults;
        let queryReadback;
        let queryCount = 0;
        const queryPipelines = [];
        const metrics = profiling ? {dispatches: 0, bindGroupHits: 0, bindGroupsCreated: 0, buffersReused: 0,
            workerEncodeMs: 0, workerAllocateMs: 0, workerPrepareMs: 0,
            workerQueueWaitMs: 0, gpuMs: 0, timestampBatches: 0} : null;
        const temporaryIds = [];
        let scopeOpen = true;
        const releaseResources = reusable => {
            if (uniforms) {
                if (reusable) uniformPool.push(uniforms);
                else {
                    forgetUniform(uniforms.id);
                    uniforms.buffer.destroy();
                }
            }
            for (const id of temporaryIds) {
                const buffer = buffers.get(id);
                if (buffer) {
                    forgetBuffer(buffer);
                    buffer.destroy();
                }
                buffers.delete(id);
            }
            querySet?.destroy();
            queryResults?.destroy();
            queryReadback?.destroy();
        };
        device.pushErrorScope("validation");
        try {
            const clears = [];
            const uploadSizes = new Map(message.uploads.map(({ id, data }) => [id, data.byteLength]));
            for (const { id, size } of message.creates.sort((a, b) => a.id - b.id)) {
                buffers.set(id, allocateBuffer(id, size, uploadSizes.get(id), clears, metrics));
            }
            for (const { id, source } of message.pipelines) sources.set(id, source);
            for (const { id, data } of message.uploads) device.queue.writeBuffer(bufferFor(id), 0, data);
            const allocated = metrics ? performance.now() : 0;
            if (metrics) metrics.workerAllocateMs = allocated - start;

            const prepared = decodeCommands(message.commands).map(command => {
                if (command.op === "dispatch") return { ...command, uniformBytes: Math.max(16, command.parameters.length * 4) };
                if (command.op === "copy") {
                    if (command.source !== command.destination &&
                        (command.source_offset | command.destination_offset | command.size) % 4 === 0) return command;
                    let source = command.source;
                    let sourceOffset = command.source_offset;
                    let snapshot;
                    if (source === command.destination) {
                        const alignedOffset = Math.floor(sourceOffset / 4) * 4;
                        const size = Math.ceil((sourceOffset % 4 + command.size) / 4) * 4;
                        source = nextTemporaryId--;
                        buffers.set(source, device.createBuffer({
                            size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                        }));
                        temporaryIds.push(source);
                        snapshot = { source: command.source, sourceOffset: alignedOffset, destination: source, size };
                        sourceOffset %= 4;
                    }
                    return {
                        op: "dispatch", pipeline: COPY_PIPELINE, bindings: [source, command.destination],
                        parameters: [sourceOffset, command.destination_offset, command.size, 0], uniformBytes: 16, snapshot,
                        workgroups: [Math.ceil((command.destination_offset % 4 + command.size) / (4 * 256)), 1, 1],
                    };
                }
                if (command.op === "fill" && (command.value !== 0 || (command.offset | command.size) % 4 !== 0)) {
                    return {
                        op: "dispatch", pipeline: FILL_PIPELINE, bindings: [command.destination],
                        parameters: [command.offset, command.size, command.value, 0], uniformBytes: 16,
                        workgroups: [Math.ceil((command.offset % 4 + command.size) / (4 * 256)), 1, 1],
                    };
                }
                return command;
            });
            let uniformBytes = 0;
            for (const command of prepared) {
                if (command.op !== "dispatch") continue;
                command.uniformOffset = uniformBytes;
                uniformBytes += Math.ceil(command.uniformBytes / 256) * 256;
            }
            uniforms = uniformAllocation(Math.max(256, uniformBytes));
            const packed = new Uint32Array(Math.max(256, uniformBytes) / 4);
            for (const command of prepared) {
                if (command.op === "dispatch") packed.set(command.parameters, command.uniformOffset / 4);
            }
            device.queue.writeBuffer(uniforms.buffer, 0, packed);
            if (metrics) metrics.workerPrepareMs = performance.now() - allocated;
            const encoder = device.createCommandEncoder({ label: "uzu-batch" });
            for (const buffer of clears) encoder.clearBuffer(buffer);
            if (timestampQueries) {
                const capacity = Math.min(8192, Math.max(2, prepared.filter(command => command.op === "dispatch").length * 2));
                querySet = device.createQuerySet({ type: "timestamp", count: capacity });
                queryResults = device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
                queryReadback = device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            }
            let pass;
            const endPass = () => { if (pass) { pass.end(); pass = undefined; } };
            for (const command of prepared) {
                if (command.snapshot) {
                    endPass();
                    const snapshot = command.snapshot;
                    encoder.copyBufferToBuffer(bufferFor(snapshot.source), snapshot.sourceOffset,
                        bufferFor(snapshot.destination), 0, snapshot.size);
                }
                if (command.op === "dispatch") {
                    if (!pass) {
                        const descriptor = querySet && queryCount + 2 <= querySet.count
                            ? { timestampWrites: {querySet, beginningOfPassWriteIndex: queryCount++, endOfPassWriteIndex: queryCount++} }
                            : {};
                        if (descriptor.timestampWrites) {
                            const shape = profileDispatches ? pipelineFor(command.pipeline).shapeWords : [];
                            queryPipelines.push(shape.length >= 2
                                ? `${command.pipeline}:${shape.map(([name, word]) => `${name}=${command.parameters[word]}`).join(":")}`
                                : command.pipeline);
                        }
                        pass = encoder.beginComputePass(descriptor);
                    }
                    if (metrics) metrics.dispatches++;
                    pass.setPipeline(pipelineFor(command.pipeline).pipeline);
                    pass.setBindGroup(0, bindGroupFor(command.pipeline, command.bindings, uniforms,
                        command.uniformOffset, command.uniformBytes, metrics));
                    pass.dispatchWorkgroups(...command.workgroups);
                    // Detailed profiling deliberately splits passes. Use its
                    // attribution to locate costly kernels, not as a throughput
                    // benchmark: pass boundaries can add device overhead.
                    if (profileDispatches) endPass();
                } else if (command.op === "copy") {
                    endPass();
                    encoder.copyBufferToBuffer(bufferFor(command.source), command.source_offset,
                        bufferFor(command.destination), command.destination_offset, command.size);
                } else if (command.op === "fill") {
                    endPass();
                    encoder.clearBuffer(bufferFor(command.destination), command.offset, command.size);
                } else {
                    throw new Error(`Unknown WebGPU command ${command.op}`);
                }
            }
            endPass();
            for (const request of message.readbacks ?? []) {
                const staging = readbackAllocation(Math.ceil(request.size / 4) * 4);
                cachedReadbacks.push({ ...request, staging });
                encoder.copyBufferToBuffer(bufferFor(request.id), 0, staging, 0, Math.ceil(request.size / 4) * 4);
            }
            if (queryCount) {
                encoder.resolveQuerySet(querySet, 0, queryCount, queryResults, 0);
                encoder.copyBufferToBuffer(queryResults, 0, queryReadback, 0, queryCount * 8);
            }
            device.queue.submit([encoder.finish()]);
            const submitted = performance.now();
            if (metrics) metrics.workerEncodeMs = submitted - start;
            const validation = device.popErrorScope();
            scopeOpen = false;
            const queueDone = device.queue.onSubmittedWorkDone().then(() => {
                if (metrics) metrics.workerQueueWaitMs = performance.now() - submitted;
            });
            const timestamps = queryCount ? queryReadback.mapAsync(GPUMapMode.READ).then(() => {
                const values = new BigUint64Array(queryReadback.getMappedRange(), 0, queryCount);
                const byPipeline = new Map();
                for (let i = 0; i < queryCount; i += 2) {
                    const elapsed = Number(values[i + 1] - values[i]) / 1e6;
                    metrics.gpuMs += elapsed;
                    if (profileDispatches) {
                        const pipeline = queryPipelines[i / 2];
                        const current = byPipeline.get(pipeline) ?? [0, 0];
                        current[0]++;
                        current[1] += elapsed;
                        byPipeline.set(pipeline, current);
                    }
                }
                if (profileDispatches) metrics.pipelineTimings = Array.from(byPipeline, ([pipeline, [count, elapsed]]) => [pipeline, count, elapsed]);
                metrics.timestampBatches++;
                queryReadback.unmap();
            }) : Promise.resolve();
            const readbacks = Promise.all(cachedReadbacks.map(async request => {
                try {
                    const [, error] = await Promise.all([request.staging.mapAsync(GPUMapMode.READ), validation]);
                    if (error) throw error;
                    if (Atomics.load(new Int32Array(request.signal, 0, 2), 0) !== 0) return;
                    new Uint8Array(request.signal, HEADER_BYTES, request.size)
                        .set(new Uint8Array(request.staging.getMappedRange(), 0, request.size));
                    finish(request.signal);
                } catch (error) {
                    finish(request.signal, error);
                    throw error;
                } finally {
                    releaseReadback(request.staging);
                }
            }));
            Promise.all([queueDone, validation, timestamps, readbacks]).then(([, error]) => {
                releaseResources(true);
                finish(signal, error, performance.now() - start, metrics ?? undefined);
            }, error => {
                releaseResources(false);
                finish(signal, error);
            });
        } catch (error) {
            if (scopeOpen) device.popErrorScope().catch(() => {});
            releaseResources(false);
            for (const request of message.readbacks ?? []) finish(request.signal, error);
            for (const request of cachedReadbacks) releaseReadback(request.staging);
            finish(signal, error);
        }
    }

    async function readBuffer(message) {
        activeSignals.add(message.signal);
        let staging;
        let scopeOpen = true;
        device.pushErrorScope("validation");
        try {
            const size = Math.ceil(message.size / 4) * 4;
            staging = readbackAllocation(size);
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(bufferFor(message.id), 0, staging, 0, size);
            device.queue.submit([encoder.finish()]);
            const validation = device.popErrorScope();
            scopeOpen = false;
            await staging.mapAsync(GPUMapMode.READ);
            const error = await validation;
            if (error) throw error;
            new Uint8Array(message.signal, HEADER_BYTES, message.size)
                .set(new Uint8Array(staging.getMappedRange(), 0, message.size));
            staging.unmap();
            finish(message.signal);
        } catch (error) {
            finish(message.signal, error);
        } finally {
            if (scopeOpen) await device.popErrorScope().catch(() => {});
            releaseReadback(staging);
        }
    }

    async function initializeOrUpload(message) {
        activeSignals.add(message.signal);
        let validationOpen = true;
        let memoryOpen = true;
        let failure;
        device.pushErrorScope("out-of-memory");
        device.pushErrorScope("validation");
        try {
            if (message.type === "initializeBuffer") {
                if (buffers.has(message.id)) throw new Error("Weights buffer ID already exists");
                const buffer = device.createBuffer({
                    label: `uzu-weights-${message.id}`, size: message.size,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });
                buffers.set(message.id, buffer);
            } else {
                device.queue.writeBuffer(bufferFor(message.id), message.offset, message.data);
            }
            const validation = device.popErrorScope();
            validationOpen = false;
            const memory = device.popErrorScope();
            memoryOpen = false;
            const [invalid, exhausted] = await Promise.all([validation, memory, device.queue.onSubmittedWorkDone()]);
            if (invalid || exhausted) throw invalid || exhausted;
        } catch (error) {
            failure = error;
            // Failed initializers never escape to Rust. Do not retain or pool
            // their partially initialized buffers, even when failure is local.
            const buffer = buffers.get(message.id);
            buffers.delete(message.id);
            if (buffer) { forgetBuffer(buffer); buffer.destroy(); }
        } finally {
            const remaining = [];
            if (validationOpen) remaining.push(device.popErrorScope());
            if (memoryOpen) remaining.push(device.popErrorScope());
            await Promise.allSettled(remaining);
            finish(message.signal, failure);
        }
    }

    self.onmessage = event => {
        const message = event.data;
        if (message.type === "initialize") {
            initialization = initialize(message.signal, message.profile, message.profileDispatches);
            return;
        }
        initialization.then(() => {
            if (fatalError) {
                if (message.signal) finish(message.signal, fatalError);
                for (const request of message.readbacks ?? []) finish(request.signal, fatalError);
                return;
            }
            if (message.type === "submit") submitBatch(message);
            else if (message.type === "initializeBuffer" || message.type === "uploadBufferRange") void initializeOrUpload(message);
            else if (message.type === "readBuffer") void readBuffer(message);
            else if (message.type === "destroyBuffer") {
                const buffer = buffers.get(message.id);
                buffers.delete(message.id);
                releaseBuffer(buffer);
            }
        }).catch(error => {
            if (message.signal) finish(message.signal, error);
            else failDevice(error);
            for (const request of message.readbacks ?? []) finish(request.signal, error);
        });
    };
}
