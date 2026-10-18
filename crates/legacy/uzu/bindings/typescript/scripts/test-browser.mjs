import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { mkdtemp, rm, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_DRIVER_PATH);
console.log('Chromium:', chromium.executablePath());
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../dist-wasm');
const modelPath = process.env.UZU_MODEL_PATH;
const repository = 'trymirai/Qwen3.5-0.8B-M';
const contentTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
};
const server = createServer(async (request, response) => {
    try {
        const pathname = new URL(request.url, 'http://localhost').pathname;
        if (pathname === '/favicon.ico') {
            response.writeHead(204).end();
            return;
        }
        if (pathname === '/test.html') {
            response
                .writeHead(200, { 'Content-Type': 'text/html' })
                .end('<!doctype html><title>Uzu services test</title>');
            return;
        }
        let path;
        if (modelPath && pathname.startsWith('/model/')) {
            path = resolve(modelPath, decodeURIComponent(pathname.slice('/model/'.length)));
            if (!path.startsWith(`${resolve(modelPath)}${sep}`)) throw new Error('Invalid model path');
        } else {
            path = resolve(root, `.${decodeURIComponent(pathname)}`);
            if (!path.startsWith(`${root}${sep}`)) throw new Error('Invalid path');
        }
        const info = await stat(path);
        const range = /^bytes=(\d+)-$/.exec(request.headers.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Expose-Headers': 'Content-Length, Content-Range',
            'Accept-Ranges': 'bytes',
        };
        if (start >= info.size && range) {
            response.writeHead(416, { ...headers, 'Content-Range': `bytes */${info.size}` }).end();
            return;
        }
        response.writeHead(range ? 206 : 200, {
            ...headers,
            ...(range ? { 'Content-Range': `bytes ${start}-${info.size - 1}/${info.size}` } : {}),
            'Content-Type': contentTypes[extname(path)] ?? 'application/octet-stream',
            'Content-Length': info.size - start,
        });
        createReadStream(path, { start }).pipe(response);
    } catch {
        response.writeHead(404).end();
    }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const profile = await mkdtemp(join(tmpdir(), 'uzu-browser-'));
let browser;
try {
    browser = await chromium.launchPersistentContext(profile, {
        channel: 'chromium',
        headless: true,
        args: ['--enable-unsafe-webgpu'],
        timeout: 60_000,
    });
    const origin = `http://127.0.0.1:${server.address().port}`;
    if (modelPath) {
        await browser.route(`https://huggingface.co/${repository}/resolve/**`, async (route) => {
            const path = new URL(route.request().url()).pathname
                .split('/resolve/')[1]
                .split('/')
                .slice(1)
                .join('/');
            await route.fulfill({
                status: 307,
                headers: { 'Access-Control-Allow-Origin': '*', Location: `${origin}/model/${path}` },
                body: '',
            });
        });
    }
    const page = await browser.newPage();
    const checkpointRequests = [];
    page.on('request', (request) => {
        if (new URL(request.url()).pathname.endsWith('/model.safetensors')) checkpointRequests.push(request);
    });
    page.on('console', (message) => console.log(`[browser:${message.type()}] ${message.text()}`));
    page.on('pageerror', (error) => {
        console.error(error);
        void page.close();
    });
    page.setDefaultTimeout(600_000);
    await page.goto(`${origin}/example/index.html`);
    const gpu = await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        return {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
        };
    });
    console.log('WebGPU adapter:', gpu);
    let timer;
    const result = await Promise.race([
        page.evaluate(() => window.uzuResult),
        new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error('Browser inference timed out')), 600_000);
        }),
    ]).finally(() => clearTimeout(timer));
    assert.ok(result.text && /\b4\b|\bfour\b/i.test(result.text), 'Expected the answer to contain 4');
    console.log('Browser inference passed:', JSON.stringify(result));

    const downloadsBeforeReload = checkpointRequests.length;
    await page.goto(`${origin}/test.html`);
    const cached = await page.evaluate(async (repository) => {
        const { default: init, Engine, EngineConfig, EngineCallback } = await import('/index.js');
        await init();
        const engine = await Engine.create(EngineConfig.create());
        const model = await engine.model(repository);
        if (!model) throw new Error('Cached model missing');
        const path = await engine.modelPath(model);
        if (!path) throw new Error('Cached download not ready');
        const byPath = await engine.modelByPath(path);
        if (byPath?.identifier !== model.identifier) throw new Error('Path lookup failed');
        byPath.free();
        const info = await engine.chatModelInfo(model);
        if (!info?.context_length) throw new Error('Model metadata missing');
        info.free();
        const states = await engine.downloadStates;
        if (!states.has(model.identifier)) throw new Error('Download states missing');
        for (const state of states.values()) state.free();
        let callbacks = 0;
        await engine.registerCallback(EngineCallback.create(() => callbacks++));
        const downloader = engine.downloader(model);
        window.downloadTest = { engine, model, downloader, callbacks: () => callbacks };
        return path;
    }, repository);
    assert.equal(checkpointRequests.length, downloadsBeforeReload, 'Cached model downloaded again');
    console.log('Cache reuse and shared services passed:', cached);

    await page.evaluate(async () => {
        const { engine, model, downloader, callbacks } = window.downloadTest;
        await downloader.delete();
        const stream = await engine.download(model);
        let paused = false;
        for (;;) {
            const state = await stream.next();
            if (!state) break;
            const phase = state.phase;
            const progress = state.progress;
            state.free();
            if ('Error' in phase) throw new Error(phase.Error.message);
            if (!paused && progress > 0.05 && progress < 0.9) {
                await downloader.pause();
                paused = true;
            }
        }
        stream.free();
        const state = await downloader.state;
        const phase = state?.phase;
        state?.free();
        if (!paused || !phase || !('Paused' in phase)) throw new Error('Download did not pause');
        await downloader.resume();
        const resumed = await downloader.progress();
        for (;;) {
            const state = await resumed.next();
            if (!state) break;
            const phase = state.phase;
            state.free();
            if ('Error' in phase) throw new Error(phase.Error.message);
        }
        resumed.free();
        if (!(await engine.modelPath(model))) throw new Error('Resumed download not ready');
        if (!callbacks()) throw new Error('Engine callback did not fire');
        downloader.free();
        model.free();
        engine.free();
        delete window.downloadTest;
    });
    assert.ok(
        checkpointRequests.some((request) => /^bytes=[1-9]\d*-$/.test(request.headers().range ?? '')),
        'Expected a range request when resuming',
    );
    console.log('Download pause/resume and callbacks passed');
} finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
}
