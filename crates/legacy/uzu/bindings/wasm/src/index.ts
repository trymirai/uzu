import init, * as bindings from './wasm/uzu.js';

// Like @trymirai/uzu, the package is ready once imported: this loads `uzu_bg.wasm` from next to the
// bindings. To load it from elsewhere, import `@trymirai/uzu-wasm/wasm/uzu.js` and call its `init`.
await init();

installAsyncIterators();

export * from './wasm/uzu.js';
export * from './tool.js';

interface Stream {
    next(): Promise<unknown>;
}

// napi streams are async iterables. wasm-bindgen can't declare `[Symbol.asyncIterator]`, so the
// bindings mark stream classes with a static `__uzuStream` getter and declare the method in the d.ts.
function installAsyncIterators(): void {
    for (const value of Object.values(bindings)) {
        if (typeof value !== 'function' || (value as { __uzuStream?: unknown }).__uzuStream !== true) {
            continue;
        }
        Object.defineProperty(value.prototype, Symbol.asyncIterator, {
            configurable: true,
            writable: true,
            value: async function* (this: Stream) {
                for (let item = await this.next(); item !== null && item !== undefined; item = await this.next()) {
                    yield item;
                }
            },
        });
    }
}
