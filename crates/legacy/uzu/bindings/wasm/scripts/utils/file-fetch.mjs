// The package entry fetches `uzu_bg.wasm` next to the bindings, but Node's fetch can't read file:
// URLs. Prepend this to scripts that import the package in Node.
import { readFile } from 'node:fs/promises';

const networkFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    if (url.protocol !== 'file:') {
        return networkFetch(input, init);
    }
    return new Response(await readFile(url), { headers: { 'Content-Type': 'application/wasm' } });
};
