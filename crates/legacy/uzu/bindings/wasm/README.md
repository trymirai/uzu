# uzu-wasm

WebAssembly build of [uzu](https://github.com/trymirai/uzu) for the browser.

## Usage

```js
import init, { Engine, EngineConfig } from '@trymirai/uzu-wasm';

await init();
const engine = await Engine.create(EngineConfig.create());
```

`init()` fetches `uzu_bg.wasm` next to the bindings. To serve it from another location, pass its URL: `init({ module_or_path: url })`.

## Building

From the repository root:

```bash
cargo tools build wasm
```

This compiles `uzu` for `wasm32-unknown-unknown`, generates the bindings into `src/wasm` with `wasm-bindgen`, and builds the package into `dist`.
