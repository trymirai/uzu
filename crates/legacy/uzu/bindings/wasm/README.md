# uzu-wasm

WebAssembly build of [uzu](https://github.com/trymirai/uzu) for the browser.

## Usage

Install it next to [`@trymirai/uzu`](https://www.npmjs.com/package/@trymirai/uzu), which re-exports it in browser builds (through the `browser` export condition), so the same import works in Node and in the browser:

```bash
pnpm add @trymirai/uzu @trymirai/uzu-wasm
```

```js
import { Engine, EngineConfig } from '@trymirai/uzu';

const engine = await Engine.create(EngineConfig.create());
```

Importing `@trymirai/uzu-wasm` directly works too.

The API is the same as [`@trymirai/uzu`](https://www.npmjs.com/package/@trymirai/uzu), except for what the browser build doesn't include: text-to-speech, classification, `Player`, and the `Native` download manager.

Importing the package loads `uzu_bg.wasm` from next to the bindings, so there is no `init()` to call. To load it from another location, import the bindings directly and pass its URL: `import init from '@trymirai/uzu-wasm/wasm/uzu.js'; await init({ module_or_path: url })`. That entry has no `for await` over streams and no zod tool helpers.

## Building

From the repository root:

```bash
cargo tools build wasm
```

This compiles `uzu` for `wasm32-unknown-unknown`, generates the bindings into `src/wasm` with `wasm-bindgen`, and builds the package into `dist`.

## Checking the API

This package is meant to have the same TypeScript API as [`@trymirai/uzu`](https://www.npmjs.com/package/@trymirai/uzu), so code written against one compiles against the other. To list where they differ:

```bash
pnpm check-api            # one line per difference
pnpm check-api --verbose  # full TypeScript error chains
```

It compares the two packages' declarations, so it needs `pnpm install` in both packages and the wasm bindings built with `cargo tools build wasm`.
