# Browser WebGPU example

From the repository root, enter the development shell with direnv or `nix develop`
(`nix develop .#host-xcode` when using the host Xcode toolchain on macOS), then run:

```sh
cd crates/legacy/uzu/bindings/typescript
pnpm install --frozen-lockfile
pnpm build:wasm
pnpm test:browser
```

The shell supplies wasm-bindgen, Playwright, and Chromium from nixpkgs. There is
no browser download through npm and no dependency on an installed Chrome.

The example imports the browser package, downloads the pinned
`trymirai/Qwen3.5-0.8B-M` checkpoint into OPFS, and calls
`Engine.create` → `engine.model` → `engine.chat` → `session.reply` with seed 0
and the prompt `What is 2 + 2 and why?`. It returns one complete reply, including
reasoning. The runner asserts that the answer contains 4 and closes Chromium.
The model can spend several minutes reasoning before answering.

For repeated tests, serve an existing checkpoint through the runner's local
HTTP server instead of downloading it again:

```sh
UZU_MODEL_PATH=/path/to/checkpoint pnpm test:browser
```

That directory must contain `config.json`, `tokenizer.json`, and
`model.safetensors` from revision `c12202e4c764e559960827761566aaa1fd15a87a`.
The browser still streams the files into real OPFS and uses the same package.
Each run uses a temporary browser profile, removed on completion.

`dist-wasm` contains the ESM browser package and generated TypeScript declarations.
Initialize it with the default async export before using the classes. The browser
engine currently resolves models already stored in OPFS; the example handles
download and supplies the Qwen encoding metadata normally supplied by the native
registry. Harmony, remote backends, and native registry/storage APIs are not
included in this browser entry point.
