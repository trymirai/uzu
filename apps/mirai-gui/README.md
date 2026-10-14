# Mirai

Desktop and browser chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine.
Tauri v2, React, Rust, WebGPU.

## Requirements

- macOS 26.4 or newer, Apple silicon
- Xcode 26 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- rustup; the first `cargo` run installs the nightly toolchain pinned by
  `rust-toolchain.toml` at the repository root
- Node 20.19 or newer and pnpm (`corepack enable` picks the version from `package.json`)

## Run

```bash
pnpm install
pnpm dev      # desktop app; the first build also compiles the engine and its Metal shaders, later builds are incremental
pnpm build    # .app and .dmg in desktop/target/aarch64-apple-darwin/release/bundle/, ad-hoc signed
pnpm web:dev  # browser app; complete the browser setup below first
```

`pnpm check` runs typecheck, ESLint, client tests, clippy and Rust tests;
`pnpm format` runs Prettier.

## Browser setup

The browser build runs the Rust inference engine in a dedicated WebAssembly worker
and executes kernels on WebGPU. It fetches the same production Mirai model registry
as the desktop app and downloads pinned model revisions directly from their remote
repositories. No model files need to be staged or deployed with the app.

```bash
rustup target add wasm32-unknown-unknown
# Install wasm-bindgen-cli with the same version as wasm-bindgen in Cargo.lock.
cargo install wasm-bindgen-cli --locked --version 0.2.126
pnpm web:setup # installs the pinned, checksum-verified Slang compiler
pnpm web:wasm
pnpm web:dev
```

Open `http://localhost:3001` in a browser with WebGPU enabled. Choose a model in
Local Models, download it, and open a chat. The footer shows **WebGPU** while a
model is loaded. Browser inference has no CPU backend or remote inference service.

Kernels are written in Slang and compiled to WGSL during `web:wasm`; the browser
does not download or run the Slang compiler. `web:setup` installs Slang 2026.19 in
the user's cache. An existing installation can be selected with `SLANGC` or
`SLANG_DIR`. Direct Cargo builds need one of those variables or `slangc` on `PATH`.

The catalog preserves model names, vendor icons, families, parameter sizes,
quantizations, and chat capabilities. Models whose checkpoints exceed the current
browser engine's limits remain visible with a reason and cannot be downloaded.
There is no fixed whole-checkpoint size cap; GPU memory and individual tensor
limits determine what can run on a device. The engine validates the downloaded
configuration before fetching tokenizer or weight files and reports unsupported
architectures without downloading the large checkpoint.
The registry uses the browser's reported memory class when available, or a conservative
4 GiB catalog budget when the browser does not expose one. This estimate does not
promise a model fits in memory.

Downloads use persistent Cache Storage chunks with IndexedDB metadata. Pausing,
reloading, or losing the network retains received bytes; resuming requests the
remaining byte range. A server that ignores ranges restarts that file safely.
SHA-256, Git blob SHA-1, or CRC32C checksums from the catalog verify the entire file
before the model is marked downloaded. Deletion cancels an active download across
browser tabs and removes its complete and partial files. The app checks storage
quota, requests persistent storage where available, and reports storage, network,
access, and integrity errors. Clearing site data removes models and conversations.

When loading a model, the worker composes the cached chunks into a Blob and reads
bounded ranges for GPU upload. It does not concatenate the entire checkpoint into
a JavaScript or WASM byte array. Configuration and tokenizer files remain small
byte arrays. Individual buffers must fit the adapter's limits and the wasm32
address-space limit of 2,147,483,584 bytes; aggregate checkpoints can be larger.
The model and inference state must fit available GPU memory.
Downloads include the base checkpoint, configuration, and tokenizer; optional
speculator checkpoints and repository documentation are not downloaded.

Chrome validation loaded cached 2.44 GB Qwen3.5-4B and 5.19 GB Qwen3.5-9B
checkpoints as composite Blobs, including tensor offsets above 4 GiB. Both used
186 MB of WASM memory after loading and generated the same tested token sequences
as Metal. This measures WASM memory, not total browser or GPU memory. The separate
live remote-download test uses the 125 MB LFM checkpoint; it does not claim a
complete remote download test of the 5.19 GB model.

Normal chats use the engine's recommended context allocation (up to 16,384 tokens
for bounded models on WebGPU), with responses continuing until a stop token,
context limit, or cancellation. They have no separate 1,024-token output cap.
Cancelling a model load interrupts its bounded reads/uploads; ejecting releases
the worker and GPU resources. Auto-eject changes apply immediately to idle models.

The production catalog endpoint is `https://sdk.trymirai.com/api/v1/fetch/models`.
Its backend/toolchain version is read from the engine at build time. Pinned
Hugging Face revisions supply file metadata and checksums; registry asset URLs
remain usable if repository metadata is temporarily unavailable. A cached catalog
and downloaded files remain usable when the catalog is offline. The app itself
must still be served to open it. Conversations and settings stay in browser storage;
model discovery and downloads send no chat content.

Downloaded and partially downloaded revisions stay pinned when the remote catalog
changes or removes an entry. Delete that model and refresh the catalog to download
a newer revision. This also preserves an interrupted download through catalog updates.

For deployment, build WASM and run `pnpm web:build`. Serve `client/dist-web/`
over HTTPS, including `.wasm` files with `Content-Type: application/wasm`, and set:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Development and `pnpm web:preview` set these headers automatically. A Content
Security Policy must permit module workers, `blob:` GPU workers, WASM compilation
(`wasm-unsafe-eval`), and connections to the Mirai registry/assets, Hugging Face,
and its download CDN redirects. The remote services must allow CORS; a local proxy
is not part of the browser download path.

`pnpm web:fixture /path/to/model --model lfm2.5-230m` (or `qwen3.5-0.8b`) is an
optional development helper for numerical and token-parity tests. It does not add
models to the app; its output stays in `.web-test-models/`, outside the deployed
public directory. Generated WASM and local test model files are ignored by Git.
`node scripts/test-web-downloads.mjs` runs an opt-in Chrome integration test against
the live catalog and remote LFM checkpoint, including pause, reload, range resume,
offline cache access, and deletion. It downloads approximately 125 MB into a
temporary browser profile and removes that profile after the test.
Set `UZU_WASM_DIR=client/public/wasm` to include real WASM capability and config
inspection in that test. Adding `UZU_INSPECT_ONLY=1` checks fresh-start inspection,
configuration rejection, and download error propagation using only the small config file.
See [`crates/uzu-web/README.md`](../../crates/uzu-web/README.md) for the browser
engine's validation and measured performance.

## Signed builds without a release

In GitHub Actions, open **Build / release mirai-gui**, choose **Run workflow**,
and select the branch to build. GitHub requires repository write access to run
it manually; the workflow must be on the default branch for the button to appear.

Download the `mirai-macos-…` artifact from the completed run. It contains the
Apple-signed app in a notarized DMG, plus the app archive and updater signature, and
is retained for 14 days. The app and bundled CLI are built from the selected
commit using its existing version. Manual runs do not create a release or tag,
change versions, publish an updater manifest, or upload to the website bucket.

These builds use the normal Mirai app identity, local data, and release update
feed; installing one replaces the regular app. Published engine releases still
use this workflow to build and publish the app automatically.

## Layout

```text
client/src/            React app, shared by desktop and web
  components/          app shell, icons and generic UI primitives (components/ui)
  features/            chat, chat-history, local-models, settings, welcome, runtime; each owns its page
  routes/              thin TanStack route glue
  platform/            platform contract + tauri/web adapters, the only way to the backend
  stores/              zustand stores
  styles/              theme.css: the palette, light values in @theme, dark ones in .dark
desktop/src/           Rust backend: engine, chat, downloads, storage
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
