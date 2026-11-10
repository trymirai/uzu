# Mirai

Chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine,
built from one React client as two equal apps: `desktop/` (Tauri v2 and Rust)
and `web/` (uzu.run, Uzu WASM on WebGPU). Nothing here builds by default: every
command that runs, builds or tests an app names it with a `desktop:` or `web:`
prefix.

## Shared commands

```bash
pnpm install
pnpm typecheck     # TypeScript types of the shared client
pnpm lint          # ESLint
pnpm test          # shared client tests
pnpm format        # Prettier
```

`pnpm desktop:check` and `pnpm web:check` run these checks plus the app's own.

## Run desktop

Requirements:

- macOS 26.4 or newer, Apple silicon
- Xcode 26 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- rustup; the first `cargo` run installs the nightly toolchain pinned by
  `rust-toolchain.toml` at the repository root
- Node.js 26 or newer and pnpm (`corepack enable` picks the version from `package.json`)

```bash
pnpm desktop:dev    # desktop app; the first build also compiles the engine and its Metal shaders, later builds are incremental
pnpm desktop:build  # .app and .dmg in ../../target/aarch64-apple-darwin/release/bundle/, ad-hoc signed
```

`pnpm desktop:lint` runs clippy and `pnpm desktop:test` runs the Rust tests.
Tauri serves and builds the frontend through `pnpm desktop:frontend:dev` and
`pnpm desktop:frontend:build` (`desktop/vite.config.ts`, output in `desktop/dist/`).

## Run web app (uzu.run)

The web app uses the repository's Uzu WASM bindings.
To build them, install the repository Rust toolchain, the `wasm32-unknown-unknown` target, Slang (configured with
`SLANG_DIR` or `SLANG_LIB_DIR`), and `wasm-bindgen-cli` matching the `wasm-bindgen` version in the root `Cargo.lock`.
From the repository root:

```bash
cd apps/mirai-gui
pnpm install
pnpm web:wasm
pnpm web:dev
```

Open [http://localhost:3001](http://localhost:3001) in a browser with WebGPU
(recent Chrome, Edge or Safari). The first visit opens the welcome screen;
**Get started** opens the live model catalog, and later visits open it directly.
Search families and open a family to see its models, sizes and quantization.
Download models into OPFS through Uzu's Rust `download-manager`, with progress,
pause and resume; partial downloads survive a page reload. Cancel a partial
download or delete a downloaded model to free space.

Chat with a downloaded model as on desktop: replies stream with reasoning,
reasoning effort, sampling parameters, the date-and-time, chart and chat-naming
tools, generated titles, stop, eject and auto-eject. Inference runs in the same
Uzu engine on WebGPU; `web/src/chat` is the browser counterpart of
`desktop/src/chat`. Conversations and global instructions are saved in this
browser's IndexedDB as the same Markdown files the desktop app writes, and can be
exported or cleared from Settings. Settings are kept in `localStorage`.

The catalog, chat, history, settings, application layout and route tree are
reused from `client/`. `web/` connects the chat, session and models services to
WASM and overrides only the welcome entry point in the shared router. Vite
reloads the app when files change.

`pnpm web:wasm` runs the existing `cargo tools build wasm` command and uses the
current branch's WebGPU backend and kernels. Rebuild after changing Rust code.
Once the bindings are built, running Vite only requires Node.js and pnpm. The
bindings run on the page because the current OPFS layer uses `window.navigator`.

Models previously saved by the experimental Cache Storage loader are not migrated
or deleted automatically. Download them again to store them in OPFS.

To build and preview the production version, stop the development server first
because both servers use port 3001:

```bash
pnpm web:build    # static files in web/dist/
pnpm web:preview  # serve the production build on port 3001
```

`pnpm web:typecheck` checks the web app's TypeScript types and
`pnpm web:test` runs the browser catalog and chat tests; both need the bindings
built by `pnpm web:wasm`.

The production build reports page views to Google Analytics only when it is served
from `https://uzu.run`; development and preview builds send nothing.

## Preview a fresh start

Quit Mirai, the CLI, and other apps using the shared model cache. From the
repository root, move the current state into a new backup directory:

```bash
python3 scripts/mirai-data.py backup ~/mirai-preview-backup
```

This moves chats, settings, logs, the model cache, and Mirai's WebView data
aside. Add `--keep-cache` to leave `~/.cache/mirai` available, for testing a
fresh app with existing models. Moves require the backup and data to be on
the same filesystem; model files are not copied.

After testing, quit the app and restore:

```bash
python3 scripts/mirai-data.py restore-overwrite ~/mirai-preview-backup
```

Restore replaces the test state and consumes the backup. If the backup used
`--keep-cache`, restore leaves the current cache untouched, including models
downloaded during testing. No live data is changed if the backup is incomplete.

## Signed builds without a release

In GitHub Actions, open **Build / release mirai-gui**, choose **Run workflow**,
and select the branch to build. GitHub requires repository write access to run
it manually; the workflow must be on the default branch for the button to appear.

Download the `mirai-macos-…` artifact from the completed run. It contains the
Apple-signed app in a notarized DMG, plus the app archive and updater signature, and
is retained for 14 days. The app and bundled CLI are built from the selected
commit using its existing version. Manual runs do not create a release or tag,
change versions, or publish an updater manifest.

These builds use the normal Mirai app identity, local data, and release update
feed; installing one replaces the regular app. Published engine releases still
use this workflow to build and publish the app automatically.

## Layout

```text
client/src/            React app, shared by desktop and web
  components/          app shell, icons and generic UI primitives (components/ui)
  features/            chat, chat-history, local-models, settings, runtime; each owns its page
  routes/              thin TanStack route glue
  platform/            platform contract + tauri/web adapters, the only way to the backend
  stores/              zustand stores
  styles/              theme.css: the palette, light values in @theme, dark ones in .dark
desktop/src/           Rust backend: engine, chat, downloads, storage
web/src/               uzu.run entry point and browser screens
  chat/                chat sessions over Uzu WASM, the counterpart of desktop/src/chat
  models/              model catalog and downloads over Uzu WASM
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
