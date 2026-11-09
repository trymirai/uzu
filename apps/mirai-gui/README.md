# Mirai

Desktop chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine.
Tauri v2, React, Rust.

## Requirements

- macOS 26.4 or newer, Apple silicon
- Xcode 26 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- rustup; the first `cargo` run installs the nightly toolchain pinned by
  `rust-toolchain.toml` at the repository root
- Node.js 26 or newer and pnpm (`corepack enable` picks the version from `package.json`)

## Run desktop

```bash
pnpm install
pnpm dev      # desktop app; the first build also compiles the engine and its Metal shaders, later builds are incremental
pnpm build    # .app and .dmg in ../../target/aarch64-apple-darwin/release/bundle/, ad-hoc signed
```

`pnpm check` runs typecheck, ESLint, client tests, clippy and Rust tests;
`pnpm format` runs Prettier.

## Run web app (uzu.run)

The web app only needs Node.js and pnpm; Rust, Xcode and the desktop build
are not required. From the repository root:

```bash
cd apps/mirai-gui
pnpm install
pnpm web:dev
```

Open [http://127.0.0.1:3001](http://127.0.0.1:3001) in your browser. The app
currently opens the welcome screen; model selection and inference are not yet
connected, so **Get started** is disabled. Vite reloads the app when files change.

To build and preview the production version, stop the development server first
because both servers use port 3001:

```bash
pnpm web:build    # static files in web/dist/
pnpm web:preview  # serve the production build on port 3001
```

`pnpm web:typecheck` checks the web app's TypeScript types.

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
web/                   uzu.run entry point, Vite config and browser screens
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
