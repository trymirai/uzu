# Mirai

Desktop chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine.
Tauri v2, React, Rust.

## Requirements

- macOS 26 or newer, Apple silicon
- Xcode 26 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- rustup; the first `cargo` run installs the nightly toolchain pinned by
  `rust-toolchain.toml` at the repository root
- Node 20.19 or newer and pnpm (`corepack enable` picks the version from `package.json`)

## Run

```bash
pnpm install
pnpm dev      # desktop app; the first build also compiles the engine and its Metal shaders, later builds are incremental
pnpm build    # .app and .dmg in src-tauri/target/aarch64-apple-darwin/release/bundle/, ad-hoc signed
pnpm web:dev  # frontend in a browser without the engine: no models, no chats
```

`pnpm check` runs typecheck, ESLint, client tests, clippy and Rust tests;
`pnpm format` runs Prettier.

## Layout

```text
client/src/            React app, shared by desktop and web
  components/          app shell, icons and generic UI primitives (components/ui)
  features/            chat, chat-history, local-models, settings, welcome, runtime; each owns its page
  routes/              thin TanStack route glue
  platform/            platform contract + tauri/web adapters, the only way to the backend
  stores/              zustand stores
  styles/              design tokens and base styles
src-tauri/src/         Rust backend: engine, chat, downloads, storage
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
