# Mirai

Desktop chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine.
Tauri v2, React, Rust.

## Requirements

- macOS 26 or newer, Apple silicon
- Xcode 26 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- Rust nightly, pinned by `rust-toolchain.toml` at the repository root
- Node 20, pnpm 10

## Run

```bash
pnpm install
pnpm dev      # desktop app; the first build also compiles the engine from crates/
pnpm build    # .app and .dmg in src-tauri/target/aarch64-apple-darwin/release/bundle/, ad-hoc signed
pnpm web:dev  # frontend in a browser without the engine: no models, no chats
```

`pnpm check` runs typecheck, ESLint, client tests, clippy and Rust tests;
`pnpm format` runs Prettier.

## Configuration

Optional. `API_KEY` in `.env` next to this README authenticates the model
registry; without it the registry is used anonymously. See `.env.example`.

## Layout

```text
client/src/            React app, shared by desktop and web
  platform/            platform contract + tauri/web adapters, the only way to the backend
  ui-kit/              UI components
src-tauri/src/         Rust backend: engine, chat, downloads, storage
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
