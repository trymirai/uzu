# Mirai

Desktop chat app for local LLMs on the [uzu](https://github.com/trymirai/uzu) engine.
Tauri v2, React, Rust.

## Requirements

- rustup; the first `cargo` run installs the nightly toolchain pinned by
  `rust-toolchain.toml` at the repository root
- Node.js 22.19 or newer and pnpm, both on `PATH`: `tauri dev` and `tauri build` start the client with
  `pnpm run`, and pnpm runs the dependencies' install scripts with `node`. On Node.js 24 and older,
  `corepack enable` provides the pnpm version `package.json` names; Node.js 25 and newer no longer ship
  corepack, so install pnpm with its [installer](https://pnpm.io/installation). bun can stand in for
  Node.js: with a `node` link to `bun` on `PATH`, and pnpm run as `bun x pnpm@12.9.0`, installing,
  building and the type, lint and format checks work; the client tests need Node.js, as vitest does
  not run fully under bun
- macOS 26.4 or newer on Apple silicon, with Xcode 26 and its Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`)
- Windows or Linux with a Vulkan 1.4 GPU, plus:
  - the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home), whose Slang compiles the shaders, and libclang;
    the [CLI section](../../README.md#cli) lists the variables that locate them
  - Tauri's [prerequisites](https://v2.tauri.app/start/prerequisites/) for the OS
  - on Linux, the ALSA headers (`libasound2-dev` on Debian and Ubuntu). Bundling the AppImage downloads
    linuxdeploy and its plugins from GitHub, so it needs the network. Tauri runs them extracted, with
    or without `libfuse2`, and they leave `/tmp/appimage_extracted_*` directories behind

On Linux the app needs `libvulkan1` and a Vulkan driver at runtime. Mesa's lavapipe, a software
driver that runs models on the CPU, works too, slowly. Links open in the browser through `xdg-open`
from `xdg-utils`. The .deb depends on both packages.

## Run

```bash
pnpm install
pnpm dev      # desktop app; the first build also compiles the engine and its Metal or Slang shaders, later builds are incremental
pnpm build    # bundles in ../../target/release/bundle/: ad-hoc signed .app and .dmg on macOS, NSIS installer on Windows, .deb and AppImage on Linux
pnpm web:dev  # frontend in a browser without the engine: no models, no chats
```

A Windows release build that runs without the Visual C++ Redistributable links the C and C++ runtimes
statically. Set the flag for the target alone and name the target, so build scripts keep the dynamic runtime:
Windows Defender quarantines statically linked build scripts as a false positive.

```powershell
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = "-C target-feature=+crt-static"
pnpm build --target x86_64-pc-windows-msvc  # NSIS installer in ../../target/x86_64-pc-windows-msvc/release/bundle/nsis/
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS
```

`pnpm check` runs typecheck, ESLint, client tests, clippy and Rust tests;
`pnpm format` runs Prettier.

## Preview a fresh start (macOS)

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
```

Inter and Geist Mono are bundled under the SIL Open Font License; license texts
are in `client/public/fonts/`.
