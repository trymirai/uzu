# kiban

Cross-platform OS primitives (基盤, "foundation"). One async API that compiles to
[`tokio`](https://tokio.rs) on native targets and to browser primitives — [OPFS][opfs] for
storage, `wasm-bindgen-futures` for tasks — on `wasm`. Downstream crates depend on `kiban`
instead of `tokio` or `web-sys` directly, so the same code builds and runs on both.

[opfs]: https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system

## Modules

- **`fs`** — async filesystem operations backed by `tokio::fs` on native and OPFS on `wasm`.
  `PartFile` provides resumable, append-oriented writes for partial downloads.
- **`rt`** — task runtime.
- **`time`** — time primitives.
- **`process`** — process helpers.
- **`maybe`** — the `MaybeSend` / `MaybeSync` marker traits, which require `Send` / `Sync`
  on native targets and impose no bound on `wasm`, where futures are single-threaded.

The crate also exports the `printf!` and `eprintf!` macros, which route to `println!` /
`eprintln!` on native and to the browser console on `wasm`.

## Platform

`wasm` support targets the browser and relies on OPFS for storage; note that OPFS does not
support hard links, so `fs::hard_link` falls back to copying. On native targets the full
`tokio`-backed implementations are used.
