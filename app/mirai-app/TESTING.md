# Verifying the GPUI rewrite against mirai-chat

`app/mirai-app` reimplements the Electron client (`external/mirai-chat`). "Done
correctly" means two things: it *behaves* like mirai-chat (logic parity) and it
*looks* like it (visual parity). The build machine is locked, so verification is
layered by how much it needs a display — the first two layers run headless on
CI; the third needs a screen.

## Layer 1 — Logic golden / unit tests (headless, runs today)

Pure logic gets `#[test]` + [`insta`](https://insta.rs) snapshots. No window, no
engine, no model.

- `persistence.rs` — the `StoredChat` JSON format is pinned with
  `assert_json_snapshot!`. A diff there means existing on-disk chats may stop
  round-tripping; review before accepting.
- `ui-kit` `markdown.rs` — the parser is dumped to a theme-independent string and
  snapshotted (inline bold/italic/code/links, fenced code blocks). This gates
  chat visual parity.
- `screens/chat.rs` — `conversation_for_request` (what history is sent: trailing
  placeholder dropped, errored turns excluded) and `sampling_method`
  (Default/Argmax/Stochastic, "0 = off") — the message-versions and sampling
  features that can't be exercised interactively here.

Run: `cargo test -p mirai-app -p ui-kit` (from `app/`). Snapshots live in
`snapshots/` next to the source; review changes with `cargo insta review`
(install `cargo-insta`).

To grow parity coverage, add a shared fixture (a markdown sample, a chat file)
and assert the GPUI output matches mirai-chat's for the same input.

## Layer 2 — Interaction tests (headless, `gpui::test`)

GPUI ships `#[gpui::test]` + `TestAppContext`, which simulate platform input
(clicks, keystrokes) and advance the executor without a real window server, so
they run on CI. Use these for behavior that isn't reachable as a pure function:
the version pager buttons, segmented-control selection, composer submit. Pattern
reference: the tests under `external/zed/crates/gpui`. Views that need the uzu
engine can't run here (no model on CI) — keep that logic in Layer 1 by extracting
pure helpers, as `chat.rs` now does.

## Layer 3 — Visual regression (needs a display)

Pixel-exact comparison between the GPUI (Metal) render and mirai-chat (Chromium)
is not meaningful — different rasterizers, fonts, antialiasing. So:

- Capture mirai-chat with Playwright `toHaveScreenshot()` (pixelmatch under the
  hood) as a visual *reference*, per screen/state.
- Capture the GPUI window with `screencapture -l <window-id>` on a runner with an
  unlocked/virtual display, and diff against the app's *own* committed baselines
  (odiff / pixelmatch, or the Rust engines Honeydiff / LumenDiff) to catch
  regressions in the rewrite.
- Compare GPUI vs mirai-chat *perceptually* (layout, spacing, color), not by a
  pixel threshold — the manual sign-off for the Phase 6 visual pass.

## What runs where

| Layer            | Needs display | Needs model | On CI                          |
|------------------|---------------|-------------|--------------------------------|
| 1 logic / golden | no            | no          | yes                            |
| 2 `gpui::test`   | no            | no          | yes                            |
| 3 visual         | yes           | no          | macOS runner with a GUI session |
