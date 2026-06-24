# mirai-app (GPUI) — what's left to implement

Status: the core AI feature set is built and runs (chat, cloud, routers, TTS,
model management, history, settings, welcome). This lists what remains versus
the Electron mirai-chat app, roughly prioritized: **P1** high-value, **P2**
medium, **P3** nice-to-have.

## ⚠️ Verification gaps (do first — not code, but unknown-good)
- Live **chat streaming**, **classify**, and **TTS audio** have never been driven by a human (no UI input automation; screen kept locking). Confirm they actually work on a real message/text.
- **Cloud Models**, **Routers**, **TTS** screens not visually verified yet (Local Models / Chat / Settings / Welcome were).
- **Light mode** never visually checked.

## Chat
- ✅ **DONE** (partial) — inline **bold/italic/inline-code** (via `StyledText` highlights) + **headings** + **bullets**. Still TODO: links, tables, math, blockquotes, numbered lists, inline-code monospace (HighlightStyle has no font-family).
- ✅ **DONE** (partial) — per-message **Copy** button. Still TODO: code-block syntax highlighting + per-code-block copy button.
- ✅ **DONE** (partial) — **Regenerate** (↻ on the last assistant message; re-runs the last turn, replacing the reply). Still TODO: message **versions** (keep + switch between regenerations).
- ✅ **DONE** — model selector in the composer (clickable "Model: …" trigger → overlay list of installed local models). Could later become an anchored popover instead of a centered overlay.
- **P2** **Multiline composer** (today single-line). Shift+Enter newline, autosize.
- **P2** **Title generation** via LLM (today: first user message truncated).
- **P2** Session reuse / context caching — today each send creates a fresh `ChatSession` and re-sends the full history (re-prefills every turn).
- **P3** File attachments (images/files) + context-window limit checks.
- **P3** Streaming text shimmer/animation; empty-state suggestions.

## Chats / History
- **P1** Sidebar **recent-chats list** (mirai-chat shows saved chats in the sidebar; today only the Chats screen).
- **P2** **Search** chats, **rename**, **multi-select + bulk delete** (today: single delete only).
- **P2** Global Instructions editor on the Chats page (today only in Settings).
- **P3** Export all chats to zip.
- **P3** mirai-chat **Markdown chat-file format** interop (today JSON-per-chat under `~/Library/Application Support/Mirai/chats/`).

## Local Models
- ✅ **DONE** — redesigned to match mirai-chat: two-level **family list → family detail** (Installed/Available sections, name/size/quantization, Mirai-quantization accents, "Mirai quantizations" badge, model/installed counts, param range). **Tapping an installed model starts a chat** (`LocalModelsEvent::UseModel`). Search filters families (top) / models (detail).
- **P2** Bundle **per-vendor icons** (today a generic glyph for all families).
- **P2** **Sort** dropdown (newest / size / name).
- ✅ **DONE** — delete-confirm modal (Local Models & Chats now confirm before deleting).
- **P2** Device header ("for your Mac…") + **recommended model** row (needs SDK endpoint + device memory; no uzu API).
- **P3** Quantization / "thinking" badges, model detail.

## Cloud Models
- **P1** Runtime **API-key entry** (connectors) + engine re-config (today: keys only via env at startup; screen shows a hint when none).
- **P3** Vendor icons.

## Routers
- **P2** Router **detail page** / per-router config; recommended-threshold display.
- **P3** Local vs API router distinction.

## Text-to-Speech
- ⛔ **BLOCKED by uzu API** — generation settings (language/speaker/voice/seed/speed/temperature/etc.) are NOT exposable: uzu's `session.synthesize(input: String)` takes only text, and `engine.text_to_speech(model)` takes only the model — no config struct exists in `shoji`. Needs uzu to add a TTS config (like ChatReplyConfig) first.
- **P1** **Generated-audio history** (list, replay, save/export wav).
- **P2** **Reference-voice recording** (mic capture).
- **P2** Multiline/large text input + **char counter/limit**; file upload for text.
- **P2** Playback controls (scrubber, pause/resume, progress) — today just play/stop.

## Settings
- ✅ **DONE** — reasoning toggle now hides/shows the chat reasoning panel (via a shared `settings_state` global). NOTE: uzu's `ChatConfig`/`ChatReplyConfig` have no enable-thinking flag, so this is correctly a UI control, not an inference one.
- **P2** **Auto-eject** idle timeout (unload resident model after N min) — needs runtime session tracking; drives the footer Eject button.
- **P2** Tabbed layout (General / Profile / Privacy / Connectors / About) — today single scroll page.
- **P3** Privacy tab (analytics opt-in, clear data), Profile tab, Connectors tab.
- **P3** run-on-startup, show-in-menu-bar, quick-entry shortcut toggles.

## App shell / footer
- ✅ **DONE** (UI-level) — footer shows the loaded model + an **Eject** button (stops generation, clears the indicator). NOTE: uzu has no unload API, so eject does NOT free GPU memory — it's a UI deselect. A true unload needs an `Engine::unload`/eviction API in uzu.
- **P2** **Window dragging** by the top bar (traffic lights are inset, but the titlebar drag region isn't wired).
- **P3** Sidebar collapse/toggle; "Apps" item (disabled placeholder in mirai-chat).

## Missing reusable components
- ✅ **DONE** — `ConfirmModal` (dim backdrop + card + Cancel/Confirm); used for delete-confirm in Local Models & Chats. Still need: rename modal, clear-data dialog (reuse it).
- **P1** **Dropdown / Select / Popover** (model selector, sort, settings).
- ✅ **DONE** — **Toast** notifications (`toast.rs` global; top-right, auto-dismiss). Wired to fire on model-download completion. (Could also fire on download/inference errors.)
- **P2** **Loader / spinner** (mirai-chat uses a Rive animation).
- **P2** **CopyButton**, **Checkbox** (welcome analytics opt-in), **Tooltip**, multiline **Textarea**.

## Engine / inference
- ✅ **DONE** (partial) — **temperature** + **max-tokens** steppers (gear in the composer → overlay), wired to `ChatReplyConfig` (`SamplingMethod::Stochastic` + `with_token_limit`). Still TODO: top-p / top-k / repetition-penalty, and **seed** (via `ChatConfig::with_sampling_seed`).
- **P2** Enable-thinking / reasoning-effort control wired through `ChatConfig`/`ChatReplyConfig`.
- **P2** Single-resident-session eviction coordination across chat/router/tts (mirai-chat's runtime ownership).
- **P3** Structured output / grammar (build currently drops `capability-grammar`; re-enable + a JSON-schema UI).

## System integration (Phase E — mostly deferred by design)
- **P2** Global **quick-entry keyboard shortcut**; **run-on-startup**.
- **P3** Menu-bar/tray + dock recent chats (basic app menu + ⌘Q done).
- **P3** (Likely N/A for this rebuild) Auto-update (GCS), Amplitude analytics, Sentry crash reporting, Google auth.

## Design / polish
- **P2** Animations (welcome logo reveal, fade-in/slide-up), custom thin scrollbars.
- **P2** 0.5px borders (today 1px), exact spacing/radius pass vs mirai-chat.
- **P3** Bundle a real Mirai logo (today flattened single-color mark) + vendor icons.
