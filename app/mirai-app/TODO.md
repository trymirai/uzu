# mirai-app (GPUI) — what's left to implement

Status: core AI features run (chat, cloud, routers, TTS, model management, history,
settings, welcome). Settings UI matches mirai-chat for General / Privacy / About.
Offscreen snapshot suite is green (14 screens + logic/unit tests).

## Verification

- **Automated (headless):** `cargo test -p mirai-app -p ui-kit` from `app/` — logic
  snapshots, chat request helpers, markdown golden output, 14 UI PNG baselines
  (5 engine snapshots via `cargo test -p mirai-app -- --ignored`).
- **Manual:** live chat streaming, classify, TTS audio playback, light mode, pixel
  parity vs Electron (different rasterizer — compare layout, not raw pixels).

## Settings — done vs remaining

| Item | Status |
|------|--------|
| Inner sidebar (General / Privacy / About Mirai) | ✅ |
| Instructions card, OS prefs, reasoning, auto-eject UI | ✅ |
| Privacy export chats (zip + save dialog) | ✅ |
| Privacy export logs | ✅ |
| Clear data modal (dialogs / audio / models / logs) | ✅ |
| Share usage data toggle (persists, no telemetry backend) | preference only |
| Run on startup / menu bar / global shortcut | preference only — needs macOS hooks |
| Auto-eject idle timer | preference only — needs engine unload API |

## Chat

- ✅ Markdown rendering (bold/italic/code/links, lists, headings, blockquotes, code blocks + copy)
- ✅ Message versions + regenerate pager
- ✅ Model picker, generation settings (temperature, max tokens, sampling modes)
- ✅ Multiline composer (Shift+Enter)
- **P2** ~~Title generation via LLM~~ (after first reply; vendor fallbacks)
- **P2** ~~Session reuse / context caching~~ (reuse loaded session; reset + full history each turn)
- **P3** File attachments, streaming shimmer, empty-state suggestions

## Chats / History

- ✅ Sidebar recent chats, search, bulk delete, global instructions card
- ✅ Markdown chat-file interop (`.md` mirror on save; load JSON or markdown)
- **P2** ~~Rename chat modal~~ (selection mode, single chat)
- **P3** Export all chats from Chats page (Settings export exists)

## Local Models

- ✅ Family list → detail, vendor icons, download/delete, Mirai quant styling
- **P2** Sort dropdown, device header + recommended model row
- **P3** Quantization / thinking badges on detail

## Cloud Models

- ✅ Runtime API-key entry (Connect providers UI, keychain persist, hot registry reload)
- **P3** Additional vendor polish

## Routers / TTS

- ✅ Screens + engine integration (basic)
- ⛔ TTS generation settings blocked by uzu API (no config struct on synthesize)
- ✅ TTS generated-audio history (persist + replay)
- **P2** Reference-voice recording, playback scrubber

## App shell

- ✅ Footer model indicator + Eject (UI deselect; no GPU unload in uzu yet)
- **P2** Window drag region on title bar
- **P3** Sidebar collapse, Apps placeholder

## Engine / inference

- ✅ Temperature + max-tokens + stochastic/argmax modes
- **P2** top-p / top-k / repetition-penalty / seed
- **P2** Single-resident-session eviction across chat/router/tts
- **P3** Structured output / grammar UI

## System integration (deferred)

- **P2** Global quick-entry shortcut capture, login item, menu-bar agent
- **P3** Auto-update, analytics, Sentry, Google auth (likely N/A for native rebuild)

## Design / polish

- **P2** Animations, thin scrollbars, 0.5px border pass
- **P3** Full Mirai wordmark asset bundle

See [TESTING.md](TESTING.md) for the three-layer verification strategy.
