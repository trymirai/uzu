# How to run uzu

uzu is the inference engine behind Mirai. Chat in the Mirai Mac app, from the `mirai` CLI, or call
uzu directly from Rust, Python, Swift, and TypeScript.

This guide uses one fixed model repo ID everywhere:

```text
trymirai/Qwen3.5-4B-M
```

`trymirai/Qwen3.5-4B-M` is our quantized **medium** checkpoint — the 4-bit (`-M`) build of
Qwen3.5 4B. Swap it later for another supported repo ID from the
[model library](https://trymirai.com/models) (for example the lighter 0.8B and 2B builds, or the
8-bit `-L` variants). The first run downloads the model; after that uzu reuses the local copy.

# Installation

Installing Mirai gives you both the Mac app and a `mirai` command in your terminal. No code required.

## From website

Download Mirai from [trymirai.com](https://trymirai.com/chat-for-mac), open the DMG, and move Mirai
to Applications. This also adds the `mirai` CLI to your PATH.

## From brew

```bash
brew install mirai
```

This installs the same Mac app and `mirai` CLI.

## Building from source

For development, or to build the CLI and SDK yourself:

```bash
git clone https://github.com/trymirai/uzu.git
cd uzu
cargo tools setup --include-platform-specific
```

`cargo tools setup --include-platform-specific` installs the build tools uzu needs (cmake,
clang-format, the Metal toolchain, rustup, uv, and pnpm). You can then run the CLI with
`cargo run --release -p cli`, or add the crate to your own project (see [Language APIs](#language-apis)).

If a build fails on macOS, run `xcodebuild -runFirstLaunch` and re-run
`cargo tools setup --include-platform-specific`.

# CLI

The `mirai` CLI is an interactive terminal app where you can browse, download, and chat with models.

After installing the app (website or brew):

```bash
mirai
```

From a source checkout:

```bash
cargo run --release -p cli
```

Either way, pick `trymirai/Qwen3.5-4B-M`, let it download, and start chatting.

# Language APIs

Embed inference in your own app or service. Each example downloads `trymirai/Qwen3.5-4B-M` and
prints a short reply. It's a reasoning model, so each snippet prints the final answer (`text`),
falling back to the model's thinking (`reasoning`) when it hasn't emitted a separate answer.

### Rust

```bash
cargo new uzu-demo
cd uzu-demo
cargo add uzu --git https://github.com/trymirai/uzu
cargo add tokio --features full
```

Paste this into `src/main.rs`:

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

const MODEL_REPO_ID: &str = "trymirai/Qwen3.5-4B-M";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let model = engine.model(MODEL_REPO_ID.to_string()).await?.ok_or("Model not found")?;

    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let session = engine.chat(model, ChatConfig::default()).await?;
    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short story about a tiny robot".to_string()),
    ];
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.last() {
        println!("{}", reply.message.text().or(reply.message.reasoning()).unwrap_or_default());
    }

    Ok(())
}
```

Run it with `cargo run --release`.

### Python

```bash
uv init uzu-demo
cd uzu-demo
uv add uzu
```

Paste this into `main.py`:

```python
import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig

MODEL_REPO_ID = "trymirai/Qwen3.5-4B-M"


async def main() -> None:
    engine = await Engine.create(EngineConfig.create())
    model = await engine.model(MODEL_REPO_ID)
    if model is None:
        raise RuntimeError(f"Model not found: {MODEL_REPO_ID}")

    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    session = await engine.chat(model, ChatConfig.create())
    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short story about a tiny robot"),
    ]
    replies = await session.reply(messages, ChatReplyConfig.create())
    if replies:
        message = replies[-1].message
        print(message.text or message.reasoning)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it with `uv run main.py`.

### Swift

In Xcode, add the Swift Package `https://github.com/trymirai/uzu.git`, then call uzu:

```swift
import Uzu

let modelRepoID = "trymirai/Qwen3.5-4B-M"

func runUzu() async throws {
    let engine = try await Engine.create(config: .create())
    guard let model = try await engine.model(identifier: modelRepoID) else {
        throw NSError(domain: "UzuDemo", code: 1)
    }

    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }

    let session = try await engine.chat(model: model, config: .create())
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short story about a tiny robot")
    ]
    let replies = try await session.reply(input: messages, config: .create())
    let message = replies.last?.message
    print(message?.text() ?? message?.reasoning() ?? "")
}
```

### TypeScript

```bash
mkdir uzu-demo
cd uzu-demo
bun init -y
bun add @trymirai/uzu
```

Paste this into `main.ts`:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu';

const MODEL_REPO_ID = 'trymirai/Qwen3.5-4B-M';

async function main() {
    const engine = await Engine.create(EngineConfig.create());
    const model = await engine.model(MODEL_REPO_ID);
    if (!model) {
        throw new Error(`Model not found: ${MODEL_REPO_ID}`);
    }

    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const session = await engine.chat(model, ChatConfig.create());
    const messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short story about a tiny robot'),
    ];
    const replies = await session.reply(messages, ChatReplyConfig.create());
    const message = replies.at(-1)?.message;
    console.log(message?.text ?? message?.reasoning ?? '');
}

main().catch(console.error);
```

Run it with `bun run main.ts`.

## Things to remember

- Use exact repo IDs, such as `trymirai/Qwen3.5-4B-M`.
- Keep only one session loaded at a time; each model can use a lot of memory.
- If a model is not found, check the repo ID against the [model library](https://trymirai.com/models).
- If a source build fails on macOS, run `xcodebuild -runFirstLaunch` and
  `cargo tools setup --include-platform-specific`.
