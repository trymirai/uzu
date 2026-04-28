<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord&color=brightgreen" alt="Discord"></a> <a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-brightgreen" alt="Contact us"></a> <a href="https://docs.trymirai.com"><img src="https://img.shields.io/badge/Read-Docs-brightgreen" alt="Read docs"></a> [![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE) [![Build](https://github.com/trymirai/uzu/actions/workflows/tests.yml/badge.svg)](https://github.com/trymirai/uzu/actions) [![Python](https://img.shields.io/badge/Python-orange)](bindings/python) [![Package](https://img.shields.io/pypi/v/uzu?color=orange&label=Package&v=0.4.3)](https://pypi.org/project/uzu/) [![Python](https://img.shields.io/pypi/pyversions/uzu?color=orange&label=Python&v=0.4.3)](https://pypi.org/project/uzu/) [![TypeScript](https://img.shields.io/badge/TypeScript-yellow)](bindings/typescript) [![Package](https://img.shields.io/npm/v/@trymirai/uzu?color=yellow&label=Package&v=0.4.3)](https://www.npmjs.com/package/@trymirai/uzu) [![Downloads](https://img.shields.io/npm/dm/@trymirai/uzu?color=yellow&label=Downloads&v=0.4.3)](https://www.npmjs.com/package/@trymirai/uzu) [![Swift](https://img.shields.io/badge/Swift-blue)](bindings/swift) [![SPM](https://img.shields.io/badge/SPM-compatible-blue)](Package.swift) [![Platforms](https://img.shields.io/badge/Platforms-iOS%20%7C%20macOS-blue)](Package.swift) [![Swift](https://img.shields.io/badge/Swift-5.9-blue)](https://swift.org) 

# uzu

A high-performance inference engine for AI models. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. Key features:

- Simple, high-level API
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices
- [Broad model support](https://trymirai.com/models)

## Quick Start



<details>
<summary>Rust</summary>
<br>

Add the dependency:

```toml
[dependencies]
uzu = { git = "https://github.com/trymirai/uzu", branch = "main", package = "uzu" }
```

Run the code below:

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let session = engine.chat(model, ChatConfig::default()).await?;

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];

    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
```

</details>



<details>
<summary>Python</summary>
<br>

Add the dependency:

```bash
uv add uzu==0.4.3
```

Run the code below:

```python
import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        return

    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    session = await engine.chat(model, ChatConfig.create())

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]

    replies = await session.reply(messages, ChatReplyConfig.create())
    if not replies:
        return

    message = replies[-1].message
    print(f"Reasoning: {message.reasoning}")
    print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>



<details>
<summary>Swift</summary>
<br>

Add the dependency:

```swift
dependencies: [
    .package(url: "https://github.com/trymirai/uzu.git", from: "0.4.3")
]
```

Run the code below:

```swift
import Uzu

public func runQuickStart() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let session = try await engine.chat(model: model, config: .create())
    
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")
    ]
    
    let reply = try await session.reply(input: messages, config: .create())
    guard let message = reply.last?.message else {
        return
    }
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}
```

</details>



<details>
<summary>TypeScript</summary>
<br>

Add the dependency:

```bash
pnpm add @trymirai/uzu@0.4.3
```

Run the code below:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }

    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let session = await engine.chat(model, ChatConfig.create());

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];

    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;

    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
```

</details>


<br>

Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<**rust** | **python** | **swift** | **typescript**\> \<**chat** | **chat-cloud** | **chat-speculation-classification** | **chat-speculation-summarization** | **chat-structured-output** | **classification** | **quick-start** | **text-to-speech**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];
    let session = engine.chat(model, ChatConfig::default()).await?;
    let stream = session.reply_with_stream(messages, ChatReplyConfig::default()).await;
    let mut last_message: Option<ChatMessage> = None;
    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.first() {
                    last_message = Some(reply.message.clone());
                    println!("Generated tokens: {}", reply.stats.tokens_count_output.unwrap_or_default());
                }
            },
            ChatSessionStreamChunk::Error {
                error,
            } => {
                println!("Error: {error}");
            },
        }
    }
    if let Some(message) = last_message {
        println!("Reasoning: {}", message.reasoning().unwrap_or_default());
        println!("Text: {}", message.text().unwrap_or_default());
    }

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSessionStreamChunk,
    Engine,
    EngineConfig,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]
    session = await engine.chat(model, ChatConfig.create())
    stream = await session.reply_with_stream(messages, ChatReplyConfig.create())
    message: ChatMessage | None = None
    async for chunk in stream.iterator():
        if isinstance(chunk, ChatSessionStreamChunk.Replies):
            replies = chunk.replies
            if replies:
                reply = replies[0]
                message = reply.message
                print(f"Generated tokens: {reply.stats.tokens_count_output}")
        elif isinstance(chunk, ChatSessionStreamChunk.Error):
            print(f"Error: {chunk.error}")
    if message is not None:
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Uzu

public func runChat() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")
    ]
    let session = try await engine.chat(model: model, config: .create())
    let stream = await session.replyWithStream(input: messages, config: .create())
    var message: ChatMessage? = nil
    for try await update in stream.iterator() {
        switch update {
        case .replies(let replies):
            let reply = replies.last
            message = reply?.message
            print("Generated tokens: \(reply?.stats.tokensCountOutput ?? 0)")
        case .error(let error):
            print("Error: \(error)")
        }
    }
    print("Reasoning: \(message?.reasoning() ?? "empty")")
    print("Text: \(message?.text() ?? "empty")")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSessionStreamChunkError, ChatSessionStreamChunkReplies, Engine, EngineConfig } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];
    let session = await engine.chat(model, ChatConfig.create());
    let stream = await session.replyWithStream(messages, ChatReplyConfig.create());
    let message: ChatMessage | undefined;
    for await (const chunk of stream) {
        if (chunk instanceof ChatSessionStreamChunkReplies) {
            message = chunk.replies[0]?.message;
            console.log('Generated tokens: ', chunk.replies[0]?.stats.tokensCountOutput);
        } else if (chunk instanceof ChatSessionStreamChunkError) {
            console.error('Error: ', chunk.error);
        }
    }
    console.log('Reasoning: ', message?.reasoning);
    console.log('Text: ', message?.text);
}

main().catch((error) => {
    console.error(error);
});
```

</details>


<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::ReasoningEffort,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default().with_openai_api_key("OPENAI_API_KEY".to_string());
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("gpt-5".to_string()).await?.ok_or("Model not found")?;

    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Low),
        ChatMessage::user().with_text("How LLMs work".to_string()),
    ];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.first() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort


async def main() -> None:
    engine_config = EngineConfig.create().with_openai_api_key("OPENAI_API_KEY")
    engine = await Engine.create(engine_config)

    model = await engine.model("gpt-5")
    if model is None:
        raise RuntimeError("Model not found")

    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Low),
        ChatMessage.user().with_text("How LLMs work"),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(messages, ChatReplyConfig.create())
    if replies:
        message = replies[0].message
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Uzu

public func runChatCloud() async throws {
    let engineConfig = EngineConfig.create().withOpenaiApiKey(openaiApiKey: "OPENAI_API_KEY")
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .low),
        ChatMessage.user().withText(text: "How LLMs work")
    ]
    
    let session = try await engine.chat(model: model, config: .create())
    let reply = try await session.reply(input: messages, config: .create())
    guard let message = reply.last?.message else {
        return
    }
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create().withOpenaiApiKey('OPENAI_API_KEY');
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('gpt-5');
    if (!model) {
        throw new Error('Model not found');
    }

    let messages = [
        ChatMessage.system().withReasoningEffort("Low" as ReasoningEffort),
        ChatMessage.user().withText('How LLMs work')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;
    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
```

</details>


### Chat using speculation preset for classification

In this example, we will use the `classification` speculation preset to determine the sentiment of the user's input:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{Feature, ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPreset},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let feature = Feature {
        name: "sentiment".to_string(),
        values: vec![
            "Happy".to_string(),
            "Sad".to_string(),
            "Angry".to_string(),
            "Fearful".to_string(),
            "Surprised".to_string(),
            "Disgusted".to_string(),
        ],
    };
    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Classification {
        feature: feature.clone(),
    }));
    let session = engine.chat(model, chat_config).await?;

    let text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    let prompt = format!(
        "Text is: \"{text_to_detect_feature}\". Choose {} from the list: {}. Answer with one word. Don't add a dot at the end.",
        feature.name,
        feature.values.join(", ")
    );
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(prompt),
    ];

    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(32)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        println!("Prediction: {}", reply.message.text().unwrap_or_default());
        println!("Generated tokens: {}", reply.stats.tokens_count_output.unwrap_or_default());
    }

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSpeculationPreset,
    Engine,
    EngineConfig,
    Feature,
    ReasoningEffort,
    SamplingMethod,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    feature = Feature(
        "sentiment",
        ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted"],
    )
    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Classification(feature))
    session = await engine.chat(model, chat_config)

    text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling."
    prompt = (
        f'Text is: "{text_to_detect_feature}". '
        f"Choose {feature.name} from the list: {', '.join(feature.values)}. "
        "Answer with one word. Don't add a dot at the end."
    )
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(prompt),
    ]

    chat_reply_config = ChatReplyConfig.create().with_token_limit(32).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    if replies:
        reply = replies[0]
        print(f"Prediction: {reply.message.text}")
        print(f"Generated tokens: {reply.stats.tokens_count_output}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Uzu

public func runChatSpeculationClassification() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let feature = Feature(name: "sentiment", values: [
        "Happy",
        "Sad",
        "Angry",
        "Fearful",
        "Surprised",
        "Disgusted",
    ])
    let chatConfig = ChatConfig.create().withSpeculationPreset(speculationPreset: .classification(feature: feature))
    let session = try await engine.chat(model: model, config: chatConfig)
    
    let textToDetectFeature =
            "Today's been awesome! Everything just feels right, and I can't stop smiling."
    let prompt = "Text is: \"\(textToDetectFeature)\". Choose \(feature.name) from the list: \(feature.values.joined(separator: ", ")). Answer with one word. Don't add a dot at the end."
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: prompt)
    ]
    
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(tokenLimit: 32).withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: chatReplyConfig)
    guard let reply = replies.last else {
        return
    }
    
    print("Prediction: \(reply.message.text() ?? "empty")")
    print("Generated tokens: \(reply.stats.tokensCountOutput ?? 0)")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetClassification, Engine, EngineConfig, Feature, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const feature = new Feature('sentiment', [
        'Happy',
        'Sad',
        'Angry',
        'Fearful',
        'Surprised',
        'Disgusted',
    ]);
    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetClassification(feature));
    let session = await engine.chat(model, chatConfig);

    const textToDetectFeature =
        "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    const prompt =
        `Text is: "${textToDetectFeature}". Choose ${feature.name} from the list: ${feature.values.join(', ')}. ` +
        "Answer with one word. Don't add a dot at the end.";
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];

    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(32).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];

    if (reply) {
        console.log('Prediction: ', reply.message.text);
        console.log('Generated tokens: ', reply.stats.tokensCountOutput);
    }
}

main().catch((error) => {
    console.error(error);
});
```

</details>


<br>You can view the stats to see that the answer will be ready immediately after the prefill step, and actual generation won’t even start due to speculative decoding, which significantly improves generation speed.

### Chat using speculation preset for summarization

In this example, we will use the `summarization` speculation preset to generate a summary of the input text:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPreset},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let text_to_summarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. \
        It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. \
        LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. \
        These models have a wide range of applications, including chatbots, content creation, translation, and code generation. \
        One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. \
        They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. \
        Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. \
        As these models grow in size and sophistication, they continue to enhance human-computer interactions, \
        making AI-powered communication more natural and effective.";
    let prompt = format!("Text is: \"{text_to_summarize}\". Write only summary itself.");
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(prompt),
    ];

    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Summarization {}));
    let session = engine.chat(model, chat_config).await?;

    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(256)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        println!("Summary: {}", reply.message.text().unwrap_or_default());
        println!("Generation t/s: {}", reply.stats.generate_tokens_per_second.unwrap_or_default());
    }

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSpeculationPreset,
    Engine,
    EngineConfig,
    ReasoningEffort,
    SamplingMethod,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    text_to_summarize = (
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. "
        "It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. "
        "LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. "
        "These models have a wide range of applications, including chatbots, content creation, translation, and code generation. "
        "One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. "
        "They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. "
        "Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. "
        "As these models grow in size and sophistication, they continue to enhance human-computer interactions, "
        "making AI-powered communication more natural and effective."
    )
    prompt = f'Text is: "{text_to_summarize}". Write only summary itself.'
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(prompt),
    ]

    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Summarization())
    session = await engine.chat(model, chat_config)

    chat_reply_config = ChatReplyConfig.create().with_token_limit(256).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    if replies:
        reply = replies[0]
        print(f"Summary: {reply.message.text}")
        print(f"Generation t/s: {reply.stats.generate_tokens_per_second}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Uzu

public func runChatSpeculationSummarization() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let textToSummarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    let prompt = "Text is: \"\(textToSummarize)\". Write only summary itself."
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: prompt)
    ]
    
    let chatConfig = ChatConfig.create().withSpeculationPreset(speculationPreset: .summarization)
    let session = try await engine.chat(model: model, config: chatConfig)
    
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(tokenLimit: 256).withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: chatReplyConfig)
    guard let reply = replies.last else {
        return
    }
    
    print("Summary: \(reply.message.text() ?? "empty")")
    print("Generation t\\s: \(reply.stats.generateTokensPerSecond ?? 0.0)")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetSummarization, Engine, EngineConfig, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const textToSummarize =
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    const prompt = `Text is: "${textToSummarize}". Write only summary itself.`;
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];

    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetSummarization);
    let session = await engine.chat(model, chatConfig);

    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(256).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];

    if (reply) {
        console.log('Summary: ', reply.message.text);
        console.log('Generation t\\s: ', reply.stats.generateTokensPerSecond);
    }
}

main().catch((error) => {
    console.error(error);
});
```

</details>


<br>You will notice that the model’s run count is lower than the actual number of generated tokens due to speculative decoding, which significantly improves generation speed.

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

<details>
<summary>Rust</summary>

```rust
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{Grammar, ReasoningEffort},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Country {
    name: String,
    capital: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CountryList {
    countries: Vec<Country>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let schema_string = serde_json::to_string(&schema_for!(CountryList))?;
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields"
                .to_string(),
        ),
    ];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let chat_reply_config = ChatReplyConfig::default().with_grammar(Some(Grammar::JsonSchema {
        schema: schema_string,
    }));
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        if let Some(text) = reply.message.text() {
            let parsed: CountryList = serde_json::from_str(&text)?;
            println!("{parsed:#?}");
        }
    }

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio
import json

from pydantic import BaseModel

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    Grammar,
    ReasoningEffort,
)


class Country(BaseModel):
    name: str
    capital: str


class CountryList(BaseModel):
    countries: list[Country]


def structured_response(response: str | None, model_type: type[BaseModel]) -> BaseModel | None:
    if not response:
        return None
    return model_type.model_validate_json(response)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    schema_string = json.dumps(CountryList.model_json_schema())
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields"
        ),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(
        messages,
        ChatReplyConfig.create().with_grammar(Grammar.JsonSchema(schema_string)),
    )
    if replies:
        countries = structured_response(replies[0].message.text, CountryList)
        print(countries)


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import FoundationModels
import Uzu

@Generable()
struct Country: Codable {
    let name: String
    let capital: String
}

public func runChatStructuredOutput() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields")
    ]
    
    let session = try await engine.chat(model: model, config: .create())
    let reply = try await session.reply(input: messages, config: .create().withGrammar(grammar: .fromType([Country].self)))
    guard let message = reply.last?.message else {
        return
    }
    guard let countries: [Country] = message.textDecoded() else {
        return
    }
    print(countries)
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, GrammarJsonSchema, ReasoningEffort } from '@trymirai/uzu';
import * as z from "zod";

const CountryType = z.object({
    name: z.string(),
    capital: z.string(),
});
const CountryListType = z.array(CountryType);

function structuredResponse<T extends z.ZodType>(response: string | null | undefined, type: T): z.infer<T> | undefined {
    if (!response) {
        return undefined;
    }
    const data = JSON.parse(response);
    const result = type.parse(data);
    return result;
}

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let schema = z.toJSONSchema(CountryListType);
    let schemaString = JSON.stringify(schema);
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText('Give me a JSON object containing a list of 3 countries, where each country has name and capital fields')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create().withGrammar(new GrammarJsonSchema(schemaString)));
    let message = reply[0]?.message;
    let countries = structuredResponse(message?.text, CountryListType);
    console.log(countries);
}

main().catch((error) => {
    console.error(error);
});
```

</details>


### Classification

In this example, we will use a classification model to determine whether the user's input is safe from a moderation perspective:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::classification::ClassificationMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("trymirai/chat-moderation-router".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let messages = vec![ClassificationMessage::user("Hi".to_string())];

    let session = engine.classification(model).await?;
    let output = session.classify(messages).await?;
    println!("Output: {:?}", output.probabilities.values);

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio

from uzu import ClassificationMessage, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("trymirai/chat-moderation-router")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    messages = [ClassificationMessage.user("Hi")]

    session = await engine.classification(model)
    output = await session.classify(messages)
    print(f"Output: {output.probabilities.values}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Uzu

public func runClassification() async throws {
    let engine = try await Engine.create(config: .create())
    
    guard let model = try await engine.model(identifier: "trymirai/chat-moderation-router") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let messages = [
        ClassificationMessage.user(content: "Hi")
    ]
    
    let session = try await engine.classification(model: model)
    let output = try await session.classify(input: messages)
    print("Output: \(output.probabilities.values)")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { ClassificationMessage, Engine, EngineConfig } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('trymirai/chat-moderation-router');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let messages = [
        ClassificationMessage.user('Hi')
    ];

    let session = await engine.classification(model);
    let output = await session.classify(messages);
    console.log('Output: ', output.probabilities.values);
}

main().catch((error) => {
    console.error(error);
});
```

</details>


### Text to Speech

In this example, we will generate audio from text:

<details>
<summary>Rust</summary>

```rust
use uzu::{
    engine::{Engine, EngineConfig},
    session::text_to_speech::TextToSpeechSessionStreamChunk,
    types::basic::PcmBatch,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("fishaudio/s1-mini".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let text = "London is the capital of United Kingdom and one of the world's most influential cities, \
        known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. \
        Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace \
        with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum \
        and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.";
    let output_path = dirs::home_dir().ok_or("Home not found")?.join("Desktop").join("output.wav");

    let session = engine.text_to_speech(model).await?;
    let stream = session.synthesize_stream(text.to_string()).await;
    let mut pcm_batches: Vec<PcmBatch> = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            TextToSpeechSessionStreamChunk::PcmBatch {
                batch,
            } => {
                pcm_batches.push(batch);
            },
            TextToSpeechSessionStreamChunk::Error {
                error,
            } => {
                println!("Error: {error}");
            },
        }
    }

    let pcm_batch_first = pcm_batches.first().ok_or("No batches")?;
    let pcm_batch_full = PcmBatch {
        samples: pcm_batches.iter().flat_map(|batch| batch.samples.iter().copied()).collect(),
        sample_rate: pcm_batch_first.sample_rate,
        channels: pcm_batch_first.channels,
        lengths: vec![pcm_batches.iter().flat_map(|batch| batch.lengths.iter().copied()).sum()],
    };
    pcm_batch_full.save_as_wav(output_path.to_string_lossy().to_string())?;
    println!("Output saved to: {}", output_path.display());

    Ok(())
}
```

</details>

<details>
<summary>Python</summary>

```python
import asyncio
from pathlib import Path

from uzu import Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("fishaudio/s1-mini")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    text = (
        "London is the capital of United Kingdom and one of the world's most influential cities, "
        "known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. "
        "Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace "
        "with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum "
        "and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year."
    )
    output_path = Path.home() / "Desktop" / "output.wav"
    session = await engine.text_to_speech(model)
    pcm_batch = await session.synthesize(text)
    pcm_batch.save_as_wav(str(output_path))
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

<details>
<summary>Swift</summary>

```swift
import Foundation
import Uzu

public func runTextToSpeech() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "fishaudio/s1-mini") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let text = "London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year."
    let outputPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Desktop")
        .appendingPathComponent("output.wav")
    let session = try await engine.textToSpeech(model: model)
    let pcmBatch = try await session.synthesize(input: text)
    try pcmBatch.saveAsWav(path: outputPath.path())
    print("Output saved to: \(outputPath.path())")
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { Engine, EngineConfig } from '@trymirai/uzu';
import { homedir } from "os";
import { join } from "path";

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('fishaudio/s1-mini');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const text = "London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.";
    const outputPath = join(homedir(), "Desktop", "output.wav");
    let session = await engine.textToSpeech(model);
    let pcmBatch = await session.synthesize(text);
    pcmBatch.saveAsWav(outputPath);
    console.log('Output saved to: ', outputPath);
}

main().catch((error) => {
    console.error(error);
});
```

</details>



## Development

`uzu` is a native Rust crate with bindings available for:

- `Swift` via [uniffi-rs](https://github.com/mozilla/uniffi-rs)
- `Python` via [pyo3](https://github.com/PyO3/pyo3)
- `TypeScript` via [napi-rs](https://github.com/napi-rs/napi-rs)

It supports:

- Backends:
    - `metal`
    - `cpu`
- Targets:
    - `aarch64-apple-darwin`
    - `aarch64-apple-ios`
    - `aarch64-apple-ios-sim`
    - `aarch64-pc-windows-msvc` _(in progress)_
    - `aarch64-unknown-linux-gnu` _(in progress)_
    - `wasm32-wasip1-threads` _(in progress)_
    - `x86_64-apple-darwin`
    - `x86_64-pc-windows-msvc` _(in progress)_
    - `x86_64-unknown-linux-gnu` _(in progress)_

<br>
For initial setup we recommend running <code>cargo tools setup</code>, which installs all necessary dependencies (<code>rustup</code>, <code>uv</code>, <code>pnpm</code>, <code>Rust targets</code>, <code>Metal toolchain</code>, ...) if not already present.

<br>
To unify cross-language development we introduce <code>cargo tools</code>:

- Install language specific dependencies: `cargo tools install typescript`
- Build: `cargo tools build rust --targets apple`
- Test: `cargo tools test python`
- Run example: `cargo tools example swift chat`

## Model Format

`uzu` uses its own model format. You can download a test model:

```bash
./scripts/download_test_model.sh
```

Or download any supported model that has already been converted:

```bash
cd ./tools/
uv run downloader list             # show the list of supported models
uv run downloader download {REPO}  # download a specific model
```

Models downloaded for development are stored at `./workspace/models/0.4.3/`.

You can also export a model yourself with [lalamo](https://github.com/trymirai/lalamo):

```bash
git clone https://github.com/trymirai/lalamo.git
cd lalamo
uv run lalamo list-models
uv run lalamo convert meta-llama/Llama-3.2-1B-Instruct
```

## CLI

You can run `uzu` in [CLI](https://docs.trymirai.com/overview/cli) mode:

```bash
cargo run --release -p cli -- help
```

```text
Usage: cli [COMMAND]

Commands:
  run    Run a model with the specified path
  serve  Start a server with the specified model path
  bench  Run benchmarks for the specified model
  help   Print this message or the help of the given subcommand(s)
```

## Benchmarks

To run benchmarks:

```bash
cargo run --release -p cli -- bench ./workspace/models/0.4.3/{MODEL_NAME} ./workspace/models/0.4.3/{MODEL_NAME}/benchmark_task.json ./workspace/models/0.4.3/{MODEL_NAME}/benchmark_result.json
```

`benchmark_task.json` is automatically generated after the model is downloaded via `./tools/`.



## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
