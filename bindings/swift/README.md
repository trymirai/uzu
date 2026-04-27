<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-swift-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord&color=brightgreen" alt="Discord"></a> <a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-brightgreen" alt="Contact us"></a> <a href="https://docs.trymirai.com"><img src="https://img.shields.io/badge/Read-Docs-brightgreen" alt="Read docs"></a> [![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE) ![Build](https://github.com/trymirai/uzu/actions/workflows/tests.yml/badge.svg) ![Swift](https://img.shields.io/badge/Swift-blue) ![SPM](https://img.shields.io/badge/SPM-compatible-blue) ![Platforms](https://img.shields.io/badge/Platforms-iOS%20%7C%20macOS-blue) [![Swift](https://img.shields.io/badge/Swift-5.9-blue)](https://swift.org) 

# uzu

A high-performance inference engine for AI models. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. Key features:

- Simple, high-level API
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices
- [Broad model support](https://trymirai.com/models)

## Quick Start



Add the dependency:

```swift
dependencies: [
    .package(url: "https://github.com/trymirai/uzu.git", from: "0.1.9")
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


<br>
Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<**swift**\> \<**chat** | **chat-cloud** | **chat-speculation-classification** | **chat-speculation-summarization** | **chat-structured-output** | **classification** | **quick-start** | **text-to-speech**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

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

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

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

### Chat using speculation preset for classification

In this example, we will use the `classification` speculation preset to determine the sentiment of the user's input:

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

<br>You can view the stats to see that the answer will be ready immediately after the prefill step, and actual generation won’t even start due to speculative decoding, which significantly improves generation speed.

### Chat using speculation preset for summarization

In this example, we will use the `summarization` speculation preset to generate a summary of the input text:

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

<br>You will notice that the model’s run count is lower than the actual number of generated tokens due to speculative decoding, which significantly improves generation speed.

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

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

### Classification

In this example, we will use a classification model to determine whether the user's input is safe from a moderation perspective:

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

### Text to Speech

In this example, we will generate audio from text:

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



## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
