import Uzu

@MainActor public func exampleQuickStart() async throws {
    // snippet:quick-start
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
    // endsnippet:quick-start
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}

@MainActor public func exampleChat() async throws {
    // snippet:engine-create
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    // endsnippet:engine-create
    
    // snippet:model-choose
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    // endsnippet:model-choose
    
    // snippet:model-download
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    // endsnippet:model-download
    
    // snippet:session-create-general
    let session = try await engine.chat(model: model, config: .create())
    // endsnippet:session-create-general
    
    // snippet:session-input-general
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")
    ]
    // endsnippet:session-input-general
    
    // snippet:session-run-general
    let reply = try await session.reply(input: messages, config: .create())
    guard let message = reply.last?.message else {
        return
    }
    // endsnippet:session-run-general
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}

@MainActor public func exampleSummarization() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    // snippet:session-input-summarization
    let textToSummarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    let prompt = "Text is: \"\(textToSummarize)\". Write only summary itself."
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: prompt)
    ]
    // endsnippet:session-input-summarization
    
    // snippet:session-create-summarization
    let chatConfig = ChatConfig.create().withSpeculationPreset(speculationPreset: .summarization)
    let session = try await engine.chat(model: model, config: chatConfig)
    // endsnippet:session-create-summarization
    
    // snippet:session-run-summarization
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(tokenLimit: 256).withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: chatReplyConfig)
    guard let reply = replies.last else {
        return
    }
    // endsnippet:session-run-summarization
    
    print("Summary: \(reply.message.text() ?? "empty")")
    print("Generation t\\s: \(reply.stats.generateTokensPerSecond ?? 0.0)")
}

@MainActor public func exampleClassification() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    // snippet:session-create-classification
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
    // endsnippet:session-create-classification
    
    // snippet:session-input-classification
    let textToDetectFeature =
            "Today's been awesome! Everything just feels right, and I can't stop smiling."
    let prompt = "Text is: \"\(textToDetectFeature)\". Choose \(feature.name) from the list: \(feature.values.joined(separator: ", ")). Answer with one word. Don't add a dot at the end."
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: prompt)
    ]
    // endsnippet:session-input-classification
    
    // snippet:session-run-classification
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(tokenLimit: 32).withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: chatReplyConfig)
    guard let reply = replies.last else {
        return
    }
    // endsnippet:session-run-classification
    
    print("Prediction: \(reply.message.text() ?? "empty")")
    print("Generated tokens: \(reply.stats.tokensCountOutput ?? 0)")
}

@MainActor public func runSnippets() async throws {
    try await exampleQuickStart()
    try await exampleChat()
    try await exampleSummarization()
    try await exampleClassification()
}
