import Uzu

public func runChatForClassification() async throws {
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
