import Uzu

public func runCloud() async throws {
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
