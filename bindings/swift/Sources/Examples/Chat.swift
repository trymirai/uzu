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
