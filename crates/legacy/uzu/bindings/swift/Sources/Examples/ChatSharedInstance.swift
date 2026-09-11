import Foundation
import Uzu

public func runChatSharedInstance() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)

    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()

    // The chatInstance owns the loaded model and can be shared between sessions.
    let chatInstance = try await engine.chatInstance(model: model, config: .create())

    let firstSession = try await engine.chatWithInstance(instance: chatInstance)
    let replies = try await firstSession.reply(
        input: [ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")],
        config: .create()
    )
    if let message = replies.last?.message {
        print("First session reasoning: \(message.reasoning() ?? "")")
        print("First session text: \(message.text() ?? "")")
    }

    // The second session reuses the already-loaded weights instead of loading the model again.
    let secondSession = try await engine.chatWithInstance(instance: chatInstance)
    let secondReplies = try await secondSession.reply(
        input: [ChatMessage.user().withText(text: "What is the capital of France?")],
        config: .create()
    )
    if let message = secondReplies.last?.message {
        print("\nSecond session reasoning: \(message.reasoning() ?? "")")
        print("Second session text: \(message.text() ?? "")")
    }
}
