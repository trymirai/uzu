import Foundation
import Uzu

public func runChatCloud() async throws {
    let engineConfig = EngineConfig.create().withOpenaiApiKey(openaiApiKey: "OPENAI_API_KEY")
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()
    
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
