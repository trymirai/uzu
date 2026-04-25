import FoundationModels
import Uzu

@Generable()
struct Country: Codable {
    let name: String
    let capital: String
}

public func runStructuredOutput() async throws {
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
