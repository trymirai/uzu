import Foundation
import FoundationModels
import Uzu

@Generable
private struct Coordinate: Codable, Sendable {
    @Guide(description: "Latitude in decimal degrees.")
    let latitude: Double

    @Guide(description: "Longitude in decimal degrees.")
    let longitude: Double
}

private struct GetCurrentLocation: Tool {
    let description = "Returns current location in coordinates"

    @Generable
    struct Arguments {
    }

    func call(arguments: Arguments) async throws -> Coordinate {
        Coordinate(latitude: 51.5074, longitude: -0.1278)
    }
}

private struct GetCurrentTemperature: Tool {
    let description = "Returns temperature in provided location"

    func call(arguments: Coordinate) async throws -> Double {
        _ = arguments
        return 25.0
    }
}

public func runToolCalls() async throws {
    let engine = try await Engine.create(config: .create())
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        throw ToolCallsExampleError.modelNotFound
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\u{001B}[2K\nDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()

    let session = try await engine.chat(model: model, config: .create())
    try await session.addTool(GetCurrentLocation())
    try await session.addTool(GetCurrentTemperature())

    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "What temperature is it now at my location?"),
    ]
    let reply_config = ChatReplyConfig.create().withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: reply_config)
    guard let message = replies.last?.message else {
        return
    }

    print("Reasoning: \(message.reasoning() ?? "")")
    print("Text: \(message.text() ?? "")")
}

private enum ToolCallsExampleError: Swift.Error {
    case modelNotFound
}
