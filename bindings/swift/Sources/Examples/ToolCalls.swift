import FoundationModels
import Uzu

@Generable
private struct Coordinate: Codable, Sendable {
    @Guide(description: "Latitude in decimal degrees.")
    let latitude: Double

    @Guide(description: "Longitude in decimal degrees.")
    let longitude: Double
}

private struct GetCurrentLocationTool: Tool {
    let description = "Return the current location in coordinates."

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> Coordinate {
        Coordinate(latitude: 51.5074, longitude: -0.1278)
    }
}

private struct GetCurrentTemperatureTool: Tool {
    let description = "Return the temperature at the provided coordinates."

    func call(arguments: Coordinate) async throws -> Double {
        _ = arguments
        return 25.0
    }
}

public func runToolCalls() async throws {
    let engine = try await Engine.create(config: .create())
    guard let model = try await engine.model(identifier: "mlx-community/Qwen3.5-9B-MLX-8bit") else {
        throw ToolCallsExampleError.modelNotFound
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }

    let session = try await engine.chat(model: model, config: .create())
    try await session.addTool(GetCurrentLocationTool())
    try await session.addTool(GetCurrentTemperatureTool())

    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "What temperature is it now at my location?"),
    ]
    let replies = try await session.reply(input: messages, config: .create())
    guard let message = replies.last?.message else {
        return
    }

    print("Reasoning: \(message.reasoning() ?? "")")
    print("Text: \(message.text() ?? "")")
}

private enum ToolCallsExampleError: Swift.Error {
    case modelNotFound
}
