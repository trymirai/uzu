import FoundationModels
import Uzu

@Generable
private struct Coordinate: Codable, Sendable {
    @Guide(description: "Latitude in decimal degrees.")
    let latitude: Double

    @Guide(description: "Longitude in decimal degrees.")
    let longitude: Double
}

@UzuToolFunction(
    name: "get_current_location",
    description: "Return the current location in coordinates."
)
private func getCurrentLocation() -> Coordinate {
    Coordinate(latitude: 51.5074, longitude: -0.1278)
}

/// Return the temperature at the provided coordinates.
/// - Parameters:
///   - latitude: Latitude in decimal degrees.
///   - longitude: Longitude in decimal degrees.
@UzuToolFunction(name: "get_current_temperature")
private func getCurrentTemperature(
    latitude: Double,
    longitude: Double
) -> Double {
    _ = (latitude, longitude)
    return 25.0
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
    try await session.addTool(getCurrentLocationTool)
    try await session.addTool(getCurrentTemperatureTool)

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
