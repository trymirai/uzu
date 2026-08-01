import Foundation
import FoundationModels
import XCTest
@testable import Uzu

@Generable
private struct Coordinate: Codable, Sendable {
    @Guide(description: "Latitude in decimal degrees.")
    let latitude: Double

    @Guide(description: "Longitude in decimal degrees.")
    let longitude: Double
}

@Generable
private struct EmptyInput {}

private struct GetCurrentLocationTool: Tool {
    let description = "Return the current location in coordinates."

    func call(arguments: EmptyInput) async throws -> Coordinate {
        Coordinate(latitude: 51.5074, longitude: -0.1278)
    }
}

private struct GetCurrentTemperatureTool: Tool {
    let name = "get_current_temperature"
    let description = "Return the temperature at the provided coordinates."

    func call(arguments: Coordinate) async throws -> Double {
        _ = arguments
        return 25.0
    }
}

final class ToolTests: XCTestCase {
    func testFoundationModelsToolBuildsDefinitionAndInvokesHandler() async throws {
        let tool = GetCurrentTemperatureTool()
        let definition = foundationModelsToolDefinition(for: tool)

        XCTAssertEqual(definition.name, "get_current_temperature")
        XCTAssertEqual(definition.description, "Return the temperature at the provided coordinates.")

        let parameters = try XCTUnwrap(definition.parameters)
        let schema = try JSONSerialization.jsonObject(with: Data(parameters.json.utf8)) as? [String: Any]
        XCTAssertEqual(schema?["type"] as? String, "object")

        let handler = FoundationModelsToolHandler(tool: tool)
        let resultJson = try await handler.invokeJson(
            argumentsJson: #"{"latitude":51.5074,"longitude":-0.1278}"#
        )
        XCTAssertEqual(resultJson, "25")
    }

    func testFoundationModelsToolSupportsEmptyArgumentsAndStructuredOutput() async throws {
        let tool = GetCurrentLocationTool()
        let definition = foundationModelsToolDefinition(for: tool)

        XCTAssertEqual(definition.name, "GetCurrentLocationTool")

        let handler = FoundationModelsToolHandler(tool: tool)
        let resultJson = try await handler.invokeJson(argumentsJson: "{}")
        let result = try JSONDecoder().decode(Coordinate.self, from: Data(resultJson.utf8))
        XCTAssertEqual(result.latitude, 51.5074)
        XCTAssertEqual(result.longitude, -0.1278)
    }
}
