import XCTest
@testable import Uzu

@UzuToolFunction
private func echoWithoutFoundationModelsImport(value: String) -> String {
    value
}

private struct Prefixer: Sendable {
    let prefix: String

    @UzuToolFunction
    func prefix(value: String) -> String {
        prefix + value
    }
}

private enum StaticTools {
    @UzuToolFunction
    static func answer() -> Int {
        42
    }
}

private actor Counter {
    @UzuToolFunction
    func adding(value: Int) -> Int {
        40 + value
    }
}

final class ToolMacroImportTests: XCTestCase {
    func testMacroIsSelfContainedInUzuModule() async throws {
        let result = try await echoWithoutFoundationModelsImportTool.invoke(
            argumentsJson: #"{"value":"hello"}"#
        )
        XCTAssertEqual(result, #""hello""#)
    }

    func testMacroSupportsInstanceAndStaticFunctions() async throws {
        let prefixed = try await Prefixer(prefix: "Hello, ").prefixTool.invoke(
            argumentsJson: #"{"value":"Ada"}"#
        )
        XCTAssertEqual(prefixed, #""Hello, Ada""#)

        let answer = try await StaticTools.answerTool.invoke(argumentsJson: "{}")
        XCTAssertEqual(answer, "42")

        let actorTool = await Counter().addingTool
        let actorResult = try await actorTool.invoke(argumentsJson: #"{"value":2}"#)
        XCTAssertEqual(actorResult, "42")
    }
}
