import Foundation
import FoundationModels
import XCTest
@testable import Uzu

@Generable
private struct AdditionArguments: Codable, Sendable {
    let left: Int
    let right: Int
}

@Sendable private func addReference(_ arguments: AdditionArguments) -> Int {
    arguments.left + arguments.right
}

private enum ExpectedToolError: Swift.Error {
    case failed
}

@Sendable private func throwingReference(_ arguments: AdditionArguments) throws -> Int {
    _ = arguments
    throw ExpectedToolError.failed
}

/// Return a greeting, or no result when no name is provided.
@UzuToolFunction(name: "optional_greeting")
private func optionalGreeting(name: String?) -> String? {
    name.map { "Hello, \($0)!" }
}

final class ToolTests: XCTestCase {
    func testTypedToolBuildsDefinitionAndInvokesHandler() async throws {
        let tool = UzuToolDescriptor<AdditionArguments, Int>(
            name: "add",
            description: "Add two integers",
            parameters: AdditionArguments.self,
            returning: Int.self
        ) { arguments in
            arguments.left + arguments.right
        }

        XCTAssertEqual(tool.definition.name, "add")
        XCTAssertEqual(tool.definition.description, "Add two integers")

        let parameters = try XCTUnwrap(tool.definition.parameters)
        let schema = try JSONSerialization.jsonObject(with: Data(parameters.json.utf8)) as? [String: Any]
        XCTAssertEqual(schema?["type"] as? String, "object")

        let resultJson = try await tool.invoke(argumentsJson: #"{"left":20,"right":22}"#)
        XCTAssertEqual(resultJson, "42")
        let result = try await tool.invoke(AdditionArguments(left: 10, right: 5))
        XCTAssertEqual(result, 15)
    }

    func testFunctionReferencesAndErrors() async throws {
        let add = UzuToolDescriptor<AdditionArguments, Int>(
            name: "add_reference",
            parameters: AdditionArguments.self,
            returning: Int.self,
            handler: addReference
        )
        let result = try await add.invoke(AdditionArguments(left: 19, right: 23))
        XCTAssertEqual(result, 42)

        let failing = UzuToolDescriptor<AdditionArguments, Int>(
            name: "throwing_reference",
            parameters: AdditionArguments.self,
            returning: Int.self,
            handler: throwingReference
        )
        do {
            _ = try await failing.invoke(AdditionArguments(left: 1, right: 2))
            XCTFail("Expected the function reference to throw")
        } catch is ExpectedToolError {
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testMacroSupportsOptionalParametersAndResults() async throws {
        XCTAssertEqual(optionalGreetingTool.definition.name, "optional_greeting")
        XCTAssertEqual(
            optionalGreetingTool.definition.description,
            "Return a greeting, or no result when no name is provided."
        )

        let greeting = try await optionalGreetingTool.invoke(argumentsJson: #"{"name":"Ada"}"#)
        XCTAssertEqual(greeting, #""Hello, Ada!""#)

        let noGreeting = try await optionalGreetingTool.invoke(argumentsJson: #"{"name":null}"#)
        XCTAssertEqual(noGreeting, "null")
        XCTAssertTrue(optionalGreetingTool.definition.returnDefinition?.json.contains("anyOf") == true)
    }

    func testParameterlessToolAcceptsOnlyEmptyArguments() async throws {
        let tool = UzuToolDescriptor<Void, String>(
            name: "current_location",
            returning: String.self
        ) {
            "London"
        }

        let resultJson = try await tool.invoke(argumentsJson: "{}")
        XCTAssertEqual(resultJson, #""London""#)

        do {
            _ = try await tool.invoke(argumentsJson: #"{"unexpected":true}"#)
            XCTFail("Expected non-empty arguments to fail")
        } catch let error as UzuToolError {
            guard case .unexpectedArguments = error else {
                return XCTFail("Unexpected tool error: \(error)")
            }
        }
    }

    func testVoidResultIsEncodedAsNull() async throws {
        let tool = UzuToolDescriptor<AdditionArguments, Void>(
            name: "record_addition",
            parameters: AdditionArguments.self
        ) { _ in }

        XCTAssertNil(tool.definition.returnDefinition)
        let resultJson = try await tool.invoke(argumentsJson: #"{"left":1,"right":2}"#)
        XCTAssertEqual(resultJson, "null")
    }

    func testRawToolUsesValueJson() async throws {
        let definition = ToolFunction(
            name: "echo",
            description: "Echo JSON",
            parameters: Value(json: #"{"type":"object"}"#),
            returnDefinition: Value(json: #"{}"#)
        )
        let tool = UzuRawToolFunction(definition: definition) { arguments in
            arguments
        }

        let json = #"{"message":"hello"}"#
        let resultJson = try await tool.invoke(argumentsJson: json)
        XCTAssertEqual(resultJson, json)
    }
}
