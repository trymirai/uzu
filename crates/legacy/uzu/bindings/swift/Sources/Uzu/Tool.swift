import Foundation
import FoundationModels

public enum FoundationModelsToolError: Error, Equatable, LocalizedError, Sendable {
    case parametersMustBeObject(toolName: String)

    public var errorDescription: String? {
        switch self {
        case .parametersMustBeObject(let toolName):
            return "FoundationModels tool '\(toolName)' must describe its parameters with an object schema"
        }
    }
}

extension ChatSession {
    public func addTool<T: FoundationModels.Tool>(_ tool: T) async throws
    where T.Output: FoundationModels.ConvertibleToGeneratedContent {
        let definition = try foundationModelsToolDefinition(for: tool)
        try await addForeignTool(
            tool: ForeignTool(
                definition: definition,
                handler: FoundationModelsToolHandler(tool: tool)
            )
        )
    }
}

final class FoundationModelsToolHandler<T: FoundationModels.Tool>: ForeignToolHandler,
    @unchecked Sendable
where T.Output: FoundationModels.ConvertibleToGeneratedContent {
    private let tool: T

    init(tool: T) {
        self.tool = tool
    }

    func invokeJson(argumentsJson: String) async throws -> String {
        do {
            let content = try FoundationModels.GeneratedContent(json: argumentsJson)
            let arguments = try T.Arguments(content)
            let output = try await tool.call(arguments: arguments)
            return output.generatedContent.jsonString
        } catch let error as ForeignToolError {
            throw error
        } catch {
            throw ForeignToolError.Invocation(message: String(describing: error))
        }
    }
}

func foundationModelsToolDefinition<T: FoundationModels.Tool>(for tool: T) throws -> ToolFunction {
    let parametersJson = tool.parameters.debugDescription
    let parameters = try JSONSerialization.jsonObject(with: Data(parametersJson.utf8))
    guard let schema = parameters as? [String: Any], schema["type"] as? String == "object" else {
        throw FoundationModelsToolError.parametersMustBeObject(toolName: tool.name)
    }

    return ToolFunction(
        name: tool.name,
        description: tool.description,
        parameters: Value(json: parametersJson),
        returnDefinition: nil
    )
}
