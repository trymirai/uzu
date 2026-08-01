import FoundationModels

extension ChatSession {
    public func addTool<T: FoundationModels.Tool>(_ tool: T) async throws {
        try await addForeignTool(
            tool: ForeignTool(
                definition: foundationModelsToolDefinition(for: tool),
                handler: FoundationModelsToolHandler(tool: tool)
            )
        )
    }
}

final class FoundationModelsToolHandler<T: FoundationModels.Tool>: ForeignToolHandler, @unchecked Sendable {
    private let tool: T

    init(tool: T) {
        self.tool = tool
    }

    func invokeJson(argumentsJson: String) async throws -> String {
        do {
            let content = try FoundationModels.GeneratedContent(json: argumentsJson)
            let arguments = try T.Arguments(content)
            let output = try await tool.call(arguments: arguments)
            guard let generatedOutput = output as? any FoundationModels.ConvertibleToGeneratedContent else {
                throw ForeignToolError.Invocation(
                    message: "FoundationModels tool output '\(String(reflecting: T.Output.self))' cannot be converted to generated content"
                )
            }
            return generatedOutput.generatedContent.jsonString
        } catch let error as ForeignToolError {
            throw error
        } catch {
            throw ForeignToolError.Invocation(message: String(describing: error))
        }
    }
}

func foundationModelsToolDefinition<T: FoundationModels.Tool>(for tool: T) -> ToolFunction {
    ToolFunction(
        name: tool.name,
        description: tool.description,
        parameters: Value(json: tool.parameters.debugDescription),
        returnDefinition: nil
    )
}
