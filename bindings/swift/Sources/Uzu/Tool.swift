import Foundation
@_exported import FoundationModels

public protocol UzuTool: Sendable {
    var definition: ToolFunction { get }

    func invoke(argumentsJson: String) async throws -> String
}

public enum UzuToolError: Swift.Error, LocalizedError, Sendable {
    case unexpectedArguments

    public var errorDescription: String? {
        switch self {
        case .unexpectedArguments:
            "This tool does not accept arguments"
        }
    }
}

public final class UzuToolDescriptor<Parameters, Result>: UzuTool, @unchecked Sendable {
    public let definition: ToolFunction

    private let decodeArguments: @Sendable (String) throws -> Parameters
    private let encodeResult: @Sendable (Result) throws -> String
    private let handler: @Sendable (Parameters) async throws -> Result

    private init(
        definition: ToolFunction,
        decodeArguments: @escaping @Sendable (String) throws -> Parameters,
        encodeResult: @escaping @Sendable (Result) throws -> String,
        handler: @escaping @Sendable (Parameters) async throws -> Result
    ) {
        self.definition = definition
        self.decodeArguments = decodeArguments
        self.encodeResult = encodeResult
        self.handler = handler
    }

    public convenience init(
        name: String,
        description: String = "",
        parameters: Parameters.Type,
        returning: Result.Type,
        handler: @escaping @Sendable (Parameters) async throws -> Result
    ) where Parameters: Codable & Generable, Result: Codable & Generable {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: generationSchema(for: parameters),
                returns: generationSchema(for: returning)
            ),
            decodeArguments: { json in
                try JSONDecoder().decode(Parameters.self, from: Data(json.utf8))
            },
            encodeResult: { result in
                String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
            },
            handler: handler
        )
    }

    public convenience init<Wrapped>(
        name: String,
        description: String = "",
        parameters: Parameters.Type,
        returning: Result.Type,
        handler: @escaping @Sendable (Parameters) async throws -> Result
    ) where Parameters: Codable & Generable, Result == Wrapped?, Wrapped: Codable & Generable {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: generationSchema(for: parameters),
                returns: nullableGenerationSchema(for: Wrapped.self)
            ),
            decodeArguments: { json in
                try JSONDecoder().decode(Parameters.self, from: Data(json.utf8))
            },
            encodeResult: { result in
                String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
            },
            handler: handler
        )
    }

    public convenience init(
        name: String,
        description: String = "",
        returning: Result.Type,
        handler: @escaping @Sendable () async throws -> Result
    ) where Parameters == Void, Result: Codable & Generable {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: emptyParametersSchema,
                returns: generationSchema(for: returning)
            ),
            decodeArguments: { json in
                try requireEmptyArguments(json)
            },
            encodeResult: { result in
                String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
            },
            handler: { _ in try await handler() }
        )
    }

    public convenience init<Wrapped>(
        name: String,
        description: String = "",
        returning: Result.Type,
        handler: @escaping @Sendable () async throws -> Result
    ) where Parameters == Void, Result == Wrapped?, Wrapped: Codable & Generable {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: emptyParametersSchema,
                returns: nullableGenerationSchema(for: Wrapped.self)
            ),
            decodeArguments: { json in
                try requireEmptyArguments(json)
            },
            encodeResult: { result in
                String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
            },
            handler: { _ in try await handler() }
        )
    }

    public convenience init(
        name: String,
        description: String = "",
        parameters: Parameters.Type,
        handler: @escaping @Sendable (Parameters) async throws -> Void
    ) where Parameters: Codable & Generable, Result == Void {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: generationSchema(for: parameters),
                returns: nil
            ),
            decodeArguments: { json in
                try JSONDecoder().decode(Parameters.self, from: Data(json.utf8))
            },
            encodeResult: { _ in "null" },
            handler: handler
        )
    }

    public convenience init(
        name: String,
        description: String = "",
        handler: @escaping @Sendable () async throws -> Void
    ) where Parameters == Void, Result == Void {
        self.init(
            definition: toolDefinition(
                name: name,
                description: description,
                parameters: emptyParametersSchema,
                returns: nil
            ),
            decodeArguments: { json in
                try requireEmptyArguments(json)
            },
            encodeResult: { _ in "null" },
            handler: { _ in try await handler() }
        )
    }

    public func invoke(_ parameters: Parameters) async throws -> Result {
        try await handler(parameters)
    }

    public func invoke(argumentsJson: String) async throws -> String {
        let arguments = try decodeArguments(argumentsJson)
        return try encodeResult(await handler(arguments))
    }
}

extension UzuToolDescriptor where Parameters == Void {
    public func invoke() async throws -> Result {
        try await handler(())
    }
}

public final class UzuRawToolFunction: UzuTool, @unchecked Sendable {
    public let definition: ToolFunction

    private let handler: @Sendable (Value) async throws -> Value

    public init(
        definition: ToolFunction,
        handler: @escaping @Sendable (Value) async throws -> Value
    ) {
        self.definition = definition
        self.handler = handler
    }

    public func invoke(argumentsJson: String) async throws -> String {
        try await handler(Value(json: argumentsJson)).json
    }
}

extension ChatSession {
    public func addTool(_ tool: any UzuTool) async throws {
        try await addForeignTool(tool: foreignTool(for: tool))
    }

    public func addTools(_ tools: [any UzuTool]) async throws {
        try await addForeignTools(tools: tools.map(foreignTool(for:)))
    }
}

private final class UzuForeignToolHandler: ForeignToolHandler, @unchecked Sendable {
    private let tool: any UzuTool

    init(tool: any UzuTool) {
        self.tool = tool
    }

    func invokeJson(argumentsJson: String) async throws -> String {
        do {
            return try await tool.invoke(argumentsJson: argumentsJson)
        } catch let error as ForeignToolError {
            throw error
        } catch {
            throw ForeignToolError.Invocation(message: String(describing: error))
        }
    }
}

private func foreignTool(for tool: any UzuTool) -> ForeignTool {
    ForeignTool(
        definition: tool.definition,
        handler: UzuForeignToolHandler(tool: tool)
    )
}

private let emptyParametersSchema = Value(
    json: #"{"type":"object","properties":{},"required":[]}"#
)

private func generationSchema<T: Generable>(for type: T.Type) -> Value {
    Value(json: type.generationSchema.debugDescription)
}

private func nullableGenerationSchema<T: Generable>(for type: T.Type) -> Value {
    let wrapped = generationSchema(for: type)
    return Value(json: #"{"anyOf":["# + wrapped.json + #",{"type":"null"}]}"#)
}

private func toolDefinition(
    name: String,
    description: String,
    parameters: Value?,
    returns: Value?
) -> ToolFunction {
    precondition(!name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "Tool name must not be empty")
    return ToolFunction(
        name: name,
        description: description,
        parameters: parameters,
        returnDefinition: returns
    )
}

private func requireEmptyArguments(_ json: String) throws {
    let value = try JSONSerialization.jsonObject(with: Data(json.utf8))
    if value is NSNull {
        return
    }
    if let object = value as? [String: Any], object.isEmpty {
        return
    }
    throw UzuToolError.unexpectedArguments
}
