import Foundation
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct UzuToolFunctionMacro: PeerMacro {
    public static func expansion(
        of attribute: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        guard let function = declaration.as(FunctionDeclSyntax.self) else {
            throw UzuToolMacroError("@UzuToolFunction can only be attached to a function")
        }
        try validate(function: function, context: context)

        let options = try MacroOptions(attribute: attribute)
        let functionName = function.name.trimmedDescription
        let toolName = options.name ?? function.name.text
        let documentation = Documentation(function: function)
        let description = options.description ?? documentation.summary
        let argumentsType = "__UzuToolArguments_\(function.name.text)"
        let parameters = function.signature.parameterClause.parameters
        let returnType = function.signature.returnClause?.type.trimmedDescription ?? "Void"
        let returnsVoid = returnType == "Void" || returnType == "()"
        let isStatic = function.modifiers.contains { modifier in
            modifier.name.tokenKind == .keyword(.static) || modifier.name.tokenKind == .keyword(.class)
        }
        let lexicalParent = context.lexicalContext.first
        let isMember = lexicalParent.map(isTypeContext) ?? false
        let isActorMember = lexicalParent?.is(ActorDeclSyntax.self) == true && !isStatic
        let callTarget = isStatic ? "Self.\(functionName)" : isMember ? "self.\(functionName)" : functionName
        let callArguments = parameters.map { parameter in
            let propertyName = toolParameterName(parameter)
            let label = parameter.firstName.trimmedDescription
            return label == "_" ? "arguments.\(propertyName)" : "\(label): arguments.\(propertyName)"
        }
        let effects = function.signature.effectSpecifiers
        let tryPrefix = effects?.throwsClause != nil ? "try " : ""
        let awaitPrefix = effects?.asyncSpecifier != nil || isActorMember ? "await " : ""
        let call = "\(tryPrefix)\(awaitPrefix)\(callTarget)(\(callArguments.joined(separator: ", ")))"
        let access = accessPrefix(function.modifiers)
        let typePrefix = isStatic ? "static " : ""

        var peers: [DeclSyntax] = []
        if !parameters.isEmpty {
            let properties = parameters.map { parameter in
                let name = toolParameterName(parameter)
                let type = parameter.type.trimmedDescription
                if let guide = documentation.parameters[unescaped(name)] {
                    return "    @Guide(description: \(swiftStringLiteral(guide)))\n    let \(name): \(type)"
                }
                return "    let \(name): \(type)"
            }
            peers.append(
                DeclSyntax(
                    stringLiteral: """
                    @Generable
                    fileprivate struct \(argumentsType): Codable, Sendable {
                    \(properties.joined(separator: "\n\n"))
                    }
                    """
                )
            )
        }

        let descriptorType = parameters.isEmpty ? "Void" : argumentsType
        var initializerArguments = [
            "name: \(swiftStringLiteral(toolName))",
            "description: \(swiftStringLiteral(description))",
        ]
        if !parameters.isEmpty {
            initializerArguments.append("parameters: \(argumentsType).self")
        }
        if !returnsVoid {
            initializerArguments.append("returning: (\(returnType)).self")
        }
        let closureParameter = parameters.isEmpty ? "" : " arguments in"
        let descriptor = """
        \(access)\(typePrefix)var \(function.name.text)Tool: some UzuTool {
            UzuToolDescriptor<\(descriptorType), \(returnsVoid ? "Void" : returnType)>(
                \(initializerArguments.joined(separator: ",\n        "))
            ) {\(closureParameter)
                \(call)
            }
        }
        """
        peers.append(DeclSyntax(stringLiteral: descriptor))
        return peers
    }
}

private struct MacroOptions {
    let name: String?
    let description: String?

    init(attribute: AttributeSyntax) throws {
        guard case let .argumentList(arguments) = attribute.arguments else {
            name = nil
            description = nil
            return
        }
        name = try Self.stringArgument(named: "name", in: arguments)
        description = try Self.stringArgument(named: "description", in: arguments)
    }

    private static func stringArgument(
        named name: String,
        in arguments: LabeledExprListSyntax
    ) throws -> String? {
        guard let argument = arguments.first(where: { $0.label?.text == name }) else {
            return nil
        }
        if argument.expression.is(NilLiteralExprSyntax.self) {
            return nil
        }
        guard let literal = argument.expression.as(StringLiteralExprSyntax.self),
              literal.segments.count == 1,
              case let .stringSegment(segment)? = literal.segments.first
        else {
            throw UzuToolMacroError("@UzuToolFunction \(name) must be a string literal")
        }
        return segment.content.text
    }
}

private struct Documentation {
    let summary: String
    let parameters: [String: String]

    init(function: FunctionDeclSyntax) {
        let lines = function.leadingTrivia.description
            .split(separator: "\n", omittingEmptySubsequences: false)
            .compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("///") else {
                    return nil
                }
                return String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            }

        var summaryLines: [String] = []
        var parameterDescriptions: [String: String] = [:]
        var readingParameters = false
        for line in lines {
            if line == "- Parameters:" {
                readingParameters = true
                continue
            }
            if line.hasPrefix("- Parameter "), let separator = line.firstIndex(of: ":") {
                let nameStart = line.index(line.startIndex, offsetBy: "- Parameter ".count)
                let name = String(line[nameStart..<separator]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
                parameterDescriptions[name] = value
                readingParameters = false
                continue
            }
            if readingParameters, line.hasPrefix("- "), let separator = line.firstIndex(of: ":") {
                let nameStart = line.index(line.startIndex, offsetBy: 2)
                let name = String(line[nameStart..<separator]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
                parameterDescriptions[name] = value
                continue
            }
            if !readingParameters {
                summaryLines.append(line)
            }
        }
        summary = summaryLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        parameters = parameterDescriptions
    }
}

private struct UzuToolMacroError: Swift.Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

private func validate(
    function: FunctionDeclSyntax,
    context: some MacroExpansionContext
) throws {
    if function.genericParameterClause != nil || function.genericWhereClause != nil {
        throw UzuToolMacroError("@UzuToolFunction does not support generic functions")
    }
    if function.modifiers.contains(where: { $0.name.tokenKind == .keyword(.mutating) }) {
        throw UzuToolMacroError("@UzuToolFunction does not support mutating functions")
    }
    if context.lexicalContext.first?.is(ProtocolDeclSyntax.self) == true {
        throw UzuToolMacroError("@UzuToolFunction cannot be used on a protocol requirement")
    }
    if context.lexicalContext.first?.is(FunctionDeclSyntax.self) == true {
        throw UzuToolMacroError("@UzuToolFunction cannot be used on a nested function")
    }
    for parameter in function.signature.parameterClause.parameters {
        if parameter.ellipsis != nil {
            throw UzuToolMacroError("@UzuToolFunction does not support variadic parameters")
        }
        if parameter.defaultValue != nil {
            throw UzuToolMacroError("@UzuToolFunction does not support default parameter values")
        }
        if parameter.type.is(AttributedTypeSyntax.self), parameter.type.trimmedDescription.contains("inout") {
            throw UzuToolMacroError("@UzuToolFunction does not support inout parameters")
        }
    }
}

private func isTypeContext(_ syntax: Syntax) -> Bool {
    syntax.is(ActorDeclSyntax.self)
        || syntax.is(ClassDeclSyntax.self)
        || syntax.is(EnumDeclSyntax.self)
        || syntax.is(ExtensionDeclSyntax.self)
        || syntax.is(StructDeclSyntax.self)
}

private func toolParameterName(_ parameter: FunctionParameterSyntax) -> String {
    let token = parameter.secondName ?? parameter.firstName
    return token.trimmedDescription
}

private func unescaped(_ identifier: String) -> String {
    identifier.trimmingCharacters(in: CharacterSet(charactersIn: "`"))
}

private func accessPrefix(_ modifiers: DeclModifierListSyntax) -> String {
    for modifier in modifiers {
        switch modifier.name.tokenKind {
        case .keyword(.public), .keyword(.package), .keyword(.fileprivate), .keyword(.private):
            return "\(modifier.name.text) "
        case .keyword(.open):
            return "public "
        default:
            continue
        }
    }
    return ""
}

private func swiftStringLiteral(_ value: String) -> String {
    String(reflecting: value)
}
