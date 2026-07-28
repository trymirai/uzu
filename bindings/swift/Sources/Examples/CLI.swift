import ArgumentParser
import Foundation

@available(macOS 26.0, iOS 26.0, *)
@main
struct Example: AsyncParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "examples",
        abstract: "uzu examples",
        version: "1.0.0"
    )

    @Argument(
        help:
            "Mode: chat | chat-for-summarization | chat-for-classification | quick-start | snippets | cloud | ssm | structured-output | classifier",
        transform: { $0.lowercased() })
    var mode: String = "chat"

    mutating func run() async throws {
        switch mode {
        case "quick-start":
            try await runQuickStart()
        case "chat":
            try await runChat()
        case "chat-cloud":
            try await runChatCloud()
        case "chat-structured-output":
            try await runChatStructuredOutput()
        case "classification":
            try await runClassification()
        default:
            throw ValidationError("Unknown mode: \(mode)")
        }
    }
}
