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

    @Argument(help:"Mode: chat | chat-cloud | chat-structured-output | quick-start | tool-calls",
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
        case "tool-calls":
            try await runToolCalls()
        default:
            throw ValidationError("Unknown mode: \(mode)")
        }
    }
}
