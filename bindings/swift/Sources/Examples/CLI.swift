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
            "Mode: chat | chat-for-summarization | chat-for-classification | quick-start | snippets | cloud | ssm | structured-output | classifier | text-to-speech",
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
        case "chat-speculation-summarization":
            try await runChatSpeculationSummarization()
        case "chat-speculation-classification":
            try await runChatSpeculationClassification()
        case "classification":
            try await runClassification()
        case "text-to-speech":
            try await runTextToSpeech()
        default:
            throw ValidationError("Unknown mode: \(mode)")
        }
    }
}
