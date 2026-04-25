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
            "Mode: chat | summarization | classification | quick-start | snippets | cloud | ssm | structured-output | classifier | text-to-speech",
        transform: { $0.lowercased() })
    var mode: String = "chat"

    mutating func run() async throws {
        switch mode {
        case "chat":
            try await runChat()
        case "summarization":
            try await runSummarization()
        case "classification":
            try await runClassification()
        case "quick-start":
            try await runQuickStart()
        case "snippets":
            try await runSnippets()
        case "cloud":
            try await runCloud()
        case "ssm":
            try await runSSM()
        case "structured-output":
            try await runStructuredOutput()
        case "classifier":
            try await runClassifier()
        case "text-to-speech":
            try await runTextToSpeech()
        default:
            throw ValidationError("Unknown mode: \(mode)")
        }
    }
}
