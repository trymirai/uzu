import Foundation
import Uzu

public func runDecision() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)

    guard let model = try await engine.model(identifier: "trymirai/d3-flash-M") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()

    let session = try await engine.decision(model: model)
    let questions: [DecisionQuestion] = [
        .choice(
            instructions: "Which team should handle this request?",
            options: [
                DecisionOption(key: "returns", description: "Refunds, replacements and damaged deliveries"),
                DecisionOption(key: "billing", description: "Payments, invoices and charges"),
                DecisionOption(key: "technical", description: "Product setup and faults"),
            ]
        ),
        .yesNo(instructions: "Does the customer have a receipt?", noDescription: nil, yesDescription: nil),
        .score(instructions: "How urgent is this request?", levels: ["Routine", "Soon", "Today"]),
    ]
    let input = DecisionInput(
        state: "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
        questions: questions
    )
    let output = try await session.decide(input: input)

    for (question, result) in zip(questions, output.results) {
        switch (question, result.answer) {
        case let (.choice(_, options), .choice(_, key, probabilities)):
            print("choice: \(key)")
            for (option, probability) in zip(options, probabilities) {
                print(String(format: "  %@ %.4f", option.key.padding(toLength: 10, withPad: " ", startingAt: 0), probability))
            }
        case let (.score(_, levels), .score(probabilities, score)):
            print(String(format: "score: %.4f", score))
            for (level, probability) in zip(levels, probabilities) {
                print(String(format: "  %@ %.4f", level.padding(toLength: 10, withPad: " ", startingAt: 0), probability))
            }
        case let (_, .yesNo(yesProbability)):
            print(String(format: "yes/no: %.4f", yesProbability))
        default:
            break
        }
        print(String(format: "  %lld tokens in %.3f s", result.stats.tokensCount, result.stats.totalDuration))
    }
}
