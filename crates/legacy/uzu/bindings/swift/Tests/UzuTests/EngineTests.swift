import XCTest
@testable import Uzu

final class EngineTests: XCTestCase {
    override func setUp() {
        super.setUp()
        executionTimeAllowance = 600
    }

    func testChatReplyProducesText() async throws {
        let engine = try await Engine.create(config: .create())

        let maybeModel = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4")
        let model = try XCTUnwrap(maybeModel, "Model not found")

        for try await update in try await engine.download(model: model).iterator() {
            print("Download progress: \(update.progress())")
        }

        let session = try await engine.chat(model: model, config: .create())

        let messages = [
            ChatMessage.system()
                .withText(text: "You are a helpful assistant")
                .withReasoningEffort(reasoningEffort: .disabled),
            ChatMessage.user().withText(text: "Hi"),
        ]

        let reply = try await session.reply(
            input: messages,
            config: .create()
                .withTokenLimit(tokenLimit: 64)
                .withSamplingMethod(samplingMethod: .greedy)
        )
        let message = try XCTUnwrap(reply.last?.message, "Reply has no messages")

        XCTAssertNotNil(message.text())
    }
}
