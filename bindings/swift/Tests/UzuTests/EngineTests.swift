import XCTest
@testable import Uzu

final class EngineTests: XCTestCase {
    override func setUp() {
        super.setUp()
        executionTimeAllowance = 600
    }

    func testChatReplyProducesText() async throws {
        let engine = try await Engine.create(config: .create())

        let maybeModel = try await engine.model(identifier: "Qwen/Qwen3-0.6B")
        let model = try XCTUnwrap(maybeModel, "Model not found")

        for try await update in try await engine.download(model: model).iterator() {
            print("Download progress: \(update.progress())")
        }

        let session = try await engine.chat(model: model, config: .create())

        let messages = [
            ChatMessage.system().withText(text: "You are a helpful assistant"),
            ChatMessage.user().withText(text: "Hi"),
        ]

        let reply = try await session.reply(input: messages, config: .create())
        let message = try XCTUnwrap(reply.last?.message, "Reply has no messages")

        XCTAssertNotNil(message.reasoning())
        XCTAssertNotNil(message.text())
    }
}
