import Uzu

public func runClassification() async throws {
    let engine = try await Engine.create(config: .create())
    
    guard let model = try await engine.model(identifier: "trymirai/chat-moderation-router") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let messages = [
        ClassificationMessage.user(content: "Hi")
    ]
    
    let session = try await engine.classification(model: model)
    let output = try await session.classify(input: messages)
    print("Output: \(output.probabilities.values)")
}
