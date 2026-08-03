import Foundation
import Uzu

public func runClassification() async throws {
    let engine = try await Engine.create(config: .create())
    
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()
    
    let messages = [
        ClassificationMessage.user(content: "Hi")
    ]
    
    let session = try await engine.classification(model: model)
    let output = try await session.classify(input: messages)
    print("Output: \(output.probabilities.values)")
}
