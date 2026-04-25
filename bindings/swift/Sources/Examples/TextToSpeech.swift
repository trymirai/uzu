import Foundation
import Uzu

public func runTextToSpeech() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "fishaudio/s1-mini") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let text = "London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year."
    let outputPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Desktop")
        .appendingPathComponent("output.wav")
    let session = try await engine.textToSpeech(model: model)
    let pcmBatch = try await session.synthesize(input: text)
    try pcmBatch.saveAsWav(path: outputPath.path())
    print("Output saved to: \(outputPath.path())")
}
