import Observation
import Foundation
import uzuFFI

public final class EngineObservableWrapper: @unchecked Sendable, ObservableObject {
    public let engine: Engine
    
    public init(config: EngineConfig) async throws {
        let engine = try await Engine.create(config: config)
        self.engine = engine
        try await engine.registerCallback(callback: .create { [weak self] in
            DispatchQueue.main.async {
                self?.objectWillChange.send()
            }
        })
    }
}
