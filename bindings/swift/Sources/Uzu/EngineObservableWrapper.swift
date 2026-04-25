import Observation
import Foundation
import uzuFFI

public final class EngineObservableWrapper: @unchecked Sendable, ObservableObject {
    @Published public private(set) var engine: Engine?
    @Published public private(set) var models: [Model] = []
    @Published public private(set) var downloadStates: [String: DownloadState] = [:]
    
    public init(config: EngineConfig) {
        Task { [weak self] in
            do {
                let engine = try await Engine.create(config: config)
                try await engine.registerCallback(callback: .create { [weak self] in
                    self?.updateState()
                })
                await MainActor.run { [weak self] in
                    self?.engine = engine
                    self?.updateState()
                }
            } catch {}
        }
    }
    
    func updateState() {
        Task.detached { [weak self] in
            let models = try await self?.engine?.models() ?? []
            let downloadStates = await self?.engine?.downloadStates() ?? [:]
            DispatchQueue.main.async { [weak self] in
                self?.models = models
                self?.downloadStates = downloadStates
            }
        }
    }
}
