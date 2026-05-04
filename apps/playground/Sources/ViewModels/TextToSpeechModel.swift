import Foundation
import Observation
import Uzu

// MARK: - Supporting Types

extension TextToSpeechModel {
    enum ViewState {
        case loading
        case idle
        case generating
        case error(Swift.Error)
    }
}

struct GenerationError: Swift.Error {
    let message: String
}

@Observable
final class TextToSpeechModel {

    // MARK: - Publicly observed properties
    var viewState: ViewState = .idle
    var inputText: String = ""
    var outputPath: URL?

    // MARK: - Private
    private let identifier: String
    private var session: TextToSpeechSession?
    private var loadingTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?

    init(identifier: String) {
        self.identifier = identifier
    }

    // MARK: - Lifecycle

    @MainActor
    func loadSession(using engineWrapper: EngineObservableWrapper) {
        guard
            case .idle = viewState,
            let engine = engineWrapper.engine,
            let model = engineWrapper.models.first(where: { $0.identifier == identifier }) else {
            return
        }

        if self.session != nil {
            tearDown()
        }

        self.viewState = .loading

        loadingTask?.cancel()
        loadingTask = Task.detached { [weak self] in
            guard let self else { return }
            let session: TextToSpeechSession?
            let newState: ViewState
            do {
                session = try await engine.textToSpeech(model: model)
                newState = .idle
            } catch {
                session = nil
                newState = .error(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.session = session
                self.viewState = newState
                self.loadingTask = nil
            }
        }
    }

    // MARK: - User intents
    func generate() {
        guard case .idle = viewState,
            let session,
            !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        
        let input = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        viewState = .generating
        outputPath = nil

        generationTask?.cancel()
        generationTask = Task.detached { [weak self] in
            guard let self else { return }

            let resultPath = URL.documentsDirectory.appendingPathComponent("tts_output.wav")
            var generationError: Swift.Error? = nil
            do {
                if FileManager.default.fileExists(atPath: resultPath.path) {
                    try FileManager.default.removeItem(at: resultPath)
                }
                let output = try await session.synthesize(input: input)
                try output.pcmBatch.saveAsWav(path: resultPath.path)
                if !FileManager.default.fileExists(atPath: resultPath.path) {
                    generationError = GenerationError(message: "Generation failed")
                }
            } catch {
                generationError = error
            }
            
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error = generationError {
                    self.viewState = .error(error)
                } else {
                    self.viewState = .idle
                    self.outputPath = resultPath
                }
                self.generationTask = nil
            }
        }
    }
    
    func tearDown() {
        loadingTask?.cancel()
        loadingTask = nil
        generationTask?.cancel()
        generationTask = nil
        session = nil
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        viewState = .idle
        outputPath = nil
    }
}
