import Foundation
import Observation
import Uzu

@Observable
final class SummarizationModel {

    // MARK: - Nested Types
    enum ViewState {
        case loading
        case idle
        case generating
        case error(Swift.Error)
    }

    struct GenerationStats: Equatable {
        let timeToFirstToken: Double
        let tokensPerSecond: Double
        let totalTime: Double

        init(stats: ChatReplyStats) {
            self.timeToFirstToken = stats.timeToFirstToken ?? 0.0
            self.tokensPerSecond = stats.generateTokensPerSecond ?? 0.0
            self.totalTime = stats.duration
        }
    }

    // MARK: - Observable Properties
    var viewState: ViewState = .idle
    var inputText: String = ""
    var summaryText: String = ""
    var stats: GenerationStats? = nil

    // MARK: - Private
    private let identifier: String
    private var session: ChatSession?
    private var loadingTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?

    init(identifier: String) {
        self.identifier = identifier
    }

    // MARK: - Public API

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
            let session: ChatSession?
            let newState: ViewState
            do {
                let config = ChatConfig.create().withSpeculationPreset(speculationPreset: .summarization)
                session = try await engine.chat(model: model, config: config)
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

    func summarise() {
        guard case .idle = viewState,
            let session,
            !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        summaryText = ""
        stats = nil
        viewState = .generating
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = "Text is: \"\(text)\". Write only summary itself."
        
        generationTask?.cancel()
        generationTask = Task.detached { [weak self] in
            guard let self else { return }
            do {
                try await session.reset()
                let stream = await session.replyWithStream(
                    input: [
                        .system().withReasoningEffort(reasoningEffort: .disabled),
                        .user().withText(text: prompt)
                    ],
                    config: .create().withSamplingMethod(samplingMethod: .greedy).withTokenLimit(tokenLimit: 1024)
                )
                for try await update in stream.iterator() {
                    if Task.isCancelled {
                        stream.cancelToken().cancel()
                        break
                    }
                    
                    switch update {
                    case .replies(let replies):
                        if let reply = replies.last {
                            Task { @MainActor [weak self] in
                                guard let self else {
                                    return
                                }
                                self.summaryText = reply.message.text() ?? ""
                                self.stats = .init(stats: reply.stats)
                            }
                        }
                    case .error(let error):
                        stream.cancelToken().cancel()
                        throw error
                    }
                }
                
                Task { @MainActor [weak self] in
                    guard let self  else {
                        return
                    }
                    self.viewState = .idle
                    self.generationTask?.cancel()
                    self.generationTask = nil
                }
            } catch {
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.viewState = .error(error)
                    self.generationTask?.cancel()
                    self.generationTask = nil
                }
            }
        }
    }

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        viewState = .idle
    }

    func tearDown() {
        loadingTask?.cancel()
        loadingTask = nil
        generationTask?.cancel()
        generationTask = nil
        session = nil
        viewState = .loading
    }
}
