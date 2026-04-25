import Foundation
import Observation
import Uzu

// MARK: - Supporting Types

extension ChatModel {
    enum ViewState {
        case loading
        case idle
        case generating
        case error(Swift.Error)
    }
}

@Observable
final class ChatModel {

    // MARK: - Publicly observed properties
    var viewState: ViewState = .idle
    var messages: [Message] = []
    var inputText: String = ""

    // MARK: - Private
    private let identifier: String
    private var session: ChatSession?
    private var loadingTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?

    init(identifier: String) {
        self.identifier = identifier
    }

    @MainActor
    func isCloud(engineWrapper: EngineObservableWrapper) -> Bool {
        guard let model = engineWrapper.models.first(where: { $0.identifier == identifier }) else {
            return false
        }
        return model.isRemote()
    }

    // MARK: - Lifecycle

    @MainActor
    func loadSession(using engineWrapper: EngineObservableWrapper, turbo: Bool) {
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
                let config = turbo ? ChatConfig.create().withSpeculationPreset(speculationPreset: .generalChat) : ChatConfig.create()
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

    // MARK: - User intents
    func sendMessage() {
        guard case .idle = viewState,
            let session,
            !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }

        let userInput = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        inputText = ""

        messages.append(Message(role: .user, content: userInput))
        let assistantMessage = Message(role: .assistant, content: "")
        messages.append(assistantMessage)
        viewState = .generating

        generationTask?.cancel()
        generationTask = Task.detached { [weak self, assistantId = assistantMessage.id] in
            guard let self else { return }
            let inputMessages: [ChatMessage] = self.messages.dropLast().map { msg in
                let role: ChatRole = (msg.role == .user) ? .user : .assistant
                return ChatMessage.forRole(role: role).withText(text: msg.content)
            }

            do {
                try await session.reset()
                let stream = await session.replyWithStream(input: inputMessages, config: .create().withTokenLimit(tokenLimit: 1024))
                for try await update in stream.iterator() {
                    if Task.isCancelled {
                        stream.cancelToken().cancel()
                        break
                    }
                    
                    switch update {
                    case .replies(let replies):
                        if let reply = replies.last {
                            Task { @MainActor [weak self] in
                                guard let self, let idx = self.messages.firstIndex(where: { $0.id == assistantId }) else {
                                    return
                                }
                                
                                self.messages[idx].reasoning = reply.message.reasoning()?.trimmingCharacters(in: .whitespacesAndNewlines)
                                self.messages[idx].content = (reply.message.text() ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                                self.messages[idx].stats = MessageStats(stats: reply.stats)
                            }
                        }
                    case .error(let error):
                        stream.cancelToken().cancel()
                        throw error
                    }
                }
                
                Task { @MainActor [weak self] in
                    guard let self else {
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

    func stopGeneration() {
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
    }
}
