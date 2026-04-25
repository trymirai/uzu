import Foundation
import Observation
import SFSymbols
import SwiftUI
import Uzu

enum MessageRole {
    case user
    case assistant
}

struct MessageStats: Equatable {
    let timeToFirstToken: Double
    let tokensPerSecond: Double
    let totalTime: Double

    init(
        timeToFirstToken: Double,
        tokensPerSecond: Double,
        totalTime: Double
    ) {
        self.timeToFirstToken = timeToFirstToken
        self.tokensPerSecond = tokensPerSecond
        self.totalTime = totalTime
    }

    init(stats: Uzu.ChatReplyStats) {
        self.timeToFirstToken = stats.timeToFirstToken ?? 0.0
        self.tokensPerSecond = stats.generateTokensPerSecond ?? 0.0
        self.totalTime = stats.duration
    }
}

struct Message: Identifiable, Equatable {
    let id = UUID()
    let role: MessageRole
    var reasoning: String?
    var content: String
    var stats: MessageStats? = nil

    static func == (lhs: Message, rhs: Message) -> Bool {
        lhs.id == rhs.id && lhs.content == rhs.content && lhs.role == rhs.role
            && lhs.stats == rhs.stats
    }
}

struct ChatView: View {

    // MARK: - Environment
    @Environment(Router.self) var router
    @EnvironmentObject private var engineWrapper: EngineObservableWrapper
    @Environment(AudioController.self) private var audioController

    // MARK: - State
    @State private var viewModel: ChatModel
    @FocusState private var inputFocused: Bool
    @State private var turbo = false

    // MARK: - Stored Properties
    let identifier: String

    init(identifier: String) {
        self.identifier = identifier
        _viewModel = State(initialValue: ChatModel(identifier: identifier))
    }

    fileprivate init(messages: [Message], identifier: String, viewState: ChatModel.ViewState) {
        self.identifier = identifier
        _viewModel = State(initialValue: ChatModel(identifier: identifier))
        viewModel.messages = messages
        viewModel.viewState = viewState
    }

    private var isInputDisabled: Bool {
        switch viewModel.viewState {
        case .loading, .generating, .error:
            return true
        case .idle:
            return false
        }
    }

    private var inputPlaceholder: String {
        switch viewModel.viewState {
        case .loading:
            return "Loading..."
        case .generating:
            return "Generating..."
        case .error(let error):
            return "Error: \(error.localizedDescription)"
        case .idle:
            return "Message..."
        }
    }

    private var isInputValid: Bool {
        !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var turboEnabled: Bool {
        switch(viewModel.viewState) {
        case .idle:
            return true
        case .loading, .generating, .error(_):
            return false
        }
    }

    @ViewBuilder
    private var turboView: some View {
        HStack(alignment: .center) {
            Spacer()
            Button(action: {
                turbo.toggle()
            }) {
                HStack(alignment: .center, spacing: 8.0) {
                    Text("Turbo")
                        .font(.monoHeading14)
                        .foregroundColor(MiraiAsset.primary.swiftUIColor)
                    Image(symbol: turbo ? .checkmarkCircleFill : .circle)
                        .font(.title24Light)
                        .foregroundColor(MiraiAsset.primary.swiftUIColor)
                }
                .opacity(turboEnabled ? 1.0 : 0.5)
            }
        }
        .padding(EdgeInsets(top: 0.0, leading: 16.0, bottom: 0.0, trailing: 16.0))
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 20) {
                        ForEach(viewModel.messages) { messageRow(message: $0) }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages) { _, _ in
                    if let last = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
            Divider()
            if !viewModel.isCloud(engineWrapper: engineWrapper) {
                Spacer(minLength: 16.0)
                turboView
                    .disabled(!turboEnabled)
            }
            inputView
                .padding()
        }
        #if os(iOS)
            .toolbar { MiraiToolbarModelName(identifier: identifier) }
        #endif
        .toolbarRole(.editor)
        .onAppear {
            viewModel.loadSession(using: engineWrapper, turbo: turbo)
        }
        .onChange(of: turbo) { _, _ in
            viewModel.loadSession(using: engineWrapper, turbo: turbo)
        }
        .onDisappear {
            viewModel.tearDown()
        }
    }

    @ViewBuilder
    private var sendMessageButton: some View {
        if case .generating = viewModel.viewState {
            Button(action: stopMessage) {
                Image(systemName: "square.fill")
                    .font(.body16Semibold)
                    .foregroundStyle(.white)
                    .frame(width: 28, height: 28)
                    .background(Color.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)
        } else {
            Button(action: sendMessage) {
                Image(symbol: .arrowUp)
                    .font(.body16Semibold)
                    .foregroundStyle(
                        isInputValid && !isInputDisabled
                            ? .white : MiraiAsset.secondary.swiftUIColor
                    )
                    .frame(width: 28, height: 28)
                    .background(
                        isInputValid && !isInputDisabled
                            ? .black : MiraiAsset.cardBorder.swiftUIColor
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .disabled(!isInputValid || isInputDisabled)
            .buttonStyle(.plain)
            #if os(macOS)
                .keyboardShortcut(.defaultAction)
            #endif
        }
    }

    @ViewBuilder
    private var inputView: some View {
        HStack(alignment: .center, spacing: 12) {
            ZStack(alignment: .topLeading) {
                SendTextView(text: $viewModel.inputText) {
                    if isInputValid && !isInputDisabled {
                        sendMessage()
                    }
                }
                .focused($inputFocused)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .fixedSize(horizontal: false, vertical: true)
                .disabled(isInputDisabled)

                Text(inputPlaceholder)
                    .font(.monoBody16)
                    .foregroundStyle(MiraiAsset.secondary.swiftUIColor)
                    .opacity(viewModel.inputText.isEmpty ? 1.0 : 0.0)
                    .allowsHitTesting(false)
            }
            sendMessageButton
        }
        .padding(12)
        .background(MiraiAsset.card.swiftUIColor)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    @ViewBuilder
    private func messageRow(message: Message) -> some View {
        if message.role == .user {
            userMessageView(message: message)
        } else {
            assistantMessageView(message: message)
        }
    }

    @ViewBuilder
    private func userMessageView(message: Message) -> some View {
        HStack {
            Spacer(minLength: 64)
            Text(LocalizedStringKey(message.content))
                .font(.monoBody16)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(MiraiAsset.card.swiftUIColor)
                .foregroundStyle(.primary)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func assistantMessageView(message: Message) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                if let reasoning = message.reasoning {
                    Text(LocalizedStringKey(reasoning))
                        .font(.monoCaption12)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(MiraiAsset.card.swiftUIColor)
                        .foregroundStyle(.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .textSelection(.enabled)
                    Spacer()
                        .frame(height: 8.0)
                }
                Text(LocalizedStringKey(message.content))
                    .font(.monoBody16)
                    .textSelection(.enabled)
                if let stats = message.stats {
                    Rectangle()
                        .fill(MiraiAsset.cardBorder.swiftUIColor)
                        .frame(height: 1)
                        .padding(.vertical, 4)
                    VStack(alignment: .leading, spacing: 4) {
                        metricRow(
                            label: "Time to first token:",
                            value: String(format: "%.3f s", stats.timeToFirstToken)
                        )
                        if stats.tokensPerSecond > 0 {
                            metricRow(
                                label: "Tokens per second:",
                                value: String(format: "%.3f t/s", stats.tokensPerSecond)
                            )
                        }
                        metricRow(
                            label: "Total time:",
                            value: String(format: "%.3f s", stats.totalTime)
                        )
                    }
                }
            }
            Spacer(minLength: 64)
        }
    }

    private func stopMessage() {
        viewModel.stopGeneration()
    }

    private func sendMessage() {
        audioController.pause()
        inputFocused = false
        viewModel.sendMessage()
    }

    // MARK: - Metric row helper

    @ViewBuilder
    private func metricRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.monoCaption12)
                .foregroundStyle(MiraiAsset.secondary.swiftUIColor)
            Text(value)
                .font(.monoBody16)
        }
    }

}
