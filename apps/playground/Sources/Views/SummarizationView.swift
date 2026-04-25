import Foundation
import Observation
import SFSymbols
import SwiftUI
import Uzu

#if canImport(UIKit)
    import UIKit
#endif

struct SummarizationView: View {

    // MARK: - Environment
    @Environment(Router.self) private var router
    @EnvironmentObject private var engineWrapper: EngineObservableWrapper
    @Environment(AudioController.self) private var audioController
    @State private var viewModel: SummarizationModel

    // MARK: - Stored Properties
    let identifier: String

    static let textsToSummarize =
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective."

    // MARK: - State

    @FocusState private var inputFocused: Bool

    // MARK: - Type Definitions
    // ViewState provided by SummarizationModel.ViewState

    // MARK: - Body
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 20) {
                        input
                        if !viewModel.summaryText.isEmpty {
                            summary
                        }

                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding()
                }
                .onChange(of: viewModel.summaryText) { _, _ in
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            Divider()
            bottomBar
                .padding()
        }
        .background(MiraiAsset.background.swiftUIColor)
        #if os(iOS)
            .toolbar { MiraiToolbarModelName(identifier: identifier) }
        #endif
        .toolbarRole(.editor)
        .onAppear {
            viewModel.loadSession(using: engineWrapper)
        }
        .onDisappear {
            viewModel.tearDown()
        }
    }

    // MARK: - UI

    private var input: some View {
        ZStack(alignment: .topLeading) {
            TextField("Enter text to summarise...", text: $viewModel.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .focused($inputFocused)
                .font(.monoBody16)
                .disabled(isInputDisabled)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .background(MiraiAsset.card.swiftUIColor)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            if viewModel.inputText.isEmpty {
                Text("Enter text to summarise...")
                    .font(.monoBody16)
                    .foregroundStyle(MiraiAsset.secondary.swiftUIColor)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .allowsHitTesting(false)
            }
        }
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(LocalizedStringKey(viewModel.summaryText))
                .font(.monoBody16)
                .textSelection(.enabled)

            if let stats = viewModel.stats {
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
        .padding(.leading, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var bottomBar: some View {
        HStack {
            Spacer()
            Button(action: {
                if case .generating = viewModel.viewState {
                    viewModel.stop()
                } else {
                    viewModel.summarise()
                }
            }) {
                Text(actionButtonTitle)
                    .font(.monoHeading14Bold)
                    .foregroundStyle(buttonTextColor)
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(buttonBackgroundColor)
                    .cornerRadius(12)
            }
            .disabled(isActionDisabled)
            Spacer()
        }
    }

    // MARK: - Logic – Helpers

    private var isActionDisabled: Bool {
        if case .generating = viewModel.viewState {
            return false
        }
        return !isInputValid || isInputDisabled
    }

    private var isInputDisabled: Bool {
        switch viewModel.viewState {
        case .loading, .generating, .error:
            return true
        case .idle:
            return false
        }
    }

    private var isInputValid: Bool {
        !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var actionButtonTitle: String {
        switch viewModel.viewState {
        case .loading:
            return "Loading..."
        case .generating:
            return "STOP"
        case .error(let error):
            return "Error: \(error.localizedDescription)"
        case .idle:
            return "SUMMARISE"
        }
    }

    private var buttonTextColor: Color {
        if case .generating = viewModel.viewState {
            return MiraiAsset.contrast.swiftUIColor
        }
        return isInputDisabled || !isInputValid
            ? MiraiAsset.secondary.swiftUIColor
            : MiraiAsset.contrast.swiftUIColor
    }

    private var buttonBackgroundColor: Color {
        if case .generating = viewModel.viewState {
            return MiraiAsset.primary.swiftUIColor
        }
        return isInputDisabled || !isInputValid
            ? MiraiAsset.card.swiftUIColor : MiraiAsset.primary.swiftUIColor
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

    init(identifier: String) {
        self.identifier = identifier
        _viewModel = State(initialValue: SummarizationModel(identifier: identifier))
        viewModel.inputText = SummarizationView.textsToSummarize
    }
}
