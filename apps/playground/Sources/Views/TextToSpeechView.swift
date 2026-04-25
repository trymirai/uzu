import Foundation
import Observation
import SFSymbols
import SwiftUI
import AVFoundation
import Uzu

#if canImport(UIKit)
    import UIKit
#endif

struct TextToSpeechView: View {

    // MARK: - Environment
    @Environment(Router.self) private var router
    @EnvironmentObject private var engineWrapper: EngineObservableWrapper
    @Environment(AudioController.self) private var audioController
    @State private var viewModel: TextToSpeechModel
    @State private var isPlaying: Bool

    // MARK: - Stored Properties
    let identifier: String

    static let textForSynthesis = "London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.";

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
                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding()
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
            TextField("Enter text for synthesis...", text: $viewModel.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .focused($inputFocused)
                .font(.monoBody16)
                .disabled(isInputDisabled)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .background(MiraiAsset.card.swiftUIColor)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
    }

    private var bottomBar: some View {
        HStack(alignment: .center, spacing: 16.0) {
            Spacer()
            Button(action: {
                if case .generating = viewModel.viewState {
                    //viewModel.stopGeneration()
                } else {
                    self.audioController.pause()
                    self.isPlaying = false
                    viewModel.generate()
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
            Button(action: {
                if let url = self.viewModel.outputPath {
                    self.toggleAudio(url: url)
                }
            }) {
                Text(self.isPlaying ? "PAUSE" : "PLAY")
                    .font(.monoHeading14Bold)
                    .foregroundStyle(isPlayEnabled ? MiraiAsset.contrast.swiftUIColor : MiraiAsset.secondary.swiftUIColor)
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(isPlayEnabled ? MiraiAsset.primary.swiftUIColor : MiraiAsset.card.swiftUIColor)
                    .cornerRadius(12)
            }
            .disabled(!isPlayEnabled)
            Spacer()
        }
    }

    // MARK: - Logic – Helpers

    private var isActionDisabled: Bool {
        if case .generating = viewModel.viewState {
            return true
        }
        return !isInputValid || isInputDisabled
    }
    
    private var isPlayEnabled: Bool {
        if case .generating = viewModel.viewState {
            return false
        }
        return self.viewModel.outputPath != nil
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
            return "GENERATE"
        }
    }

    private var buttonTextColor: Color {
        return isActionDisabled
            ? MiraiAsset.secondary.swiftUIColor
            : MiraiAsset.contrast.swiftUIColor
    }

    private var buttonBackgroundColor: Color {
        return isActionDisabled
            ? MiraiAsset.card.swiftUIColor : MiraiAsset.primary.swiftUIColor
    }

    init(identifier: String) {
        self.identifier = identifier
        self.isPlaying = false
        _viewModel = State(initialValue: TextToSpeechModel(identifier: identifier))
        viewModel.inputText = TextToSpeechView.textForSynthesis
    }
    
    func toggleAudio(url: URL) {
        if self.audioController.isPlaying {
            self.audioController.pause()
            self.isPlaying = false
        } else {
            self.audioController.setAsset(AVURLAsset(url: url))
            self.audioController.play()
            self.isPlaying = true
        }
    }
}
