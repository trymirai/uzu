import Foundation
import Observation
import SFSymbols
import SwiftUI
import Uzu

#if canImport(UIKit)
    import UIKit
#endif

struct ModelListView: View {
    // MARK: - Type Definitions
    enum Mode {
        case choose(next: Router.ModelListDestination)
        case manage
    }

    // MARK: - Stored Properties
    let mode: Mode

    // MARK: - Environment
    @EnvironmentObject var engineWrapper: EngineObservableWrapper
    @Environment(Router.self) var router

    // MARK: - State
    @State var selectedIdentifier: String?
    @State private var selectedModelSection: ModelSection?

    var body: some View {
        VStack(spacing: 0) {
            header
            modelList
            Spacer()
            bottomBar
                .padding(20)
        }
        .background(MiraiAsset.background.swiftUIColor)
        #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            MiraiToolbarLogo()
        }
        .toolbarBackground(MiraiAsset.background.swiftUIColor, for: .automatic)
        .toolbarBackground(.visible, for: .automatic)
        #if os(macOS)
            .toolbarRole(.editor)
        #endif
    }

    // MARK: - UI

    // Header
    private var header: some View {
        VStack(spacing: 0) {
            VStack(spacing: 8) {
                Text("Available AI Models")
                    .font(.monoBody16Semibold)
                    .textCase(.uppercase)
                    .foregroundColor(MiraiAsset.primary.swiftUIColor)

                Text("select a local model to install")
                    .font(.monoCaption12)
                    .foregroundColor(MiraiAsset.secondary.swiftUIColor)
            }
            .padding(.vertical, 28)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("modelListHeaderContainer")
    }

    // Model List
    private var modelList: some View {
        let group = grouping
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Cloud section (visible only when choosing a Chat model)
                    if case .choose(let next) = mode, next == .chat, !group.cloud.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Cloud")
                                .font(.monoCaption12Semibold)
                                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
                                .textCase(.uppercase)
                                .padding(.horizontal, 4)
                            ForEach(group.cloud, id: \.identifier) { model in
                                CloudModelRow(
                                    model: model, isSelected: selectedIdentifier == model.identifier
                                )
                                .id(model.identifier)
                                .onTapGesture {
                                    selectedIdentifier = model.identifier
                                    selectedModelSection = nil
                                }
                            }
                        }
                    }
                    ForEach(sectionOrder, id: \.self) { section in
                        let identifiers = group.bySection[section] ?? []
                        if !identifiers.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text(section.title)
                                    .font(.monoCaption12Semibold)
                                    .foregroundColor(MiraiAsset.secondary.swiftUIColor)
                                    .textCase(.uppercase)
                                    .padding(.horizontal, 4)
                                ForEach(identifiers, id: \.self) { identifier in
                                    if let model = group.modelByIdentifier[identifier],
                                       let state = engineWrapper.downloadStates[identifier] {
                                        ModelRowView(
                                            identifier: identifier,
                                            model: model,
                                            state: state,
                                            isSelected: selectedIdentifier == identifier
                                        )
                                        .equatable()
                                        .id(identifier)
                                        .onTapGesture {
                                            select(identifier: identifier)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, 20)
            }
            .onChange(of: engineWrapper.models) { _, _ in
                guard
                    let selectedIdentifier,
                    let model = group.modelByIdentifier[selectedIdentifier],
                    model.isLocal(),
                    let state = engineWrapper.downloadStates[model.identifier]
                else { return }

                let newSection = section(for: state)

                if newSection != selectedModelSection {
                    selectedModelSection = newSection
                    withAnimation {
                        proxy.scrollTo(selectedIdentifier, anchor: .center)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var bottomBar: some View {
        switch mode {
        case .choose:
            chooseBottomBar
        case .manage:
            manageBottomBar
        }
    }

    @ViewBuilder
    private var chooseBottomBar: some View {
        if let state = selectedModelState {
            switch state.phase {
            case .notDownloaded, .paused, .locked:
                downloadButton
            case .downloading:
                waitingText
            case .error:
                retryButton
            case .downloaded:
                chooseButton
            }
        } else if
            let selectedId = selectedIdentifier,
            models()
                .filter({ $0.isRemote() })
                .contains(where: { $0.identifier == selectedId })
        {
            // Cloud models are selectable only for Chat
            if case .choose(let next) = mode, next == .chat {
                Button(action: { proceed() }) {
                    Text("CHOOSE CLOUD MODEL")
                        .font(.monoHeading14Bold)
                        .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                        .frame(maxWidth: .infinity)
                        .frame(height: 56)
                        .background(MiraiAsset.primary.swiftUIColor)
                        .cornerRadius(12)
                }
            } else {
                disabledChooseButton
            }
        } else {
            disabledChooseButton
        }
    }
    @ViewBuilder
    private var manageBottomBar: some View {
        if let state = selectedModelState {
            switch state.phase {
            case .notDownloaded, .error, .locked:
                downloadButton
            case .downloading:
                pauseButton
            case .paused:
                resumeButton
            case .downloaded:
                deleteButton
            }
        } else if
            let selectedId = selectedIdentifier,
            models()
                .filter({ $0.isRemote() })
                .contains(where: { $0.identifier == selectedId })
        {
            disabledManageButton
        } else {
            disabledManageButton
        }
    }

    // MARK: - Bottom bar buttons & texts

    private var disabledChooseButton: some View {
        Button(action: {}) {
            Text("CHOOSE")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.card.swiftUIColor)
                .cornerRadius(12)
        }
        .disabled(true)
    }

    private var disabledManageButton: some View {
        Button(action: {}) {
            Text("SELECT MODEL")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.card.swiftUIColor)
                .cornerRadius(12)
        }
        .disabled(true)
    }

    private var downloadButton: some View {
        Button(action: {
            downloadSelectedModel()
        }) {
            Text("DOWNLOAD")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
        .accessibilityIdentifier("downloadButton")
    }

    private var retryButton: some View {
        Button(action: {
            downloadSelectedModel()
        }) {
            Text("RETRY DOWNLOAD")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
    }

    private var chooseButton: some View {
        Button(action: {
            proceed()
        }) {
            Text("CHOOSE")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
    }

    private var waitingText: some View {
        Text("Wait for the models to be installed...")
            .font(.monoHeading14Medium)
            .foregroundColor(MiraiAsset.secondary.swiftUIColor)
            .frame(maxWidth: .infinity)
            .frame(height: 56)
            .multilineTextAlignment(.center)
    }

    private var deleteButton: some View {
        Button(action: {
            deleteSelectedModel()
        }) {
            Text("DELETE")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
        .accessibilityIdentifier("deleteButton")
    }

    private var pauseButton: some View {
        Button(action: {
            pauseSelectedModel()
        }) {
            Text("PAUSE")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
        .accessibilityIdentifier("pauseButton")
    }

    private var resumeButton: some View {
        Button(action: {
            resumeSelectedModel()
        }) {
            Text("RESUME")
                .font(.monoHeading14Bold)
                .foregroundColor(MiraiAsset.contrast.swiftUIColor)
                .frame(maxWidth: .infinity)
                .frame(height: 56)
                .background(MiraiAsset.primary.swiftUIColor)
                .cornerRadius(12)
        }
        .accessibilityIdentifier("resumeButton")
    }

    // MARK: - Helpers
    private var selectedModelState: DownloadState? {
        guard let selectedIdentifier else { return nil }
        guard let model = models()
            .filter({ $0.isLocal() })
            .first(where: { $0.identifier == selectedIdentifier }) else {
            return nil
        }
        let state = engineWrapper.downloadStates[model.identifier]
        return state
    }

    private func select(identifier: String) {
        selectedIdentifier = identifier
        if let model = models()
            .filter({ $0.isLocal() })
            .first(where: { $0.identifier == selectedIdentifier }),
           let state = engineWrapper.downloadStates[model.identifier] {
            selectedModelSection = section(for: state)
        }
    }

    private func downloadSelectedModel() {
        guard
            let selectedIdentifier,
            let model = engineWrapper.models.first(where: {$0.identifier == selectedIdentifier}) else {
            return
        }
        print("[ModelListView] downloadSelectedModel called for: \(selectedIdentifier)")
        if let state = selectedModelState {
            print("[ModelListView] Current state phase: \(state.phase)")
        }
        Task {
            try? await engineWrapper.engine?.downloader(model: model).resume()
        }
    }

    private func deleteSelectedModel() {
        guard
            let selectedIdentifier,
            let model = engineWrapper.models.first(where: {$0.identifier == selectedIdentifier}) else {
            return
        }
        Task {
            try? await engineWrapper.engine?.downloader(model: model).delete()
        }
    }

    private func proceed() {
        guard let selectedIdentifier else { return }
        guard case .choose(let next) = mode else { return }
        switch next {
        case .classification:
            router.navigate(to: .classification(identifier: selectedIdentifier))
        case .summarization:
            router.navigate(to: .summarization(identifier: selectedIdentifier))
        case .chat:
            router.navigate(to: .chat(identifier: selectedIdentifier))
        case .textToSpeech:
            router.navigate(to: .textToSpeech(identifier: selectedIdentifier))
        }
    }

    private func pauseSelectedModel() {
        guard
            let selectedIdentifier,
            let model = engineWrapper.models.first(where: {$0.identifier == selectedIdentifier}) else {
            return
        }
        Task {
            try? await engineWrapper.engine?.downloader(model: model).pause()
        }
    }

    private func resumeSelectedModel() {
        guard
            let selectedIdentifier,
            let model = engineWrapper.models.first(where: {$0.identifier == selectedIdentifier}) else {
            return
        }
        Task {
            try? await engineWrapper.engine?.downloader(model: model).resume()
        }
    }

    // MARK: - Section helpers

    private enum ModelSection: Int {
        case installed
        case installing
        case paused
        case notInstalled

        var title: String {
            switch self {
            case .installed: return "Installed"
            case .installing: return "Installing"
            case .paused: return "Paused"
            case .notInstalled: return "Not Installed"
            }
        }
    }

    private var sectionOrder: [ModelSection] {
        [.installed, .installing, .paused, .notInstalled]
    }
    
    private func models() -> [Uzu.Model] {
        engineWrapper.models.filter({ model in
            switch self.mode {
            case .manage:
                return true
            case let .choose(next):
                switch(next) {
                case .chat, .classification, .summarization:
                    return model.specializations.contains(.chat)
                case .textToSpeech:
                    return model.specializations.contains(.textToSpeech)
                }
            }
        })
    }

    private typealias Grouping = (
        cloud: [Uzu.Model],
        local: [Uzu.Model],
        modelByIdentifier: [String: Uzu.Model],
        bySection: [ModelSection: [String]]
    )

    private var grouping: Grouping {
        let all = engineWrapper.models.filter { model in
            switch self.mode {
            case .manage:
                return true
            case let .choose(next):
                switch next {
                case .chat, .classification, .summarization:
                    return model.specializations.contains(.chat)
                case .textToSpeech:
                    return model.specializations.contains(.textToSpeech)
                }
            }
        }
        let cloud = all.filter { $0.isRemote() }.sorted { $0.name() < $1.name() }
        let local = all.filter { $0.isLocal() }
        let modelByIdentifier = Dictionary(uniqueKeysWithValues: all.map { ($0.identifier, $0) })

        var bySection: [ModelSection: [String]] = [:]
        for model in local {
            guard let state = engineWrapper.downloadStates[model.identifier] else { continue }
            bySection[section(for: state), default: []].append(model.identifier)
        }
        for key in bySection.keys { bySection[key]?.sort() }

        return (cloud, local, modelByIdentifier, bySection)
    }

    private func section(for state: DownloadState) -> ModelSection {
        switch state.phase {
        case .downloaded: return .installed
        case .downloading: return .installing
        case .paused: return .paused
        case .notDownloaded: return .notInstalled
        case .error: return .notInstalled
        case .locked: return .notInstalled
        }
    }

}

// MARK: - Cloud model row

private struct CloudModelRow: View {
    let model: Uzu.Model
    let isSelected: Bool
    @EnvironmentObject private var engineWrapper: EngineObservableWrapper

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "cloud.fill" : "cloud")
                    .foregroundColor(MiraiAsset.primary.swiftUIColor)
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name())
                        .font(.monoHeading14)
                        .foregroundColor(MiraiAsset.primary.swiftUIColor)
                    HStack(spacing: 4) {
                        badge(text: model.family?.vendor.name() ?? model.registry.name())
                    }
                }
                Spacer()
                Text("cloud")
                    .font(.monoCaption12)
                    .foregroundColor(MiraiAsset.secondary.swiftUIColor)
            }
        }
        .padding(16)
        .background(MiraiAsset.card.swiftUIColor)
        .cornerRadius(12)
    }

    @ViewBuilder
    private func badge(text: String) -> some View {
        Text(text)
            .font(.monoBadge10Semibold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(MiraiAsset.cardBorder.swiftUIColor)
            .foregroundColor(MiraiAsset.secondary.swiftUIColor)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}
