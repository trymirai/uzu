import Observation
import SwiftUI
import Uzu
import Firebase
#if os(iOS)
import UIKit
#endif

@main
struct PlaygroundApp: App {
    #if os(iOS)
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #elseif os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif
    
    @StateObject private var engineWrapper: EngineObservableWrapper
    @State private var router: Router
    @State private var audioController: AudioController

    init() {
        _ = _firebaseInitialized
        
        let engineWrapper = EngineObservableWrapper(config: .create())
        _engineWrapper = StateObject(wrappedValue: engineWrapper)
        
        self.router = Router()
        self.audioController = AudioController()
    }

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $router.navPath) {
                HomeView()
                    .navigationDestination(for: Router.Destination.self) { destination in
                        switch destination {
                        case let .modelSelection(next):
                            ModelListView(mode: .choose(next: next))
                        case let .chat(identifier):
                            ChatView(identifier: identifier)
                        case let .classification(identifier):
                            ClassificationView(identifier: identifier)
                        case let .summarization(identifier):
                            SummarizationView(identifier: identifier)
                        case let .textToSpeech(identifier: identifier):
                            TextToSpeechView(identifier: identifier)
                        case .about:
                            AboutView()
                        case .modelManagement:
                            ModelListView(mode: .manage)
                        }
                    }
            }
            .environmentObject(engineWrapper)
            .environment(router)
            .environment(audioController)
            .tint(MiraiAsset.primary.swiftUIColor)
        }
    }
}

private let _firebaseInitialized: Void = {
    FirebaseApp.configure()
}()
