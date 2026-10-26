import SwiftUI

@main
struct BenchmarksApp: App {
    init() {
        PythonRuntime.start()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
