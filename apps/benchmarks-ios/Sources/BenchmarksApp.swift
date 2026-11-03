import SwiftUI

@main
struct BenchmarksApp: App {
    @State private var benchmarks = Benchmarks()
    @State private var server: BenchmarkServer?

    init() {
        signal(SIGPIPE, SIG_IGN)
        setenv("HOME", URL.cachesDirectory.path, 1)
        setenv("SSL_CERT_FILE", "\(Bundle.main.resourcePath!)/envs/mlx/certifi/cacert.pem", 1)
        FileManager.default.changeCurrentDirectoryPath(URL.documentsDirectory.path)
    }

    var body: some Scene {
        WindowGroup {
            ContentView(benchmarks: benchmarks)
                .task {
                    UIApplication.shared.isIdleTimerDisabled = true
                    do {
                        server = try BenchmarkServer(benchmarks: benchmarks)
                    } catch {
                        benchmarks.status = "Server failed: \(error)"
                    }
                }
        }
    }
}
