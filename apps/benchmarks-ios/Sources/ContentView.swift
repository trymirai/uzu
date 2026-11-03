import SwiftUI

struct ContentView: View {
    let benchmarks: Benchmarks

    var body: some View {
        TabView {
            Tab("Log", systemImage: "text.alignleft") {
                NavigationStack {
                    log.navigationTitle("Log")
                }
            }
            Tab("Models", systemImage: "shippingbox") {
                NavigationStack {
                    models.navigationTitle("Models")
                }
            }
        }
        .tabBarMinimizeBehavior(.onScrollDown)
        .tabViewBottomAccessory {
            HStack {
                Text(benchmarks.status)
                Spacer()
                if benchmarks.running {
                    ProgressView()
                }
            }
            .font(.footnote)
            .padding(.horizontal)
        }
    }

    private var log: some View {
        ScrollView {
            Text(benchmarks.log)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .defaultScrollAnchor(.bottom)
        .toolbar {
            ShareLink(item: benchmarks.log)
        }
        .overlay {
            if benchmarks.log.isEmpty {
                ContentUnavailableView(
                    "Waiting for the Mac", systemImage: "laptopcomputer.and.iphone",
                    description: Text("Run benchmarks with bench-ios from scripts/benchmarks."))
            }
        }
    }

    private var models: some View {
        List(benchmarks.models) { model in
            LabeledContent(
                model.name,
                value: ByteCountFormatter.string(fromByteCount: model.bytes, countStyle: .file))
        }
        .overlay {
            if benchmarks.models.isEmpty {
                ContentUnavailableView("No cached models", systemImage: "shippingbox")
            }
        }
        .refreshable { benchmarks.refresh() }
    }
}
