import SwiftUI

struct ContentView: View {
    @AppStorage("model") private var model = "mlx-community/Qwen3.5-0.8B-4bit"
    @AppStorage("request") private var request =
        #"{"prompt_text": "Tell me about London", "max_tokens": 256, "num_runs": 3}"#
    @State private var output = ""
    @State private var running = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    TextField("Hugging Face repository or local path", text: $model)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section("Requests") {
                    TextEditor(text: $request)
                        .font(.footnote.monospaced())
                        .frame(minHeight: 120)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section {
                    Button(running ? "Running…" : "Run", action: run)
                        .disabled(running)
                }
                Section("Output") {
                    Text(output)
                        .font(.footnote.monospaced())
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("MLX")
        }
        .task {
            let arguments = ProcessInfo.processInfo.arguments
            if let index = arguments.firstIndex(of: "-request"),
                arguments.indices.contains(index + 1)
            {
                request = arguments[index + 1]
            }
            if UserDefaults.standard.bool(forKey: "run") {
                run()
            }
        }
    }

    private func run() {
        running = true
        let (model, request) = (model, request)
        let thread = Thread {
            let text = PythonRuntime.run(engine: "mlx", model: model, requests: request)
            DispatchQueue.main.async {
                output = text
                running = false
                if UserDefaults.standard.bool(forKey: "run") {
                    exit(0)
                }
            }
        }
        thread.stackSize = 1 << 24
        thread.start()
    }
}
