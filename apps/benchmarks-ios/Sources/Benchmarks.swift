import Foundation
import Observation

@Observable
final class Benchmarks {
    var status = "Starting server"
    private(set) var running = false
    private(set) var models = ModelStore.list()
    private var transcript = Data()

    var log: String {
        String(decoding: transcript, as: UTF8.self)
    }

    func start(
        _ job: @escaping () -> Int32,
        input: @escaping (FileHandle) -> Void,
        output: @escaping (UInt8, Data) -> Void,
        completion: @escaping (Int32) -> Void
    ) -> Bool {
        guard !running else {
            return false
        }
        running = true
        transcript = Data()
        let thread = Thread {
            let status = StandardStreams.run(
                input: input,
                output: { stream, data in
                    output(stream, data)
                    DispatchQueue.main.async {
                        self.transcript = (self.transcript + data).suffix(1 << 16)
                    }
                },
                job
            )
            DispatchQueue.main.async {
                self.running = false
                self.refresh()
                completion(status)
            }
        }
        thread.stackSize = 1 << 24
        thread.start()
        return true
    }

    func delete(path: String) throws {
        try ModelStore.delete(path: path)
        refresh()
    }

    func refresh() {
        models = ModelStore.list()
    }
}
