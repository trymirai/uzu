import Foundation
import Network

final class BenchmarkServer {
    static let port: NWEndpoint.Port = 7171

    private struct Command: Decodable {
        let command: String
        let engine: Engine?
        let arguments: [String]?
        let path: String?
    }

    private enum Failure: Error {
        case invalid(String)
        case busy
    }

    private let benchmarks: Benchmarks
    private let listener: NWListener

    init(benchmarks: Benchmarks) throws {
        self.benchmarks = benchmarks
        listener = try NWListener(using: .tcp, on: Self.port)
        listener.stateUpdateHandler = { state in
            DispatchQueue.main.async {
                switch state {
                case .ready:
                    benchmarks.status = "Listening on port \(Self.port)"
                case .failed(let error):
                    benchmarks.status = "Server failed: \(error)"
                default:
                    break
                }
            }
        }
        listener.newConnectionHandler = { [unowned self] connection in
            connection.start(queue: DispatchQueue(label: "BenchmarkServer.connection"))
            readCommand(connection, buffer: Data())
        }
        listener.start(queue: DispatchQueue(label: "BenchmarkServer"))
    }

    private func readCommand(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) {
            data, _, isComplete, error in
            let buffer = buffer + (data ?? Data())
            if let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = buffer[..<newline]
                let rest = Data(buffer[buffer.index(after: newline)...])
                DispatchQueue.main.async { self.handle(line, rest: rest, connection) }
            } else if isComplete || error != nil {
                connection.cancel()
            } else {
                self.readCommand(connection, buffer: buffer)
            }
        }
    }

    private func handle(_ line: Data, rest: Data, _ connection: NWConnection) {
        let send = { (kind: UInt8, payload: Data) in
            var frame = Data([kind])
            withUnsafeBytes(of: UInt32(payload.count).bigEndian) { frame.append(contentsOf: $0) }
            connection.send(content: frame + payload, completion: .idempotent)
        }
        let finish = { (status: Int32) in
            withUnsafeBytes(of: status.bigEndian) { send(3, Data($0)) }
        }
        do {
            let command = try JSONDecoder().decode(Command.self, from: line)
            switch (command.command, command.engine, command.arguments, command.path) {
            case ("models", _, _, _):
                benchmarks.refresh()
                send(1, try JSONEncoder().encode(benchmarks.models) + Data("\n".utf8))
            case ("device", _, _, _):
                let device = ["physical_memory_bytes": ProcessInfo.processInfo.physicalMemory]
                send(1, try JSONEncoder().encode(device) + Data("\n".utf8))
            case ("delete", _, _, let path?):
                try benchmarks.delete(path: path)
            case ("run", let engine?, let arguments?, _):
                let job = { engine.run(arguments: arguments) }
                let input = { (handle: FileHandle) in
                    DispatchQueue.global().async { self.forward(connection, rest, to: handle) }
                }
                guard benchmarks.start(job, input: input, output: send, completion: finish) else {
                    throw Failure.busy
                }
                return
            default:
                throw Failure.invalid(String(decoding: line, as: UTF8.self))
            }
            finish(0)
            drain(connection)
        } catch {
            send(2, Data("\(error)\n".utf8))
            finish(1)
            drain(connection)
        }
    }

    private func drain(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) {
            _, _, isComplete, error in
            if isComplete || error != nil {
                connection.cancel()
            } else {
                self.drain(connection)
            }
        }
    }

    private func forward(_ connection: NWConnection, _ buffer: Data, to handle: FileHandle) {
        var buffer = buffer
        while buffer.count >= 4 {
            let length = Int(buffer.prefix(4).reduce(UInt32(0)) { $0 << 8 | UInt32($1) })
            guard buffer.count >= 4 + length else {
                break
            }
            guard length > 0,
                (try? handle.write(contentsOf: buffer.dropFirst(4).prefix(length))) != nil
            else {
                try? handle.close()
                drain(connection)
                return
            }
            buffer = Data(buffer.dropFirst(4 + length))
        }
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) {
            data, _, isComplete, error in
            if let data, !data.isEmpty {
                self.forward(connection, buffer + data, to: handle)
            } else if isComplete || error != nil {
                try? handle.close()
                connection.cancel()
            } else {
                self.forward(connection, buffer, to: handle)
            }
        }
    }
}
