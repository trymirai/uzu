import Foundation

enum Engine: String, Codable {
    case llamacpp
    case mldrift
    case mlx
    case mlxserve
    case mtplx
    case omlx
    case splash
    case uzu

    func run(configuration: String) -> Int32 {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("model-\(UUID().uuidString).json")
        do {
            try configuration.write(to: file, atomically: true, encoding: .utf8)
        } catch {
            fputs("Cannot write model configuration: \(error)\n", stderr)
            return 1
        }
        defer { try? FileManager.default.removeItem(at: file) }
        let arguments = [file.path]
        return switch self {
        case .llamacpp:
            Self.main(benchmarks_llamacpp_main, ["bench-llamacpp"] + arguments)
        case .mldrift:
            Self.mldrift(arguments)
        case .mlx:
            PythonRuntime.main("mlx", "bench_mlx", arguments)
        case .mlxserve:
            Self.mlxserve(arguments)
        case .mtplx:
            PythonRuntime.main("mtplx", "bench_mtplx", arguments)
        case .omlx:
            PythonRuntime.main("omlx", "bench_omlx", arguments)
        case .splash:
            PythonRuntime.main("splash", "bench_splash", arguments)
        case .uzu:
            Self.main(benchmarks_uzu_main, ["bench-uzu"] + arguments)
        }
    }

    private static func mldrift(_ arguments: [String]) -> Int32 {
        guard let file = arguments.first,
            let data = FileManager.default.contents(atPath: file),
            let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let directory = config["model"] as? String
        else {
            fputs("Expected a model.json with a model directory\n", stderr)
            return 2
        }
        guard FileManager.default.fileExists(atPath: "\(directory)/layer_info.txt") else {
            fputs("\(directory) has no ML Drift weights; convert them on a Mac\n", stderr)
            return 2
        }
        return main(
            benchmarks_mldrift_main,
            ["engine-mldrift", "--config=\(file)", "--checkpoint=\(directory)", "--weights=\(directory)"])
    }

    private static func mlxserve(_ arguments: [String]) -> Int32 {
        let paths = PythonRuntime.modelPaths(arguments)
        guard !paths.isEmpty else {
            return 1
        }
        return main(benchmarks_mlxserve_main, ["engine-mlxserve"] + paths)
    }

    private static func main(
        _ entry: (Int32, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32,
        _ arguments: [String]
    ) -> Int32 {
        var argv = arguments.map { strdup($0) } + [nil]
        defer {
            for argument in argv {
                free(argument)
            }
        }
        return entry(Int32(arguments.count), &argv)
    }
}
