import Foundation

struct CachedModel: Identifiable, Encodable {
    let name: String
    let path: String
    let bytes: Int64

    var id: String { path }
}

enum ModelStore {
    private enum Failure: Error {
        case notCached(String)
    }

    static func list() -> [CachedModel] {
        let caches = URL.cachesDirectory.appending(path: ".cache")
        var models: [CachedModel] = []
        for url in contents(caches.appending(path: "huggingface/hub"))
        where url.lastPathComponent.hasPrefix("models--") {
            let repository = url.lastPathComponent.trimmingPrefix("models--")
            models.append(model(String(repository.replacing("--", with: "/")), url))
        }
        for vendor in contents(caches.appending(path: "mirai/models")) {
            for url in contents(vendor) {
                models.append(model(url.lastPathComponent, url))
            }
        }
        for url in contents(URL.documentsDirectory.appending(path: "models")) {
            models.append(model("models/\(url.lastPathComponent)", url))
        }
        return models
    }

    static func delete(path: String) throws {
        guard list().contains(where: { $0.path == path }) else {
            throw Failure.notCached(path)
        }
        try FileManager.default.removeItem(atPath: path)
    }

    private static func model(_ name: String, _ url: URL) -> CachedModel {
        var bytes: Int64 = 0
        let files = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey])
        while let file = files?.nextObject() as? URL {
            bytes += Int64(
                (try? file.resourceValues(forKeys: [.totalFileAllocatedSizeKey])
                    .totalFileAllocatedSize) ?? 0)
        }
        return CachedModel(name: name, path: url.path, bytes: bytes)
    }

    private static func contents(_ url: URL) -> [URL] {
        let urls = try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
        return (urls ?? []).sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
