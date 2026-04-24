import Foundation

extension ChatMessage {
    public func textDecoded<T: Decodable>() -> T? {
        guard let text = self.text() else {
            return nil
        }
        let decoder = JSONDecoder()
        let jsonData = Data(text.utf8)
        let entity = try? decoder.decode(T.self, from: jsonData)
        return entity
    }
}
