import AVFoundation
import SwiftUI

extension NSDataAsset: @retroactive AVAssetResourceLoaderDelegate {
    @objc
    public func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest
    ) -> Bool {
        if let infoRequest = loadingRequest.contentInformationRequest {
            infoRequest.contentType = typeIdentifier
            infoRequest.contentLength = Int64(self.data.count)
            infoRequest.isByteRangeAccessSupported = true
        }

        if let dataRequest = loadingRequest.dataRequest {
            let subdataRange =
                Int(dataRequest.requestedOffset)..<Int(dataRequest.requestedOffset)
                + dataRequest.requestedLength
            dataRequest.respond(with: self.data.subdata(in: subdataRange))
            loadingRequest.finishLoading()
            return true
        }
        return false
    }

    var url: URL? {
        guard let name = self.name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
            let url = URL(string: "NSDataAsset://\(name))")
        else { return nil }

        return url
    }
}

extension AVURLAsset {
    enum Error: Swift.Error {
        case loadingAssetFailed

        var localizedDescription: String {
            switch self {
            case .loadingAssetFailed: return "Failed to load asset."
            }
        }

        var recoverySuggestion: String? {
            switch self {
            case .loadingAssetFailed: return "Try using a different asset."
            }
        }
    }

    public convenience init?(_ dataAsset: NSDataAsset) {
        guard let url = dataAsset.url
        else { return nil }

        self.init(url: url)
        self.resourceLoader.setDelegate(dataAsset, queue: .main)
        // Retain the weak delegate for the lifetime of AVURLAsset
        objc_setAssociatedObject(
            self, "AVURLAsset+NSDataAsset", dataAsset, .OBJC_ASSOCIATION_RETAIN
        )
    }
}
