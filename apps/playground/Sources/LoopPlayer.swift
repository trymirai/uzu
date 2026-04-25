import AVFoundation
import Foundation
import Observation

#if os(macOS)
    import AppKit
#endif

@Observable
final class LoopPlayer {

    var isPlaying = false

    private let player: AVPlayer = {
        let p = AVPlayer()
        p.actionAtItemEnd = .none
        return p
    }()

    private var endObserver: Any?
    
    var url: URL? {
        if
            let item = player.currentItem,
            let asset = item.asset as? AVURLAsset
        {
            let url = asset.url
            return url
        }
        return nil
    }

    init() {
        #if os(iOS)
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [])
            try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self,
                let item = note.object as? AVPlayerItem,
                item == self.player.currentItem
            else { return }

            item.seek(to: .zero) { [weak self] _ in
                guard let self else { return }
                if self.isPlaying { self.player.play() }
            }
        }
    }

    deinit {
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
        }
    }

    func update(asset: AVURLAsset?) {
        if let asset {
            player.replaceCurrentItem(with: AVPlayerItem(asset: asset))
        } else {
            player.replaceCurrentItem(with: nil)
        }
    }

    func play() {
        guard !isPlaying else { return }
        player.play()
        isPlaying = true
    }

    func pause() {
        guard isPlaying else { return }
        player.pause()
        isPlaying = false
    }
}
