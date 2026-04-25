import AVFoundation
import Foundation
import Observation

@Observable
final class AudioController {

    // Internal LoopPlayer instance, lazily created when an asset is set.
    private var _player: LoopPlayer?

    /// Read-only access to the underlying LoopPlayer (may be nil if no asset set).
    var player: LoopPlayer? { _player }

    var isPlaying: Bool {
        _player?.isPlaying ?? false
    }

    /// Provide / update the audio asset to play in a loop.
    func setAsset(_ asset: AVURLAsset?) {
        if _player == nil {
            _player = LoopPlayer()
        }
        _player?.update(asset: asset)
    }
    
    var url: URL? {
        _player?.url
    }

    /// Start playback (no-op if already playing or no player set).
    func play() {
        _player?.play()
    }

    /// Pause playback (no-op if already paused or no player set).
    func pause() {
        _player?.pause()
    }

    /// Toggle between playing and paused states.
    func toggle() {
        isPlaying ? pause() : play()
    }
}
