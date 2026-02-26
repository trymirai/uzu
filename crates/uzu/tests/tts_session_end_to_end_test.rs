#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::path::PathBuf;
use std::sync::Mutex;

use uzu::session::{TtsSession, types::Input};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn load_optional_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LALAMO_UZU_MODEL_PATH") {
        let path = PathBuf::from(path);
        return if path.join("config.json").exists() && path.join("model.safetensors").exists() {
            Some(path)
        } else {
            None
        };
    }

    let default = PathBuf::from("/tmp/lalamo_nanocodec_convert");
    if default.join("config.json").exists() && default.join("model.safetensors").exists() {
        Some(default)
    } else {
        None
    }
}

#[test]
fn tts_session_synthesizes_audio_from_text_on_normal_export() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_model_path() else {
        println!("Skipping TTS session test: set LALAMO_UZU_MODEL_PATH (or create /tmp/lalamo_nanocodec_convert)");
        return;
    };

    let session = TtsSession::new(model_path).expect("tts session");
    let pcm = session.synthesize(Input::Text("hello".to_string())).expect("synthesize");

    assert_eq!(pcm.sample_rate(), session.runtime().config().sample_rate());
    assert_eq!(pcm.lengths().len(), 1);
    assert_eq!(pcm.channels(), 1);
    assert!(pcm.lengths()[0] > 0, "expected non-empty waveform for non-empty text");
    assert_eq!(pcm.samples().len(), pcm.lengths()[0] * pcm.channels());
}
