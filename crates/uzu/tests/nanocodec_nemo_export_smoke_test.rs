#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::Device;

use uzu::{
    audio_codec::{NanoCodecDecoder},
    backends::metal::MTLContext,
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn nanocodec_export_from_nemo_then_decode_smoke() {
    let nemo_path = match std::env::var("UZU_NANOCODEC_NEMO").ok() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Skipping: UZU_NANOCODEC_NEMO not set");
            return;
        },
    };
    assert!(nemo_path.exists(), ".nemo not found");

    let export_dir = tempfile::TempDir::new().expect("temp dir");

    let (_cfg, _paths) =
        uzu::audio_codec::nemo::export_nanocodec_from_nemo(
            &nemo_path,
            export_dir.path(),
        )
        .expect("export_nanocodec_from_nemo");

    let ctx = create_test_context().expect("MTLContext");
    let ctx = std::rc::Rc::new(ctx);

    let decoder = NanoCodecDecoder::load_from_export_dir(ctx, export_dir.path())
        .expect("load_from_export_dir");

    let batch_size = 1usize;
    let seq_len = 4usize;
    let num_codebooks = 13usize;

    let tokens: Vec<i32> = (0..(batch_size * num_codebooks * seq_len))
        .map(|i| ((i * 37 + 11) % 2016) as i32)
        .collect();
    let lengths = vec![seq_len as i32; batch_size];

    let (audio, audio_len) =
        decoder.decode(&tokens, &lengths, batch_size, seq_len).expect("decode");

    assert_eq!(audio_len[0] as usize, seq_len * 1764);
    assert_eq!(audio.len(), audio_len[0] as usize);
}

