#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::Device;
use uzu::{audio_codec::NanoCodecDecoder, backends::metal::MTLContext};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn nanocodec_decode_smoke_env() {
    // To run with real weights (preferred):
    //   export UZU_NANOCODEC_EXPORT_DIR=/path/to/export_dir
    // where export_dir is produced by the Rust `.nemo` exporter
    // (`uzu::audio_codec::nemo::export_nanocodec_from_nemo`) and contains:
    //   - nanocodec_config.json
    //   - audio_decoder.safetensors
    //
    // Alternatively:
    //   export UZU_NANOCODEC_DECODER_SAFETENSORS=/path/to/audio_decoder.safetensors
    // Then:
    //   cargo test -p uzu --test nanocodec_decode_smoke_test -- --nocapture

    let ctx = create_test_context().expect("MTLContext");
    let ctx = std::rc::Rc::new(ctx);

    let decoder = if let Some(export_dir) =
        std::env::var("UZU_NANOCODEC_EXPORT_DIR").ok()
    {
        let export_dir = PathBuf::from(export_dir);
        assert!(export_dir.exists(), "export dir not found");
        NanoCodecDecoder::load_from_export_dir(ctx, &export_dir)
            .expect("NanoCodecDecoder::load_from_export_dir")
    } else if let Some(decoder_path) =
        std::env::var("UZU_NANOCODEC_DECODER_SAFETENSORS").ok()
    {
        let decoder_path = PathBuf::from(decoder_path);
        assert!(decoder_path.exists(), "decoder safetensors not found");
        // Default NanoCodec 22k 1.78kbps 12.5fps config:
        // - 13 codebooks
        // - 4 dims per codebook
        // - levels [8,7,6,6] (product 2016)
        NanoCodecDecoder::load(ctx, &decoder_path, 13, 4, Box::new([8, 7, 6, 6]))
            .expect("NanoCodecDecoder::load")
    } else {
        eprintln!(
            "Skipping: set UZU_NANOCODEC_EXPORT_DIR or UZU_NANOCODEC_DECODER_SAFETENSORS"
        );
        return;
    };

    let batch_size = 1usize;
    let seq_len = 8usize;
    let tokens: Vec<i32> = (0..(batch_size * 13 * seq_len))
        .map(|i| ((i * 37 + 11) % 2016) as i32)
        .collect();
    let lengths = vec![seq_len as i32; batch_size];

    let (audio, audio_len) =
        decoder.decode(&tokens, &lengths, batch_size, seq_len).expect("decode");

    assert_eq!(audio_len.len(), batch_size);
    // NanoCodec 22k 12.5fps uses samples_per_frame=1764 (upsample rates [7,7,6,3,2])
    assert_eq!(audio_len[0] as usize, seq_len * 1764);
    assert_eq!(audio.len(), audio_len[0] as usize);
}

