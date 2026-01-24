#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::Device;

use uzu::{
    audio_codec::{NanoCodecEncoder},
    backends::metal::MTLContext,
    Array,
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn nanocodec_encoder_smoke_from_nemo_export() {
    let nemo_path = match std::env::var("UZU_NANOCODEC_NEMO").ok() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Skipping: UZU_NANOCODEC_NEMO not set");
            return;
        },
    };
    assert!(nemo_path.exists(), ".nemo not found");

    let export_dir = tempfile::TempDir::new().expect("temp dir");
    let (cfg, _paths) = uzu::audio_codec::nemo::export_nanocodec_from_nemo(
        &nemo_path,
        export_dir.path(),
    )
    .expect("export");

    let ctx = create_test_context().expect("MTLContext");
    let ctx = std::rc::Rc::new(ctx);

    let encoder = NanoCodecEncoder::load_from_export_dir(
        ctx,
        export_dir.path(),
        cfg.audio_encoder.down_sample_rates.clone().into_boxed_slice(),
    )
    .expect("load encoder");

    let batch_size = 1usize;
    let seq_len = cfg.samples_per_frame * 3;
    let audio: Vec<f32> = (0..seq_len)
        .map(|i| ((i as f32) * 0.01).sin() * 0.5)
        .collect();
    let lengths = vec![seq_len as i32];

    let (encoded, encoded_len) =
        encoder
            .encode_latents(&audio, &lengths, batch_size, seq_len)
            .expect("encode_latents");

    assert_eq!(encoded.shape(), &[1, cfg.audio_encoder.encoded_dim, 3]);
    assert_eq!(encoded_len, vec![3]);
}

