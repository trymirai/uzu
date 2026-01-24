#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::Device;

use uzu::{audio_codec::NanoCodecModel, backends::metal::MTLContext};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn nanocodec_roundtrip_smoke_from_nemo() {
    let nemo_path = match std::env::var("UZU_NANOCODEC_NEMO").ok() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Skipping: UZU_NANOCODEC_NEMO not set");
            return;
        },
    };
    assert!(nemo_path.exists(), ".nemo not found");

    let ctx = create_test_context().expect("MTLContext");
    let ctx = std::rc::Rc::new(ctx);

    let model =
        NanoCodecModel::load_from_nemo(ctx, &nemo_path).expect("load");

    let batch_size = 1usize;
    let seq_len = 2000usize; // intentionally not multiple of samples_per_frame
    let audio: Vec<f32> = (0..seq_len)
        .map(|i| ((i as f32) * 0.01).sin() * 0.2)
        .collect();
    let audio_len = vec![seq_len as i32];

    let (out, out_len) =
        model.roundtrip(&audio, &audio_len, batch_size, seq_len).expect("roundtrip");

    let expected_len = 2 * model.samples_per_frame;
    assert_eq!(out_len, vec![expected_len as i32]);
    assert_eq!(out.len(), expected_len);
    for &x in out.iter() {
        assert!(x.is_finite());
        assert!(x >= -1.0001 && x <= 1.0001);
    }
}

