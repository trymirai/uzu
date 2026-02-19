use std::path::PathBuf;

pub fn handle_nanocodec_roundtrip(
    nemo_path: String,
    input_wav: String,
    output_wav: String,
) -> Result<(), String> {
    let nemo_path = PathBuf::from(nemo_path);
    if !nemo_path.exists() {
        return Err(format!(".nemo archive not found: {}", nemo_path.display()));
    }

    let input_wav = PathBuf::from(input_wav);
    if !input_wav.exists() {
        return Err(format!("input wav not found: {}", input_wav.display()));
    }

    let output_wav = PathBuf::from(output_wav);

    let export_dir = std::env::temp_dir().join(format!("uzu-audio-export-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&export_dir)
        .map_err(|error| format!("failed to create temporary export directory: {error}"))?;

    let (_config, paths) = uzu::audio::tools::nemo::export_nanocodec_from_nemo(&nemo_path, &export_dir)
        .map_err(|error| format!("failed to export NanoCodec from .nemo: {error}"))?;

    Err(format!(
        "audio prototype CLI is now integration-only; exported codec assets to {} (config: {}, encoder: {}, decoder: {}, quantizer: {}). runtime encode/decode wiring is pending and output wav was not written to {}",
        export_dir.display(),
        paths.config_json.display(),
        paths.audio_encoder_safetensors.display(),
        paths.audio_decoder_safetensors.display(),
        paths.vector_quantizer_safetensors.display(),
        output_wav.display(),
    ))
}
