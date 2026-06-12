use super::*;

pub(super) fn load_audio_runtime_from_tts_config(
    tts_config: &TTSConfig,
    model_path: &Path,
) -> AudioResult<(RuntimeConfigJson, StructuredAudioCodecGraph)> {
    let AnyTTSAudioDecoderConfig::DescriptAudioCodecConfig(cfg) = &tts_config.audio_decoder_config;
    let weights_path = model_path.join("model.safetensors");
    let vocoder_data_type = DataType::BF16;

    let total_codebooks = cfg
        .n_codebooks
        .checked_add(1)
        .ok_or(AudioError::Runtime("FishAudio codebook count overflow while building runtime config".to_string()))?;
    let codebook_size_i32 = i32::try_from(cfg.codebook_size)
        .map_err(|_| AudioError::Runtime("FishAudio codebook_size exceeds i32 kernel range".to_string()))?;
    if codebook_size_i32 <= 1 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let upsample_factor = cfg
        .downsample_factor
        .iter()
        .chain(cfg.decoder_rates.iter())
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("FishAudio upsample factor overflow".to_string()))?;

    let runtime = RuntimeConfigJson {
        sample_rate: cfg.samplerate,
        num_groups: total_codebooks,
        num_levels_per_group: vec![codebook_size_i32],
        eps: default_eps(),
    };
    let decoder = StructuredAudioCodecGraph {
        config: cfg.clone(),
        weights_path: weights_path.display().to_string(),
        codebook_size: cfg.codebook_size,
        semantic_codebook_size: cfg.semantic_codebook_size,
        input_dim: cfg.input_dim,
        total_codebooks,
        upsample_factor,
        vocoder_data_type,
    };
    Ok((runtime, decoder))
}
