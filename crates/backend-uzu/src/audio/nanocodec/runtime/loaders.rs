use super::*;

pub(super) fn resolve_descript_audio_codec_vocoder_data_type(
    top_level_precision: Option<DataType>,
    config: &DescriptAudioCodecConfig,
) -> AudioResult<DataType> {
    let mut resolved_precision: Option<DataType> = None;
    for (field_name, precision) in [
        ("tts_config.activation_precision", top_level_precision),
        ("tts_config.audio_decoder_config.precision", Some(config.precision)),
        ("tts_config.audio_decoder_config.quantizer_config.precision", config.quantizer_config.precision),
    ] {
        if let Some(precision) = precision {
            if let Some(existing) = resolved_precision {
                if existing != precision {
                    return Err(AudioError::Runtime(format!(
                        "conflicting DescriptAudioCodec precision in Lalamo export: {field_name}={precision:?} conflicts with {existing:?}"
                    )));
                }
            } else {
                resolved_precision = Some(precision);
            }
        }
    }

    let precision = resolved_precision.ok_or(AudioError::Runtime(
        "missing DescriptAudioCodec precision in Lalamo export; expected one of tts_config.activation_precision, \
tts_config.audio_decoder_config.precision, or tts_config.audio_decoder_config.quantizer_config.precision"
            .to_string(),
    ))?;
    let data_type: DataType = precision.into();
    if !matches!(data_type, DataType::F32 | DataType::F16 | DataType::BF16) {
        return Err(AudioError::Runtime(format!(
            "unsupported DescriptAudioCodec vocoder precision in Lalamo export: {precision:?} (expected float32/float16/bfloat16)"
        )));
    }
    Ok(data_type)
}

pub(super) fn load_audio_runtime_from_tts_config(
    tts_config: &TtsConfig,
    model_path: &Path,
) -> AudioResult<(RuntimeConfigJson, StructuredAudioCodecGraph)> {
    let cfg = match &tts_config.audio_decoder_config {
        TtsAudioDecoderConfig::DescriptAudioCodecConfig {
            config,
        } => config,
    };
    let fishaudio_weights = model_path.join("model.safetensors");
    if !fishaudio_weights.is_file() {
        return Err(AudioError::Runtime(format!(
            "missing exported FishAudio decoder weights '{}'",
            fishaudio_weights.display()
        )));
    }

    let vocoder_data_type = resolve_descript_audio_codec_vocoder_data_type(tts_config.activation_precision, cfg)?;

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
        weights_path: fishaudio_weights.display().to_string(),
        codebook_size: cfg.codebook_size,
        semantic_codebook_size: cfg.semantic_codebook_size,
        input_dim: cfg.input_dim,
        total_codebooks,
        upsample_factor,
        vocoder_data_type,
    };
    Ok((runtime, decoder))
}
