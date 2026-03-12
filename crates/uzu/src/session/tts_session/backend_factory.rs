use super::*;

pub(super) fn build_audio_decoder_backend(
    audio: &AudioGenerationContext,
) -> Result<Box<dyn AudioDecoderBackend>, Error> {
    Ok(Box::new(audio_backend::NanoCodecAudioDecoderBackend::new(audio.clone())))
}

pub(super) fn build_text_decoder_backend(
    tts_model_config: &TtsModelConfig,
    audio: &AudioGenerationContext,
    model_path: &Path,
    options: &TtsSessionOptions,
) -> Result<Box<dyn SemanticDecoderBackend>, Error> {
    match &tts_model_config.tts_config.text_decoder_config {
        TtsTextDecoderConfig::StubTextDecoderConfig {
            num_codebooks,
            codebook_size,
        } => {
            if *num_codebooks == 0 || *codebook_size == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            if *num_codebooks != audio.num_codebooks() {
                return Err(Error::UnableToLoadConfig);
            }
            let default_seed = load_stub_seed(model_path.join("model.safetensors")).unwrap_or(DEFAULT_STUB_SEED);
            Ok(Box::new(StubTextDecoderRuntime {
                num_codebooks: *num_codebooks,
                codebook_size: *codebook_size,
                default_seed,
            }))
        },
        TtsTextDecoderConfig::FishAudioTextDecoderConfig {
            config,
        } => fishaudio::build_fishaudio_text_decoder_runtime(config, audio, model_path, &options.text_decoder),
    }
}
