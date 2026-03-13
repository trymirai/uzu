use super::*;
use crate::config::{ModelType, TtsModelConfig, TtsTextDecoderConfig};

pub(super) struct LoadedTtsRuntime {
    pub(super) audio: AudioGenerationContext,
    pub(super) audio_decoder: Box<dyn AudioDecoderBackend>,
    pub(super) text_decoder: Box<dyn SemanticDecoderBackend>,
    pub(super) prompt_template: String,
    pub(super) drop_initial_newline: bool,
}

pub(super) fn build_audio_decoder_backend(
    audio: &AudioGenerationContext,
) -> Result<Box<dyn AudioDecoderBackend>, Error> {
    Ok(Box::new(audio_backend::NanoCodecAudioDecoderBackend::new(audio.clone())))
}

pub(super) fn load_tts_runtime(
    model_path: &Path,
    model_metadata: &ModelMetadata,
    options: &TtsSessionOptions,
) -> Result<LoadedTtsRuntime, Error> {
    if model_metadata.model_type != ModelType::TtsModel {
        return Err(Error::UnableToLoadConfig);
    }

    let tts_model_config = model_metadata.model_config.as_tts().ok_or(Error::UnableToLoadConfig)?.clone();
    let audio =
        tts_model_config.create_audio_generation_context_with_model_path_and_options(model_path, options.audio_runtime)?;
    let text_decoder = build_text_decoder_backend(&tts_model_config, &audio, model_path, options)?;
    let audio_decoder = build_audio_decoder_backend(&audio)?;

    Ok(LoadedTtsRuntime {
        audio,
        audio_decoder,
        text_decoder,
        prompt_template: tts_model_config.message_processor_config.prompt_template.clone(),
        drop_initial_newline: tts_model_config.message_processor_config.drop_initial_newline,
    })
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
