use super::*;
use crate::config::{ModelType, TtsModelConfig, TtsTextDecoderConfig};

pub(super) struct LoadedTtsRuntime<B: Backend> {
    pub(super) audio: AudioGenerationContext<B>,
    pub(super) audio_decoder: Box<dyn AudioDecoderBackend>,
    pub(super) text_decoder: Box<dyn SemanticDecoderBackend>,
    pub(super) message_processor_config: TtsMessageProcessorConfig,
}

pub(super) fn build_audio_decoder_backend<B: Backend + Send + Sync>(
    audio: &AudioGenerationContext<B>
) -> Result<Box<dyn AudioDecoderBackend>, Error> {
    Ok(Box::new(audio_backend::NanoCodecAudioDecoderBackend::new(audio.clone())))
}

pub(super) fn load_tts_runtime<B: Backend + Send + Sync>(
    model_path: &Path,
    model_metadata: &ModelMetadata,
    options: &TtsSessionOptions,
) -> Result<LoadedTtsRuntime<B>, Error> {
    if model_metadata.model_type != ModelType::TtsModel {
        return Err(Error::InvalidTtsModelConfig(format!(
            "expected model_type={:?}, got {:?}",
            ModelType::TtsModel,
            model_metadata.model_type
        )));
    }

    let tts_model_config = model_metadata
        .model_config
        .as_tts()
        .ok_or_else(|| Error::InvalidTtsModelConfig("missing TTS model config".to_string()))?
        .clone();
    let audio: AudioGenerationContext<B> = tts_model_config
        .create_audio_generation_context_with_model_path(model_path)?;
    let text_decoder = build_text_decoder_backend(&tts_model_config, &audio, model_path, options)?;
    let audio_decoder = build_audio_decoder_backend(&audio)?;
    let message_processor_config = tts_model_config.message_processor_config.clone();

    Ok(LoadedTtsRuntime {
        audio,
        audio_decoder,
        text_decoder,
        message_processor_config,
    })
}

pub(super) fn build_text_decoder_backend<B: Backend>(
    tts_model_config: &TtsModelConfig,
    audio: &AudioGenerationContext<B>,
    model_path: &Path,
    options: &TtsSessionOptions,
) -> Result<Box<dyn SemanticDecoderBackend>, Error> {
    match &tts_model_config.tts_config.text_decoder_config {
        TtsTextDecoderConfig::FishAudioTextDecoderConfig {
            config,
        } => {
            fishaudio::validate_fishaudio_message_processor_config(&tts_model_config.message_processor_config)?;
            fishaudio::build_fishaudio_text_decoder_runtime(config, audio, model_path, &options.text_decoder)
        },
    }
}
