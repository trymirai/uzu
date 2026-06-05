use super::*;
use crate::{
    audio::AudioGenerationContext,
    config::{
        model::tts_model::TTSModelConfig, token_codec::AnyTokenCodecConfig, tts::text_decoder::AnyTTSTextDecoderConfig,
    },
};

pub(super) struct LoadedTtsRuntime<B: Backend> {
    pub(super) audio: AudioGenerationContext<B>,
    pub(super) audio_decoder: Box<dyn AudioDecoderBackend>,
    pub(super) text_decoder: Box<dyn SemanticDecoderBackend>,
    pub(super) token_codec_config: TTSCodecConfig,
}

pub(super) fn build_audio_decoder_backend<B: Backend + Send + Sync>(
    audio: &AudioGenerationContext<B>
) -> Result<Box<dyn AudioDecoderBackend>, Error> {
    Ok(Box::new(audio_backend::NanoCodecAudioDecoderBackend::new(audio.clone())))
}

pub(super) fn load_tts_runtime<B: Backend + Send + Sync>(
    model_path: &Path,
    model_config: &TTSModelConfig,
    options: &TtsSessionOptions,
) -> Result<LoadedTtsRuntime<B>, Error> {
    let audio = AudioGenerationContext::from_tts_config_and_model_path(&model_config.tts_config, model_path)?;
    let text_decoder = build_text_decoder_backend(model_config, &audio, model_path, options)?;
    let audio_decoder = build_audio_decoder_backend(&audio)?;
    let AnyTokenCodecConfig::TTSCodecConfig(token_codec_config) = &model_config.token_codec_config else {
        return Err(Error::InvalidModelConfig("expected TTSCodecConfig token codec".to_string()));
    };
    let token_codec_config = token_codec_config.clone();

    Ok(LoadedTtsRuntime {
        audio,
        audio_decoder,
        text_decoder,
        token_codec_config,
    })
}

pub(super) fn build_text_decoder_backend<B: Backend>(
    tts_model_config: &TTSModelConfig,
    audio: &AudioGenerationContext<B>,
    model_path: &Path,
    options: &TtsSessionOptions,
) -> Result<Box<dyn SemanticDecoderBackend>, Error> {
    match &tts_model_config.tts_config.text_decoder_config {
        AnyTTSTextDecoderConfig::FishAudioTextDecoderConfig(config) => {
            fishaudio::build_fishaudio_text_decoder_runtime(config, audio, model_path, &options.text_decoder)
        },
    }
}
