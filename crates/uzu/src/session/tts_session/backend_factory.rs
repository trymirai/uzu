use super::*;
use crate::config::{ModelType, TtsModelConfig, TtsTextDecoderConfig};

pub(super) struct LoadedTtsRuntime {
    pub(super) audio: AudioGenerationContext,
    pub(super) audio_decoder: Box<dyn AudioDecoderBackend>,
    pub(super) text_decoder: Box<dyn SemanticDecoderBackend>,
    pub(super) message_processor_config: TtsMessageProcessorConfig,
}

pub(super) fn build_audio_decoder_backend(
    audio: &AudioGenerationContext
) -> Result<Box<dyn AudioDecoderBackend>, Error> {
    Ok(Box::new(audio_backend::NanoCodecAudioDecoderBackend::new(audio.clone())))
}

pub(super) fn load_tts_runtime(
    model_path: &Path,
    model_metadata: &ModelMetadata,
    options: &TtsSessionOptions,
) -> Result<LoadedTtsRuntime, Error> {
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
    let audio = tts_model_config
        .create_audio_generation_context_with_model_path_and_options(model_path, options.audio_runtime)?;
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

fn validate_stub_text_decoder_contract(
    num_codebooks: usize,
    codebook_size: usize,
    audio_num_codebooks: usize,
    audio_codec_cardinality: usize,
    audio_semantic_codec_cardinality: Option<usize>,
) -> Result<(), Error> {
    if num_codebooks == 0 || codebook_size == 0 {
        return Err(Error::InvalidTtsModelConfig(format!(
            "stub decoder requires positive num_codebooks and codebook_size, got num_codebooks={num_codebooks}, codebook_size={codebook_size}"
        )));
    }
    if num_codebooks != audio_num_codebooks {
        return Err(Error::InvalidTtsModelConfig(format!(
            "stub decoder num_codebooks={num_codebooks} does not match audio num_codebooks={audio_num_codebooks}"
        )));
    }
    if codebook_size != audio_codec_cardinality {
        return Err(Error::InvalidTtsModelConfig(format!(
            "stub decoder codebook_size={codebook_size} does not match audio codec_cardinality={audio_codec_cardinality}"
        )));
    }
    if let Some(semantic_codec_cardinality) = audio_semantic_codec_cardinality {
        if semantic_codec_cardinality != audio_codec_cardinality {
            return Err(Error::InvalidTtsModelConfig(format!(
                "stub decoder requires uniform semantic/residual codec cardinality, got semantic_codec_cardinality={semantic_codec_cardinality}, residual_codec_cardinality={audio_codec_cardinality}"
            )));
        }
    }
    Ok(())
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
            validate_stub_text_decoder_contract(
                *num_codebooks,
                *codebook_size,
                audio.num_codebooks(),
                audio.codec_cardinality(),
                audio.semantic_codec_cardinality(),
            )?;
            let default_seed = load_stub_seed(model_path.join("model.safetensors")).unwrap_or(DEFAULT_STUB_SEED);
            Ok(Box::new(StubTextDecoderRuntime {
                num_codebooks: *num_codebooks,
                codebook_size: *codebook_size,
                default_seed,
            }))
        },
        TtsTextDecoderConfig::FishAudioTextDecoderConfig {
            config,
        } => {
            fishaudio::validate_fishaudio_message_processor_config(&tts_model_config.message_processor_config)?;
            fishaudio::build_fishaudio_text_decoder_runtime(config, audio, model_path, &options.text_decoder)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::validate_stub_text_decoder_contract;
    use crate::session::types::Error;

    #[test]
    fn stub_decoder_contract_accepts_uniform_codebooks() {
        let result = validate_stub_text_decoder_contract(4, 1024, 4, 1024, Some(1024));
        assert!(result.is_ok());
    }

    #[test]
    fn stub_decoder_contract_rejects_heterogeneous_semantic_codebook() {
        let result = validate_stub_text_decoder_contract(4, 1024, 4, 1024, Some(512));
        assert!(matches!(result, Err(Error::InvalidTtsModelConfig(_))));
    }
}
