#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::{fs::File, io::BufReader, path::PathBuf};

use crate::{
    audio::{AudioCodecRuntime, AudioGenerationContext, AudioPcmBatch, AudioTokenGrid, NanoCodecFsqRuntime},
    config::{ModelMetadata, ModelType},
    session::types::Error,
};

pub struct TtsCodecSession {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    model_metadata: ModelMetadata,
    audio: AudioGenerationContext,
}

impl TtsCodecSession {
    pub fn new(model_path: PathBuf) -> Result<Self, Error> {
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        Self::from_model_metadata(model_path, model_metadata)
    }

    pub fn runtime(&self) -> &NanoCodecFsqRuntime {
        self.audio.runtime()
    }

    pub fn encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> Result<AudioTokenGrid, Error> {
        self.audio.runtime().encode(pcm).map_err(Error::from)
    }

    pub fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> Result<AudioPcmBatch, Error> {
        self.audio.runtime().decode(tokens).map_err(Error::from)
    }

    fn from_model_metadata(
        model_path: PathBuf,
        model_metadata: ModelMetadata,
    ) -> Result<Self, Error> {
        if model_metadata.model_type != ModelType::TtsModel {
            return Err(Error::UnableToLoadConfig);
        }

        let tts_config = model_metadata.model_config.as_tts().ok_or(Error::UnableToLoadConfig)?;
        let audio = tts_config.create_audio_generation_context_with_model_path(&model_path)?;

        Ok(Self {
            model_path,
            model_metadata,
            audio,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::TtsCodecSession;
    use crate::{
        audio::{AudioPcmBatch, AudioTokenPacking},
        config::ModelMetadata,
        session::types::Error,
    };

    fn build_metadata(model_type: &str) -> ModelMetadata {
        serde_json::from_value(serde_json::json!({
            "toolchain_version": "0.0.0-test",
            "vendor": "NVIDIA",
            "family": "nanocodec",
            "name": "nemo-nano-codec-22khz-1.78kbps-12.5fps",
            "size": "0.1B",
            "quantization": null,
            "repo": "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
            "use_cases": [],
            "model_type": model_type,
            "model_config": {
                "tts_config": {
                    "text_decoder_config": {
                        "type": "StubTextDecoderConfig",
                        "num_codebooks": 2,
                        "codebook_size": 48
                    },
                    "audio_decoder_config": {
                        "type": "NanoCodecConfig",
                        "samplerate": 24000,
                        "quantizer_config": {
                            "num_groups": 2,
                            "quantizer_config": {
                                "num_levels": [8, 6]
                            }
                        },
                        "decoder_config": {
                            "activation_config": {
                                "leaky_relu_negative_slope": 0.01
                            }
                        },
                        "base_channels": 32,
                        "up_sample_rates": [2, 2],
                        "resblock_kernel_sizes": [3],
                        "resblock_dilations": [1]
                    },
                    "vocoder_config": {}
                },
                "message_processor_config": {
                    "prompt_template": "{{messages[0].content}}"
                }
            }
        }))
        .expect("metadata parse")
    }

    fn make_pcm(channels: usize) -> AudioPcmBatch {
        let lengths = vec![3usize, 2usize];
        let sample_count = lengths.iter().sum::<usize>() * channels;
        let samples: Vec<f32> = (0..sample_count).map(|index| (index as f32 * 0.13 - 0.6).sin()).collect();
        AudioPcmBatch::new(samples.into_boxed_slice(), 24_000, channels, lengths.into_boxed_slice()).expect("pcm")
    }

    #[test]
    fn tts_codec_session_encodes_and_decodes() {
        let metadata = build_metadata("tts_model");
        let tts_config = metadata.model_config.as_tts().expect("tts config");
        let audio = tts_config.create_audio_generation_context().expect("audio context");
        let session = TtsCodecSession {
            model_path: PathBuf::from("."),
            model_metadata: metadata,
            audio,
        };

        let pcm = make_pcm(session.runtime().config().channels());
        let tokens = session.encode(&pcm).expect("encode");
        let decoded = session.decode(&tokens).expect("decode");

        assert_eq!(tokens.packing(), AudioTokenPacking::CodebookMajor);
        assert_eq!(decoded.sample_rate(), session.runtime().config().sample_rate());
        assert_eq!(decoded.channels(), session.runtime().config().channels());
        assert_eq!(decoded.lengths(), pcm.lengths());
    }

    #[test]
    fn tts_codec_session_rejects_non_tts_model_type() {
        let metadata = build_metadata("language_model");
        let result = TtsCodecSession::from_model_metadata(PathBuf::from("."), metadata);
        assert!(matches!(result, Err(Error::UnableToLoadConfig)));
    }

    #[test]
    fn tts_codec_session_requires_weights_for_lalamo_tts_config() {
        let metadata: ModelMetadata = serde_json::from_value(serde_json::json!({
            "toolchain_version": "0.0.0-test",
            "vendor": "NVIDIA",
            "family": "nanocodec",
            "name": "nemo-nano-codec-22khz-1.78kbps-12.5fps",
            "size": "0.1B",
            "quantization": null,
            "repo": "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
            "use_cases": [],
            "model_type": "tts_model",
            "model_config": {
                "tts_config": {
                    "text_decoder_config": {
                        "type": "StubTextDecoderConfig",
                        "num_codebooks": 13,
                        "codebook_size": 336
                    },
                    "audio_decoder_config": {
                        "type": "NanoCodecConfig",
                        "samplerate": 22050,
                        "quantizer_config": {
                            "num_groups": 13,
                            "quantizer_config": {
                                "num_levels": [8, 7, 6, 6]
                            }
                        },
                        "decoder_config": {
                            "activation_config": {
                                "leaky_relu_negative_slope": 0.01
                            }
                        },
                        "base_channels": 864,
                        "up_sample_rates": [7, 7, 6, 3, 2],
                        "resblock_kernel_sizes": [3, 7, 11],
                        "resblock_dilations": [1, 3, 5]
                    },
                    "vocoder_config": {}
                },
                "message_processor_config": {
                    "prompt_template": "{{messages[0].content}}"
                }
            }
        }))
        .expect("metadata parse");

        let result = TtsCodecSession::from_model_metadata(PathBuf::from("/does/not/exist"), metadata);
        assert!(matches!(result, Err(Error::AudioCodec(_))));
    }
}
