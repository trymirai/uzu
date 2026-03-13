use uzu::{ModelConfig, ModelMetadata, ModelType};

#[test]
fn metadata_with_tts_model_type_parses() {
    let config_json = r#"
    {
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
          "prompt_template": "{% for message in messages %}{{message.content}}{% endfor %}",
          "drop_initial_newline": true
        }
      }
    }
    "#;

    let metadata: ModelMetadata = serde_json::from_str(config_json).expect("metadata parse");
    assert_eq!(metadata.model_type, ModelType::TtsModel);

    let tts_config = metadata.model_config.as_tts().expect("tts config");
    assert_eq!(
        tts_config.message_processor_config.prompt_template,
        "{% for message in messages %}{{message.content}}{% endfor %}"
    );
    assert!(tts_config.message_processor_config.drop_initial_newline);
    assert_eq!(
        tts_config.message_processor_config.default_message_fields.get("speaker_id").map(String::as_str),
        Some("speaker:0")
    );
    assert_eq!(
        tts_config.message_processor_config.default_message_fields.get("style").map(String::as_str),
        Some("interleave")
    );
}

#[test]
fn tts_model_config_helpers_work() {
    let model_config_json = r#"
    {
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
    "#;

    let model_config: ModelConfig = serde_json::from_str(model_config_json).expect("model config parse");
    assert!(model_config.is_tts());
    assert!(model_config.as_tts().is_some());
    assert!(!model_config.is_classifier());
    assert!(!model_config.is_language_model());

    let tts = model_config.as_tts().expect("tts variant");
    assert_eq!(tts.message_processor_config.prompt_template, "{{messages[0].content}}");
    assert!(tts.message_processor_config.drop_initial_newline);
    assert_eq!(
        tts.message_processor_config.default_message_fields.get("speaker_id").map(String::as_str),
        Some("speaker:0")
    );
    assert_eq!(
        tts.message_processor_config.default_message_fields.get("style").map(String::as_str),
        Some("interleave")
    );
}

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
#[test]
fn tts_model_can_build_audio_codec_runtime() {
    let model_config_json = r#"
    {
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
        }
        ,
        "vocoder_config": {}
      },
      "message_processor_config": {
        "prompt_template": "{{messages[0].content}}"
      }
    }
    "#;

    let model_config: ModelConfig = serde_json::from_str(model_config_json).expect("model config parse");
    let tts = model_config.as_tts().expect("tts variant");
    let audio = tts.create_audio_generation_context().expect("audio context");
    assert_eq!(audio.sample_rate(), 24_000);
    assert_eq!(audio.num_codebooks(), 2);
    assert_eq!(audio.codec_cardinality(), 48);

    let runtime = tts.create_audio_codec_runtime().expect("audio runtime");
    let config = runtime.config();

    assert_eq!(config.sample_rate(), 24_000);
    assert_eq!(config.num_groups(), 2);
    assert_eq!(config.num_levels_per_group(), &[8, 6]);
    assert_eq!(config.channels(), 4);
    assert_eq!(config.codec_cardinality(), 48);
    assert_eq!(config.output_packing(), uzu::audio::AudioTokenPacking::CodebookMajor);
}
