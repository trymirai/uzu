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
            "type": "DescriptAudioCodecConfig",
            "precision": "float16",
            "samplerate": 44100,
            "encoder_dim": 1024,
            "encoder_rates": [2, 4, 8, 8],
            "decoder_dim": 1536,
            "n_codebooks": 9,
            "codebook_dim": 8,
            "codebook_size": 1024,
            "semantic_codebook_size": 4096,
            "input_dim": 1024,
            "downsample_factor": [2, 2],
            "decoder_rates": [8, 8, 4, 2],
            "decoder_config": {},
            "quantizer_config": {
              "post_module_config": {
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [],
                "output_norm_config": {
                  "scale_precision": "float32",
                  "accumulation_precision": "float32",
                  "epsilon": 1e-5,
                  "scale_offset": null,
                  "upcast_mode": "full_layer",
                  "subtract_mean": true
                },
                "model_dim": 1024,
                "hidden_dim": 1024,
                "context_length": 1
              },
              "upsampler_config": {
                "block_configs": []
              }
            }
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
    assert!(tts_config.message_processor_config.default_message_fields.is_empty());
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
          "type": "DescriptAudioCodecConfig",
          "precision": "float16",
          "samplerate": 44100,
          "encoder_dim": 1024,
          "encoder_rates": [2, 4, 8, 8],
          "decoder_dim": 1536,
          "n_codebooks": 9,
          "codebook_dim": 8,
          "codebook_size": 1024,
          "semantic_codebook_size": 4096,
          "input_dim": 1024,
          "downsample_factor": [2, 2],
          "decoder_rates": [8, 8, 4, 2],
          "decoder_config": {},
          "quantizer_config": {
            "post_module_config": {
              "global_rope_config": null,
              "local_rope_config": null,
              "layer_configs": [],
              "output_norm_config": {
                "scale_precision": "float32",
                "accumulation_precision": "float32",
                "epsilon": 1e-5,
                "scale_offset": null,
                "upcast_mode": "full_layer",
                "subtract_mean": true
              },
              "model_dim": 1024,
              "hidden_dim": 1024,
              "context_length": 1
            },
            "upsampler_config": {
              "block_configs": []
            }
          }
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
    assert!(tts.message_processor_config.default_message_fields.is_empty());
}
