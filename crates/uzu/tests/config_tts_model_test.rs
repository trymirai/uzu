use uzu::{ModelConfig, ModelMetadata, ModelType};

fn descript_audio_decoder_config_json() -> serde_json::Value {
    serde_json::json!({
        "precision": "float16",
        "conv_config": {
            "precision": "float16",
            "has_biases": true
        },
        "snake_config": {
            "precision": "float16"
        },
        "decoder_block_config": {
            "precision": "float16",
            "snake_config": {
                "precision": "float16"
            },
            "trans_conv_config": {
                "precision": "float16",
                "has_biases": true
            },
            "res_unit_config": {
                "precision": "float16",
                "snake_config": {
                    "precision": "float16"
                },
                "conv_config": {
                    "precision": "float16",
                    "has_biases": true
                },
                "causal": true
            },
            "causal": true
        },
        "causal": true
    })
}

fn fishaudio_text_decoder_config_json() -> serde_json::Value {
    let norm_config = serde_json::json!({
        "scale_precision": "float32",
        "accumulation_precision": "float32",
        "epsilon": 1e-5,
        "scale_offset": null,
        "upcast_mode": "full_layer",
        "subtract_mean": true
    });
    let transformer_config = serde_json::json!({
        "global_rope_config": null,
        "local_rope_config": null,
        "layer_configs": [],
        "output_norm_config": norm_config,
        "model_dim": 256,
        "hidden_dim": 256,
        "context_length": 512
    });
    serde_json::json!({
        "type": "FishAudioTextDecoderConfig",
        "slow_embeddings_config": { "precision": "float16", "input_scale": null, "logit_soft_cap": null },
        "slow_model_config": transformer_config,
        "slow_readout_config": { "precision": "float16" },
        "fast_embeddings_config": { "precision": "float16", "input_scale": null, "logit_soft_cap": null },
        "fast_model_config": transformer_config,
        "fast_readout_config": { "precision": "float16" },
        "codebook_embeddings_config": { "precision": "float16", "input_scale": null, "logit_soft_cap": null },
        "fast_model_projection_config": null,
        "semantic_token_begin_id": 100,
        "semantic_token_end_id": 1123,
        "im_end_token_id": 4,
        "codebook_size": 1024,
        "vocab_size": 2048,
        "slow_model_dim": 256,
        "fast_model_dim": 256,
        "num_codebooks": 8,
        "max_seq_len": 4096,
        "scale_codebook_embeddings": false,
        "short_logits_size": 1124,
        "repeat_window_size": 16
    })
}

#[test]
fn metadata_with_tts_model_type_parses() {
    let config_json = serde_json::json!({
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
          "text_decoder_config": fishaudio_text_decoder_config_json(),
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
            "decoder_config": descript_audio_decoder_config_json(),
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
        },
        "message_processor_config": {
          "prompt_template": "{% for message in messages %}{{message.content}}{% endfor %}",
          "drop_initial_newline": true
        }
      }
    });

    let metadata: ModelMetadata = serde_json::from_value(config_json).expect("metadata parse");
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
    let model_config_json = serde_json::json!({
      "tts_config": {
        "text_decoder_config": fishaudio_text_decoder_config_json(),
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
          "decoder_config": descript_audio_decoder_config_json(),
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
      },
      "message_processor_config": {
        "prompt_template": "{{messages[0].content}}"
      }
    });

    let model_config: ModelConfig = serde_json::from_value(model_config_json).expect("model config parse");
    assert!(model_config.is_tts());
    assert!(model_config.as_tts().is_some());
    assert!(!model_config.is_classifier());
    assert!(!model_config.is_language_model());

    let tts = model_config.as_tts().expect("tts variant");
    assert_eq!(tts.message_processor_config.prompt_template, "{{messages[0].content}}");
    assert!(tts.message_processor_config.drop_initial_newline);
    assert!(tts.message_processor_config.default_message_fields.is_empty());
}

#[test]
fn tts_model_config_rejects_untyped_decoder_config() {
    let model_config_json = serde_json::json!({
      "tts_config": {
        "text_decoder_config": fishaudio_text_decoder_config_json(),
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
      },
      "message_processor_config": {
        "prompt_template": "{{messages[0].content}}"
      }
    });

    let error = serde_json::from_value::<ModelConfig>(model_config_json).expect_err("decoder_config should be typed");
    let error_text = error.to_string();
    assert!(
        error_text.contains("did not match any variant") || error_text.contains("conv_config"),
        "unexpected error: {error}"
    );
}
