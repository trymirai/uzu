use std::path::Path;

use super::{
    AudioDecodeStreamState, AudioDecodeStreamingMode, NanoCodecFsqRuntimeConfig, StructuredAudioCodecGraph,
    StructuredAudioConv1dLayer, StructuredAudioConvNeXtLayer, StructuredAudioConvTranspose1dLayer,
    StructuredAudioDecoderBlockLayer, StructuredAudioDecoderGraph, StructuredAudioNormLayer,
    StructuredAudioResidualUnitLayer, StructuredAudioVectorQuantizer, Tensor3Json,
    convert_lalamo_transpose_weight_oih_to_iog, pack_pcm_to_padded, resolve_descript_audio_codec_vocoder_data_type,
    unpack_padded_to_pcm,
};
use crate::{
    DataType,
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioTokenGrid, AudioTokenPacking},
    config::TtsConfig,
};
#[test]
fn config_rejects_invalid_eps() {
    let config = NanoCodecFsqRuntimeConfig::new(
        24_000,
        2,
        vec![8, 6].into_boxed_slice(),
        1.1,
        AudioTokenPacking::FrameMajor,
    );

    assert!(matches!(config, Err(AudioError::Runtime(_))));
}

#[test]
fn pack_and_unpack_round_trip_variable_lengths() {
    let channels = 4usize;
    let lengths = vec![3usize, 1usize];
    let samples: Vec<f32> =
        (0..(lengths.iter().sum::<usize>() * channels)).map(|index| (index as f32 * 0.17 - 1.2).sin()).collect();

    let batch = AudioPcmBatch::new(samples.clone().into_boxed_slice(), 24_000, channels, lengths.clone().into())
        .expect("pcm");

    let (padded, got_lengths, _got_lengths_i32, frames) = pack_pcm_to_padded(&batch, channels).expect("pack");
    assert_eq!(frames, 3);
    assert_eq!(got_lengths, lengths);

    let unpacked = unpack_padded_to_pcm(&padded, 2, channels, frames, &lengths).expect("unpack");
    assert_eq!(samples, unpacked);
}

#[test]
fn runtime_config_builder_rejects_unsupported_type() {
    let tts_config = serde_json::json!({
        "text_decoder_config": {
            "type": "StubTextDecoderConfig",
            "num_codebooks": 2,
            "codebook_size": 48
        },
        "audio_decoder_config": {
            "type": "other_codec"
        },
        "vocoder_config": {}
    });

    let error = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect_err("must fail");
    match error {
        AudioError::Runtime(message) => assert!(message.contains("failed to parse TTS config")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn parses_current_tts_config_shape() {
    let tts_config = serde_json::json!({
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
                    "num_levels": [8, 6],
                    "eps": 1e-3
                }
            },
            "decoder_config": {
                "activation_config": {
                    "leaky_relu_negative_slope": 0.01
                }
            },
            "base_channels": 4,
            "up_sample_rates": [2],
            "resblock_kernel_sizes": [3],
            "resblock_dilations": [1]
        },
        "vocoder_config": {}
    });

    let config = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect("runtime config");
    assert_eq!(config.sample_rate(), 24_000);
    assert_eq!(config.num_groups(), 2);
    assert_eq!(config.num_levels_per_group(), &[8, 6]);
    assert_eq!(config.output_packing(), AudioTokenPacking::CodebookMajor);
    assert!(config.decoder().is_none());
}

#[test]
fn parses_lalamo_tts_config_shape() {
    let tts_config = serde_json::json!({
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
                    "num_levels": [8, 7, 6, 6],
                    "eps": 1e-3
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
        "vocoder_config": {},
        "activation_precision": "float32"
    });

    let parsed: TtsConfig = serde_json::from_value(tts_config).expect("parse");
    let crate::config::TtsAudioDecoderConfig::NanoCodecConfig {
        config,
    } = parsed.audio_decoder_config
    else {
        panic!("expected NanoCodecConfig");
    };
    assert_eq!(config.samplerate, 22050);
    assert_eq!(config.quantizer_config.num_groups, 13);
    assert_eq!(config.quantizer_config.quantizer_config.num_levels, vec![8, 7, 6, 6]);
    assert_eq!(config.up_sample_rates, vec![7, 7, 6, 3, 2]);
}

#[test]
fn fishaudio_dac_config_requires_model_weights() {
    let tts_config = serde_json::json!({
        "text_decoder_config": {
            "type": "StubTextDecoderConfig",
            "num_codebooks": 10,
            "codebook_size": 1024
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
    });

    let error = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, Path::new("."))
        .expect_err("fishaudio runtime must require exported model.safetensors");
    match error {
        AudioError::Runtime(message) => {
            assert!(message.contains("model.safetensors"), "unexpected error message: {message}");
        },
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
fn fishaudio_vocoder_dtype_uses_lalamo_export_precision() {
    let tts_config = serde_json::json!({
        "activation_precision": "float16",
        "text_decoder_config": {
            "type": "StubTextDecoderConfig",
            "num_codebooks": 10,
            "codebook_size": 1024
        },
        "audio_decoder_config": {
            "type": "DescriptAudioCodecConfig",
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
            "precision": "float16",
            "quantizer_config": {
                "precision": "float16",
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
    });

    let parsed: TtsConfig = serde_json::from_value(tts_config).expect("parse fishaudio config");
    let crate::config::TtsAudioDecoderConfig::DescriptAudioCodecConfig {
        config,
    } = &parsed.audio_decoder_config
    else {
        panic!("expected DescriptAudioCodecConfig");
    };
    let dtype =
        resolve_descript_audio_codec_vocoder_data_type(parsed.activation_precision, config).expect("resolve dtype");
    assert_eq!(dtype, DataType::F16);
}

#[test]
fn fishaudio_vocoder_dtype_rejects_conflicting_export_precision() {
    let tts_config = serde_json::json!({
        "activation_precision": "float32",
        "text_decoder_config": {
            "type": "StubTextDecoderConfig",
            "num_codebooks": 10,
            "codebook_size": 1024
        },
        "audio_decoder_config": {
            "type": "DescriptAudioCodecConfig",
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
            "precision": "float16",
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
    });

    let parsed: TtsConfig = serde_json::from_value(tts_config).expect("parse fishaudio config");
    let crate::config::TtsAudioDecoderConfig::DescriptAudioCodecConfig {
        config,
    } = &parsed.audio_decoder_config
    else {
        panic!("expected DescriptAudioCodecConfig");
    };
    let error = resolve_descript_audio_codec_vocoder_data_type(parsed.activation_precision, config)
        .expect_err("must reject conflicting precision");
    match error {
        AudioError::Runtime(message) => assert!(message.contains("conflicting DescriptAudioCodec precision")),
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
fn fishaudio_vocoder_config_requires_precision_field() {
    let tts_config = serde_json::json!({
        "text_decoder_config": {
            "type": "StubTextDecoderConfig",
            "num_codebooks": 10,
            "codebook_size": 1024
        },
        "audio_decoder_config": {
            "type": "DescriptAudioCodecConfig",
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
    });

    let error = serde_json::from_value::<TtsConfig>(tts_config).expect_err("precision should be required");
    assert!(error.to_string().contains("missing field `precision`"));
}

#[test]
fn fishaudio_runtime_decode_path_handles_empty_frames() {
    let mut config = NanoCodecFsqRuntimeConfig::new(
        44_100,
        2,
        vec![4].into_boxed_slice(),
        1e-3,
        AudioTokenPacking::CodebookMajor,
    )
    .expect("config");

    let one_by_one = StructuredAudioConv1dLayer {
        weight: vec![1.0],
        bias: vec![0.0],
        cin: 1,
        cout: 1,
        kernel_size: 1,
        dilation: 1,
        groups: 1,
    };

    config.structured_decoder = Some(StructuredAudioCodecGraph {
        semantic_quantizer: StructuredAudioVectorQuantizer {
            codebook: vec![0.0, 1.0],
            codebook_dim: 1,
            out_proj: vec![1.0],
            out_bias: vec![0.0],
        },
        residual_quantizers: vec![StructuredAudioVectorQuantizer {
            codebook: vec![0.0, 1.0],
            codebook_dim: 1,
            out_proj: vec![1.0],
            out_bias: vec![0.0],
        }],
        post_module_model_dim: 1,
        post_module_transformer_config: serde_json::from_value(serde_json::json!({
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
            "model_dim": 1,
            "hidden_dim": 1,
            "context_length": 1
        }))
        .expect("post module transformer config"),
        weights_path: String::new(),
        decoder: StructuredAudioDecoderGraph {
            first_conv: one_by_one.clone(),
            upsample_blocks: Vec::new(),
            decoder_blocks: Vec::new(),
            final_snake_alpha: vec![1.0],
            final_conv: one_by_one,
            upsample_factor: 1,
        },
        codebook_size: 2,
        semantic_codebook_size: 2,
        input_dim: 1,
        total_codebooks: 2,
        upsample_factor: 1,
        vocoder_data_type: DataType::F32,
    });

    let runtime = super::NanoCodecFsqRuntime::new(config);
    let tokens = crate::audio::AudioTokenGrid::new(
        Vec::new().into_boxed_slice(),
        1,
        2,
        0,
        vec![0usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("token grid");

    let pcm = runtime.decode(&tokens).expect("decode");
    assert_eq!(pcm.sample_rate(), 44_100);
    assert_eq!(pcm.channels(), 1);
    assert_eq!(pcm.lengths(), &[0usize]);
    assert!(pcm.samples().is_empty());
}

#[test]
fn fishaudio_quantizer_decode_gpu_matches_cpu_reference_small_graph() {
    let graph = StructuredAudioCodecGraph {
        semantic_quantizer: StructuredAudioVectorQuantizer {
            codebook: vec![
                0.0, 0.1, 0.2, //
                0.3, 0.4, 0.5, //
                0.6, 0.7, 0.8, //
                0.9, 1.0, 1.1, //
            ],
            codebook_dim: 3,
            out_proj: vec![
                0.2, -0.1, 0.3, //
                0.5, 0.4, -0.2, //
            ],
            out_bias: vec![0.01, -0.02],
        },
        residual_quantizers: vec![StructuredAudioVectorQuantizer {
            codebook: vec![
                0.2, -0.2, 0.0, //
                0.1, 0.0, 0.3, //
                -0.1, 0.4, 0.2, //
                0.7, -0.3, 0.5, //
                0.6, 0.8, -0.4, //
            ],
            codebook_dim: 3,
            out_proj: vec![
                0.3, 0.1, -0.2, //
                -0.4, 0.6, 0.2, //
            ],
            out_bias: vec![0.03, -0.01],
        }],
        post_module_model_dim: 2,
        post_module_transformer_config: serde_json::from_value(serde_json::json!({
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
            "model_dim": 2,
            "hidden_dim": 2,
            "context_length": 1
        }))
        .expect("post module transformer config"),
        weights_path: String::new(),
        decoder: StructuredAudioDecoderGraph {
            first_conv: StructuredAudioConv1dLayer {
                weight: vec![1.0, 0.0],
                bias: vec![0.0],
                cin: 2,
                cout: 1,
                kernel_size: 1,
                dilation: 1,
                groups: 1,
            },
            upsample_blocks: Vec::new(),
            decoder_blocks: Vec::new(),
            final_snake_alpha: vec![1.0],
            final_conv: StructuredAudioConv1dLayer {
                weight: vec![1.0],
                bias: vec![0.0],
                cin: 1,
                cout: 1,
                kernel_size: 1,
                dilation: 1,
                groups: 1,
            },
            upsample_factor: 1,
        },
        codebook_size: 5,
        semantic_codebook_size: 4,
        input_dim: 2,
        total_codebooks: 2,
        upsample_factor: 1,
        vocoder_data_type: DataType::F32,
    };

    let batch_size = 2usize;
    let frames = 4usize;
    let lengths = vec![4usize, 2usize];
    let tokens = vec![
        // batch 0 semantic
        0, 1, 2, 3, // batch 0 residual
        4, 5, 2, 1, // batch 1 semantic
        3, 2, 1, 0, // batch 1 residual
        0, 4, 3, 2,
    ];

    let expected = graph
        .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, graph.total_codebooks, frames)
        .expect("cpu decode");
    let actual = graph
        .decode_quantizer_to_nsc(&tokens, &lengths, batch_size, graph.total_codebooks, frames)
        .expect("gpu decode");

    assert_eq!(actual.len(), expected.len());
    for (index, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        let delta = (exp - got).abs();
        assert!(delta <= 1e-5, "mismatch at index {index}: expected {exp}, got {got}, delta={delta}");
    }
}

#[test]
fn transpose_weight_conversion_matches_expected_layout() {
    let weight_oih = Tensor3Json {
        shape: [2, 2, 2],
        values: vec![
            // out=0
            1.0, 2.0, // in_group=0
            3.0, 4.0, // in_group=1
            // out=1
            5.0, 6.0, // in_group=0
            7.0, 8.0, // in_group=1
        ],
    };

    let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, 4, 2, 2).expect("convert");
    assert_eq!(converted.shape, [4, 1, 2]);
    assert_eq!(
        converted.values,
        vec![
            1.0, 2.0, // in=0, out_group=0
            3.0, 4.0, // in=1, out_group=0
            5.0, 6.0, // in=2, out_group=0
            7.0, 8.0, // in=3, out_group=0
        ]
    );
}

#[test]
fn fishaudio_streaming_context_matches_expected_value() {
    let convnext = StructuredAudioConvNeXtLayer {
        depthwise_conv: StructuredAudioConv1dLayer {
            weight: vec![1.0; 6],
            bias: vec![0.0; 2],
            cin: 2,
            cout: 2,
            kernel_size: 3,
            dilation: 1,
            groups: 2,
        },
        norm: StructuredAudioNormLayer {
            scales: vec![1.0, 1.0],
            biases: Some(vec![0.0, 0.0]),
            epsilon: 1e-5,
            subtract_mean: true,
        },
        pwconv1: vec![1.0, 0.0, 0.0, 1.0],
        pwconv1_hidden_dim: 2,
        pwconv1_bias: vec![0.0, 0.0],
        pwconv2: vec![1.0, 0.0, 0.0, 1.0],
        pwconv2_bias: vec![0.0, 0.0],
    };
    let residual_unit = StructuredAudioResidualUnitLayer {
        snake1_alpha: vec![1.0, 1.0],
        conv1: StructuredAudioConv1dLayer {
            weight: vec![1.0; 12],
            bias: vec![0.0, 0.0],
            cin: 2,
            cout: 2,
            kernel_size: 3,
            dilation: 1,
            groups: 1,
        },
        snake2_alpha: vec![1.0, 1.0],
        conv2: StructuredAudioConv1dLayer {
            weight: vec![1.0; 12],
            bias: vec![0.0, 0.0],
            cin: 2,
            cout: 2,
            kernel_size: 3,
            dilation: 1,
            groups: 1,
        },
    };

    let graph = StructuredAudioCodecGraph {
        semantic_quantizer: StructuredAudioVectorQuantizer {
            codebook: vec![0.0, 0.0, 0.0, 0.0],
            codebook_dim: 2,
            out_proj: vec![1.0, 0.0, 0.0, 1.0],
            out_bias: vec![0.0, 0.0],
        },
        residual_quantizers: vec![],
        post_module_model_dim: 2,
        post_module_transformer_config: serde_json::from_value(serde_json::json!({
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
            "model_dim": 2,
            "hidden_dim": 2,
            "context_length": 1
        }))
        .expect("post module transformer config"),
        weights_path: String::new(),
        decoder: StructuredAudioDecoderGraph {
            first_conv: StructuredAudioConv1dLayer {
                weight: vec![1.0; 8],
                bias: vec![0.0, 0.0],
                cin: 2,
                cout: 2,
                kernel_size: 3,
                dilation: 1,
                groups: 1,
            },
            upsample_blocks: vec![(
                StructuredAudioConvTranspose1dLayer {
                    weight: vec![1.0; 16],
                    bias: vec![0.0, 0.0],
                    cin: 2,
                    cout: 2,
                    kernel_size: 4,
                    stride: 2,
                    groups: 1,
                },
                convnext,
            )],
            decoder_blocks: vec![StructuredAudioDecoderBlockLayer {
                snake_alpha: vec![1.0, 1.0],
                trans_conv: StructuredAudioConvTranspose1dLayer {
                    weight: vec![1.0; 16],
                    bias: vec![0.0, 0.0],
                    cin: 2,
                    cout: 2,
                    kernel_size: 4,
                    stride: 2,
                    groups: 1,
                },
                res_unit1: residual_unit.clone(),
                res_unit2: residual_unit.clone(),
                res_unit3: residual_unit,
            }],
            final_snake_alpha: vec![1.0, 1.0],
            final_conv: StructuredAudioConv1dLayer {
                weight: vec![1.0; 10],
                bias: vec![0.0],
                cin: 2,
                cout: 1,
                kernel_size: 5,
                dilation: 1,
                groups: 1,
            },
            upsample_factor: 4,
        },
        codebook_size: 2,
        semantic_codebook_size: 2,
        input_dim: 2,
        total_codebooks: 1,
        upsample_factor: 4,
        vocoder_data_type: DataType::F32,
    };

    // Manual derivation:
    // final_conv (k=5): 4
    // decoder block residual stack: +12 => 16, then trans_conv (k=4,s=2): ceil((16+3)/2)=10
    // first_conv (k=3): +2 => 12
    // upsample block convnext dwconv (k=3): +2 => 14, then trans_conv (k=4,s=2): ceil((14+3)/2)=9
    assert_eq!(graph.streaming_vocoder_context_frames().expect("stream context"), 9);
}

#[test]
fn stream_delta_extraction_with_window_offset_matches_expected_slice() {
    let mut state =
        AudioDecodeStreamState::new(1, 2, 16, AudioDecodeStreamingMode::IncrementalStateful).expect("state");
    let first_delta = AudioTokenGrid::new(
        vec![1_u32, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice(),
        1,
        2,
        4,
        vec![4usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("first token grid");
    state.append_delta(&first_delta).expect("append first delta");
    let first_decoded = crate::audio::nanocodec::decoder::DecodedPaddedAudio {
        samples: (0..8).map(|value| value as f32).collect(),
        channels: 1,
        frames: 8,
        lengths: vec![8],
    };
    let first_out = state.extract_delta_from_padded_with_offset(&first_decoded, 0, 2).expect("first extract");
    assert_eq!(first_out.lengths, vec![8]);
    assert_eq!(first_out.samples, (0..8).map(|value| value as f32).collect::<Vec<_>>());

    let second_delta = AudioTokenGrid::new(
        vec![9_u32, 10, 11, 12].into_boxed_slice(),
        1,
        2,
        2,
        vec![2usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("second token grid");
    state.append_delta(&second_delta).expect("append second delta");
    let window_decoded = crate::audio::nanocodec::decoder::DecodedPaddedAudio {
        // Global output range is [4, 12), local window starts at global sample 4.
        samples: (4..12).map(|value| value as f32).collect(),
        channels: 1,
        frames: 8,
        lengths: vec![8],
    };
    let second_out = state.extract_delta_from_padded_with_offset(&window_decoded, 4, 2).expect("second extract");
    assert_eq!(second_out.lengths, vec![4]);
    assert_eq!(second_out.frames, 4);
    assert_eq!(second_out.samples, vec![8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn incremental_stream_state_evicts_old_frames_with_bounded_workspace() {
    let mut state =
        AudioDecodeStreamState::new(1, 2, 4, AudioDecodeStreamingMode::IncrementalStateful).expect("state");
    let first_delta = AudioTokenGrid::new(
        vec![10_u32, 11, 12, 20, 21, 22].into_boxed_slice(),
        1,
        2,
        3,
        vec![3usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("first token grid");
    state.append_delta(&first_delta).expect("append first delta");

    let second_delta = AudioTokenGrid::new(
        vec![13_u32, 14, 15, 23, 24, 25].into_boxed_slice(),
        1,
        2,
        3,
        vec![3usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("second token grid");
    state.append_delta(&second_delta).expect("append second delta");

    assert_eq!(state.total_frames(), 6);
    assert_eq!(state.stored_frame_start, 2);
    assert_eq!(state.stored_frames(), 4);

    let (tokens, lengths, frames) = state.flatten_window(2, 6).expect("flatten retained window");
    assert_eq!(frames, 4);
    assert_eq!(lengths, &[4usize]);
    assert_eq!(tokens, &[12, 13, 14, 15, 22, 23, 24, 25]);
    assert!(state.to_full_grid().is_err(), "full-grid decode should fail after eviction");
}

#[test]
fn prefix_fallback_stream_state_rejects_workspace_overflow() {
    let mut state = AudioDecodeStreamState::new(1, 2, 4, AudioDecodeStreamingMode::PrefixFallback).expect("state");
    let first_delta = AudioTokenGrid::new(
        vec![1_u32, 2, 3, 4, 5, 6].into_boxed_slice(),
        1,
        2,
        3,
        vec![3usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("first token grid");
    state.append_delta(&first_delta).expect("append first delta");

    let overflow_delta = AudioTokenGrid::new(
        vec![7_u32, 8, 9, 10].into_boxed_slice(),
        1,
        2,
        2,
        vec![2usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("overflow token grid");
    let err = state.append_delta(&overflow_delta).expect_err("prefix fallback must reject overflow");
    assert!(err.to_string().contains("workspace exceeded"), "unexpected prefix overflow error: {err}");
}
