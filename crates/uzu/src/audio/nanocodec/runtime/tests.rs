use std::path::{Path, PathBuf};

use super::{
    AudioDecodeStreamState, AudioDecodeStreamingMode, MatrixF32, NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig,
    StructuredAudioCodecGraph, StructuredAudioConv1dLayer, StructuredAudioConvNeXtLayer,
    StructuredAudioConvTranspose1dLayer, StructuredAudioDecoderBlockLayer, StructuredAudioDecoderGraph,
    StructuredAudioNormLayer, StructuredAudioResidualUnitLayer, StructuredAudioVectorQuantizer, Tensor3Json,
    convert_lalamo_transpose_weight_oih_to_iog, pack_pcm_to_padded, read_array_to_f32_vec,
    resolve_descript_audio_codec_vocoder_data_type, unpack_padded_to_pcm, write_f32_slice_to_array,
};
use crate::{
    DataType,
    array::ArrayContextExt,
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking},
    config::TtsConfig,
};

#[derive(Debug, Clone)]
struct CpuPostModuleLayer {
    pre_mixer_norm: StructuredAudioNormLayer,
    qkv_projection: MatrixF32,
    out_projection: MatrixF32,
    pre_mlp_norm: StructuredAudioNormLayer,
    up_projection: MatrixF32,
    down_projection: MatrixF32,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    attention_scale: f32,
    sliding_window_size: Option<usize>,
}

#[derive(Debug, Clone)]
struct CpuPostModule {
    rope_cosines: MatrixF32,
    rope_sines: MatrixF32,
    layers: Vec<CpuPostModuleLayer>,
    output_norm: StructuredAudioNormLayer,
    hidden_dim: usize,
}

fn load_optional_fishaudio_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LALAMO_UZU_MODEL_PATH") {
        let path = PathBuf::from(path);
        return if path.join("config.json").exists() && path.join("model.safetensors").exists() {
            Some(path)
        } else {
            None
        };
    }

    let default = PathBuf::from("/private/tmp/lalamo_fishaudio_s1mini_convert");
    if default.join("config.json").exists() && default.join("model.safetensors").exists() {
        Some(default)
    } else {
        None
    }
}

fn build_cpu_post_module_for_test(graph: &StructuredAudioCodecGraph) -> AudioResult<CpuPostModule> {
    let loader = super::TensorLoader::open(Path::new(graph.weights_path.as_str()))?;
    let post_module_tree = loader
        .tree()
        .subtree("audio_decoder")?
        .subtree("quantizer")?
        .subtree("post_module")?;
    let transformer = &graph.post_module_transformer_config;
    let first_layer = transformer
        .layer_configs
        .first()
        .ok_or(AudioError::Runtime("FishAudio post_module has no layers".to_string()))?;
    let crate::config::MixerConfig::Attention(first_attention) = &first_layer.mixer_config else {
        return Err(AudioError::Runtime(
            "FishAudio post_module first layer mixer must be AttentionConfig".to_string(),
        ));
    };
    let rope_head_dim = first_attention
        .head_dim
        .ok_or(AudioError::Runtime("FishAudio post_module head_dim missing".to_string()))?;
    let rope_tree = post_module_tree.subtree("global_rope")?;
    let rope_cosines = rope_tree.leaf_matrix_f32("cosines", transformer.context_length, rope_head_dim)?;
    let rope_sines = rope_tree.leaf_matrix_f32("sines", rope_cosines.rows, rope_cosines.cols)?;
    let output_norm = super::read_norm_layer(
        &post_module_tree.subtree("output_norm")?,
        transformer.output_norm_config.epsilon,
        transformer.output_norm_config.subtract_mean,
        false,
    )?;
    if output_norm.scales.len() != transformer.model_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio output_norm scale mismatch: expected {}, got {}",
            transformer.model_dim,
            output_norm.scales.len()
        )));
    }

    let mut layers = Vec::with_capacity(transformer.layer_configs.len());
    for (index, layer_config) in transformer.layer_configs.iter().enumerate() {
        let Some(pre_mixer_norm_config) = layer_config.pre_attention_norm_config.as_ref() else {
            return Err(AudioError::Runtime(
                "FishAudio post_module requires pre_attention_norm_config".to_string(),
            ));
        };

        let crate::config::MixerConfig::Attention(attention_config) = &layer_config.mixer_config else {
            return Err(AudioError::Runtime(
                "FishAudio post_module layer mixer must be AttentionConfig".to_string(),
            ));
        };
        let num_heads = attention_config
            .num_heads
            .ok_or(AudioError::Runtime("FishAudio attention num_heads missing".to_string()))?;
        let num_groups = attention_config
            .num_groups
            .ok_or(AudioError::Runtime("FishAudio attention num_groups missing".to_string()))?;
        let head_dim = attention_config
            .head_dim
            .ok_or(AudioError::Runtime("FishAudio attention head_dim missing".to_string()))?;
        if num_heads == 0 || num_groups == 0 || head_dim == 0 || num_heads % num_groups != 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        let layer_tree = post_module_tree.subtree("layers")?.subtree(&index.to_string())?;
        let pre_mixer_norm = super::read_norm_layer(
            &layer_tree.subtree("pre_mixer_norm")?,
            pre_mixer_norm_config.epsilon,
            pre_mixer_norm_config.subtract_mean,
            false,
        )?;
        let pre_mlp_norm = super::read_norm_layer(
            &layer_tree.subtree("pre_mlp_norm")?,
            layer_config.pre_mlp_norm_config.epsilon,
            layer_config.pre_mlp_norm_config.subtract_mean,
            false,
        )?;

        let attention_dim = num_heads
            .checked_mul(head_dim)
            .ok_or(AudioError::Runtime("FishAudio attention dimension overflow".to_string()))?;
        let mixer_tree = layer_tree.subtree("mixer")?;
        let qkv_projection = mixer_tree
            .subtree("qkv_projection")?
            .leaf_matrix_f32("weights", attention_dim * 3, transformer.model_dim)?;
        let out_projection = mixer_tree
            .subtree("out_projection")?
            .leaf_matrix_f32("weights", transformer.model_dim, attention_dim)?;
        let mlp_tree = layer_tree.subtree("mlp")?;
        let up_projection = mlp_tree.subtree("up_projection")?.leaf_matrix_f32(
            "weights",
            transformer
                .hidden_dim
                .checked_mul(2)
                .ok_or(AudioError::Runtime("FishAudio hidden dimension overflow".to_string()))?,
            transformer.model_dim,
        )?;
        let down_projection = mlp_tree
            .subtree("down_projection")?
            .leaf_matrix_f32("weights", transformer.model_dim, transformer.hidden_dim)?;

        layers.push(CpuPostModuleLayer {
            pre_mixer_norm,
            qkv_projection,
            out_projection,
            pre_mlp_norm,
            up_projection,
            down_projection,
            num_heads,
            num_groups,
            head_dim,
            attention_scale: attention_config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
            sliding_window_size: attention_config.sliding_window_size,
        });
    }

    Ok(CpuPostModule {
        rope_cosines,
        rope_sines,
        layers,
        output_norm,
        hidden_dim: transformer.hidden_dim,
    })
}

fn apply_post_module_cpu_reference_for_test(
    graph: &StructuredAudioCodecGraph,
    post_module: &CpuPostModule,
    latent_nsc: &mut [f32],
    lengths: &[usize],
    batch_size: usize,
    frames: usize,
) -> AudioResult<()> {
    if graph.post_module_model_dim != graph.input_dim {
        return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
    }

    for batch in 0..batch_size {
        let active_len = lengths[batch];
        if active_len == 0 {
            continue;
        }
        if active_len > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length: active_len,
                frames,
            });
        }

        let batch_base = batch * frames * graph.input_dim;
        let sequence = &mut latent_nsc[batch_base..batch_base + active_len * graph.input_dim];
        let mut x = sequence.to_vec();

        for layer in &post_module.layers {
            apply_post_module_cpu_layer_for_test(graph, post_module, layer, &mut x, active_len)?;
        }

        StructuredAudioCodecGraph::apply_norm_sequence(
            &mut x,
            active_len,
            graph.input_dim,
            &post_module.output_norm,
        )?;
        sequence.copy_from_slice(&x);
    }

    Ok(())
}

fn apply_post_module_cpu_layer_for_test(
    graph: &StructuredAudioCodecGraph,
    post_module: &CpuPostModule,
    layer: &CpuPostModuleLayer,
    x: &mut [f32],
    active_len: usize,
) -> AudioResult<()> {
    let mut normed = x.to_vec();
    StructuredAudioCodecGraph::apply_norm_sequence(
        &mut normed,
        active_len,
        graph.input_dim,
        &layer.pre_mixer_norm,
    )?;
    let qkv = StructuredAudioCodecGraph::linear_sequence(
        &normed,
        active_len,
        graph.input_dim,
        &layer.qkv_projection,
        None,
    )?;
    let attention_dim = layer
        .num_heads
        .checked_mul(layer.head_dim)
        .ok_or(AudioError::Runtime("attention dimension overflow".to_string()))?;
    let group_dim = layer
        .num_groups
        .checked_mul(layer.head_dim)
        .ok_or(AudioError::Runtime("group dimension overflow".to_string()))?;
    if attention_dim != group_dim {
        return Err(AudioError::Runtime("post_module CPU reference requires num_heads == num_groups".to_string()));
    }

    let mut q = vec![0.0_f32; active_len * attention_dim];
    let mut k = vec![0.0_f32; active_len * attention_dim];
    let mut v = vec![0.0_f32; active_len * attention_dim];
    for token in 0..active_len {
        let row = &qkv[token * (attention_dim * 3)..(token + 1) * (attention_dim * 3)];
        q[token * attention_dim..(token + 1) * attention_dim].copy_from_slice(&row[..attention_dim]);
        k[token * attention_dim..(token + 1) * attention_dim]
            .copy_from_slice(&row[attention_dim..attention_dim * 2]);
        v[token * attention_dim..(token + 1) * attention_dim]
            .copy_from_slice(&row[attention_dim * 2..attention_dim * 3]);
    }

    let half = layer.head_dim / 2;
    let q_source = q.clone();
    let k_source = k.clone();
    for token in 0..active_len {
        for head in 0..layer.num_heads {
            let rope_row = token.min(post_module.rope_cosines.rows.saturating_sub(1));
            for dim in 0..layer.head_dim {
                let cos = post_module.rope_cosines.values[rope_row * layer.head_dim + dim];
                let sin = post_module.rope_sines.values[rope_row * layer.head_dim + dim];
                let base = token * attention_dim + head * layer.head_dim;
                let qv = q_source[base + dim];
                let kv = k_source[base + dim];
                let q_pair = if dim < half {
                    -q_source[base + dim + half]
                } else {
                    q_source[base + dim - half]
                };
                let k_pair = if dim < half {
                    -k_source[base + dim + half]
                } else {
                    k_source[base + dim - half]
                };
                q[base + dim] = qv * cos + q_pair * sin;
                k[base + dim] = kv * cos + k_pair * sin;
            }
        }
    }

    let mut attention_output = vec![0.0_f32; active_len * attention_dim];
    for token in 0..active_len {
        let window_start =
            layer.sliding_window_size.map(|window| token.saturating_sub(window.saturating_sub(1))).unwrap_or(0);

        for head in 0..layer.num_heads {
            let query_offset = token * attention_dim + head * layer.head_dim;
            let mut logits = Vec::with_capacity(token + 1 - window_start);
            let mut max_logit = f32::NEG_INFINITY;
            for key_token in window_start..=token {
                let key_offset = key_token * attention_dim + head * layer.head_dim;
                let mut score = 0.0_f32;
                for dim in 0..layer.head_dim {
                    score += q[query_offset + dim] * k[key_offset + dim];
                }
                score *= layer.attention_scale;
                max_logit = max_logit.max(score);
                logits.push((key_token, score));
            }
            let mut denom = 0.0_f32;
            for (_, score) in logits.iter_mut() {
                *score = (*score - max_logit).exp();
                denom += *score;
            }
            if denom <= 0.0 {
                continue;
            }
            for (key_token, score) in logits {
                let weight = score / denom;
                let value_offset = key_token * attention_dim + head * layer.head_dim;
                for dim in 0..layer.head_dim {
                    attention_output[query_offset + dim] += weight * v[value_offset + dim];
                }
            }
        }
    }

    let attention_projected = StructuredAudioCodecGraph::linear_sequence(
        &attention_output,
        active_len,
        attention_dim,
        &layer.out_projection,
        None,
    )?;
    for (dst, value) in x.iter_mut().zip(attention_projected.iter()) {
        *dst += *value;
    }

    let mut mlp_in = x.to_vec();
    StructuredAudioCodecGraph::apply_norm_sequence(&mut mlp_in, active_len, graph.input_dim, &layer.pre_mlp_norm)?;
    let up = StructuredAudioCodecGraph::linear_sequence(
        &mlp_in,
        active_len,
        graph.input_dim,
        &layer.up_projection,
        None,
    )?;
    let mut hidden = vec![0.0_f32; active_len * post_module.hidden_dim];
    for token in 0..active_len {
        let up_row = &up[token * post_module.hidden_dim * 2..(token + 1) * post_module.hidden_dim * 2];
        let hidden_row = &mut hidden[token * post_module.hidden_dim..(token + 1) * post_module.hidden_dim];
        for dim in 0..post_module.hidden_dim {
            let up_val = up_row[dim];
            let gate_val = up_row[post_module.hidden_dim + dim];
            let silu = gate_val / (1.0 + (-gate_val).exp());
            hidden_row[dim] = up_val * silu;
        }
    }
    let mlp_out = StructuredAudioCodecGraph::linear_sequence(
        &hidden,
        active_len,
        post_module.hidden_dim,
        &layer.down_projection,
        None,
    )?;
    for (dst, value) in x.iter_mut().zip(mlp_out.iter()) {
        *dst += *value;
    }

    Ok(())
}

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
            codebook: MatrixF32 {
                rows: 2,
                cols: 1,
                values: vec![0.0, 1.0],
            },
            out_proj: MatrixF32 {
                rows: 1,
                cols: 1,
                values: vec![1.0],
            },
            out_bias: vec![0.0],
        },
        residual_quantizers: vec![StructuredAudioVectorQuantizer {
            codebook: MatrixF32 {
                rows: 2,
                cols: 1,
                values: vec![0.0, 1.0],
            },
            out_proj: MatrixF32 {
                rows: 1,
                cols: 1,
                values: vec![1.0],
            },
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
            codebook: MatrixF32 {
                rows: 4,
                cols: 3,
                values: vec![
                    0.0, 0.1, 0.2, //
                    0.3, 0.4, 0.5, //
                    0.6, 0.7, 0.8, //
                    0.9, 1.0, 1.1, //
                ],
            },
            out_proj: MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![
                    0.2, -0.1, 0.3, //
                    0.5, 0.4, -0.2, //
                ],
            },
            out_bias: vec![0.01, -0.02],
        },
        residual_quantizers: vec![StructuredAudioVectorQuantizer {
            codebook: MatrixF32 {
                rows: 5,
                cols: 3,
                values: vec![
                    0.2, -0.2, 0.0, //
                    0.1, 0.0, 0.3, //
                    -0.1, 0.4, 0.2, //
                    0.7, -0.3, 0.5, //
                    0.6, 0.8, -0.4, //
                ],
            },
            out_proj: MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![
                    0.3, 0.1, -0.2, //
                    -0.4, 0.6, 0.2, //
                ],
            },
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
fn fishaudio_post_module_gpu_matches_cpu_reference_on_real_export() {
    let Some(model_path) = load_optional_fishaudio_model_path() else {
        println!("Skipping FishAudio post-module parity test: set LALAMO_UZU_MODEL_PATH");
        return;
    };

    let config_bytes = std::fs::read(model_path.join("config.json")).expect("read model config");
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).expect("parse model config");
    let tts_config = config_json
        .get("model_config")
        .and_then(|value| value.get("tts_config"))
        .expect("model_config.tts_config")
        .clone();
    let runtime_config = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, &model_path)
        .expect("runtime config");
    let fishaudio = runtime_config.structured_decoder().expect("structured decoder");
    let cpu_post_module = build_cpu_post_module_for_test(fishaudio).expect("cpu post-module");

    let batch_size = 1usize;
    let frames = 6usize;
    let lengths = vec![6usize];
    let mut tokens = vec![0_u32; batch_size * fishaudio.total_codebooks * frames];
    for frame in 0..frames {
        let semantic = (frame * 17) % fishaudio.semantic_codebook_size;
        tokens[frame] = semantic as u32;
        for residual in 0..(fishaudio.total_codebooks - 1) {
            let index = ((residual + 1) * frames) + frame;
            let value = ((frame + 3) * (residual + 5)) % fishaudio.codebook_size;
            tokens[index] = value as u32;
        }
    }
    let latent_cpu = fishaudio
        .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
        .expect("decode quantizer cpu reference");
    let latent = fishaudio
        .decode_quantizer_to_nsc(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
        .expect("decode quantizer");
    assert_eq!(latent.len(), latent_cpu.len());
    let mut quantizer_max_abs_diff = 0.0_f32;
    for (&cpu_value, &gpu_value) in latent_cpu.iter().zip(latent.iter()) {
        quantizer_max_abs_diff = quantizer_max_abs_diff.max((cpu_value - gpu_value).abs());
    }
    println!("quantizer latent parity: max_abs_diff={quantizer_max_abs_diff}");
    assert!(quantizer_max_abs_diff <= 1e-4, "quantizer decode parity mismatch: {quantizer_max_abs_diff}");

    let mut cpu = latent.clone();
    apply_post_module_cpu_reference_for_test(fishaudio, &cpu_post_module, &mut cpu, &lengths, batch_size, frames)
        .expect("cpu post-module");
    let context = NanoCodecFsqRuntime::create_context().expect("create metal context");
    let mut latent_array = context.create_array(
        &[batch_size, frames, fishaudio.input_dim],
        fishaudio.vocoder_data_type,
        "fishaudio_test_post_module_single_input_nsc",
    );
    write_f32_slice_to_array(&mut latent_array, &latent).expect("write latent to array");
    let mut profile = None;
    let gpu = read_array_to_f32_vec(
        &fishaudio
            .apply_post_module_gpu_on_array(&context, &latent_array, &lengths, batch_size, frames, &mut profile)
            .expect("gpu post-module"),
    )
    .expect("read gpu post-module");

    let mut max_abs_diff = 0.0_f32;
    let mut sum_sq_diff = 0.0_f64;
    for (&cpu_value, &gpu_value) in cpu.iter().zip(gpu.iter()) {
        let diff = (cpu_value - gpu_value).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_sq_diff += f64::from(diff * diff);
    }
    let rmse = (sum_sq_diff / cpu.len() as f64).sqrt() as f32;
    println!("post-module latent parity: max_abs_diff={max_abs_diff}, rmse={rmse}");

    assert!(max_abs_diff <= 1e-4, "post-module max_abs_diff too high: {max_abs_diff}, rmse={rmse}");
    assert!(rmse <= 1e-5, "post-module rmse too high: {rmse}, max_abs_diff={max_abs_diff}");
}

#[test]
fn fishaudio_post_module_gpu_general_path_batches_lengths_in_one_command_buffer() {
    let Some(model_path) = load_optional_fishaudio_model_path() else {
        println!("Skipping FishAudio multi-batch post-module test: set LALAMO_UZU_MODEL_PATH");
        return;
    };

    let config_bytes = std::fs::read(model_path.join("config.json")).expect("read model config");
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).expect("parse model config");
    let tts_config = config_json
        .get("model_config")
        .and_then(|value| value.get("tts_config"))
        .expect("model_config.tts_config")
        .clone();
    let runtime_config = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, &model_path)
        .expect("runtime config");
    let fishaudio = runtime_config.structured_decoder().expect("structured decoder");
    let cpu_post_module = build_cpu_post_module_for_test(fishaudio).expect("cpu post-module");

    let batch_size = 3usize;
    let frames = 6usize;
    let lengths = vec![6usize, 4usize, 4usize];
    let mut tokens = vec![0_u32; batch_size * fishaudio.total_codebooks * frames];
    for batch in 0..batch_size {
        for frame in 0..frames {
            let semantic = ((batch + 1) * 17 + frame * 13) % fishaudio.semantic_codebook_size;
            let semantic_index = ((batch * fishaudio.total_codebooks) * frames) + frame;
            tokens[semantic_index] = semantic as u32;
            for residual in 0..(fishaudio.total_codebooks - 1) {
                let index = ((batch * fishaudio.total_codebooks) + residual + 1) * frames + frame;
                let value = ((batch + 3) * (frame + 5) * (residual + 7)) % fishaudio.codebook_size;
                tokens[index] = value as u32;
            }
        }
    }

    let latent_cpu = fishaudio
        .decode_quantizer_to_nsc_reference(&tokens, &lengths, batch_size, fishaudio.total_codebooks, frames)
        .expect("decode quantizer cpu reference");
    let mut cpu = latent_cpu.clone();
    apply_post_module_cpu_reference_for_test(fishaudio, &cpu_post_module, &mut cpu, &lengths, batch_size, frames)
        .expect("cpu post-module");

    let context = NanoCodecFsqRuntime::create_context().expect("create metal context");
    let mut latent = context.create_array(
        &[batch_size, frames, fishaudio.input_dim],
        fishaudio.vocoder_data_type,
        "fishaudio_test_post_module_input_nsc",
    );
    write_f32_slice_to_array(&mut latent, &latent_cpu).expect("write latent to array");
    let mut profile = Some(super::AudioDecodeProfile::default());
    let output = fishaudio
        .apply_post_module_gpu_on_array(&context, &latent, &lengths, batch_size, frames, &mut profile)
        .expect("gpu post-module");
    let gpu = read_array_to_f32_vec(&output).expect("read gpu post-module");

    let mut max_abs_diff = 0.0_f32;
    let mut sum_sq_diff = 0.0_f64;
    for (&cpu_value, &gpu_value) in cpu.iter().zip(gpu.iter()) {
        let diff = (cpu_value - gpu_value).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_sq_diff += f64::from(diff * diff);
    }
    let rmse = (sum_sq_diff / cpu.len() as f64).sqrt() as f32;
    println!("multi-batch post-module latent parity: max_abs_diff={max_abs_diff}, rmse={rmse}");

    assert!(max_abs_diff <= 2e-4, "post-module max_abs_diff too high: {max_abs_diff}, rmse={rmse}");
    assert!(rmse <= 2e-5, "post-module rmse too high: {rmse}, max_abs_diff={max_abs_diff}");

    let profile = profile.expect("profile");
    assert_eq!(profile.command_buffers.len(), batch_size, "expected one post-module command buffer per batch item");
    assert_eq!(profile.command_buffers[0].label, "post_module_len_4_batch_1");
    assert_eq!(profile.command_buffers[1].label, "post_module_len_4_batch_2");
    assert_eq!(profile.command_buffers[2].label, "post_module_len_6_batch_0");
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
        pwconv1: MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        },
        pwconv1_bias: vec![0.0, 0.0],
        pwconv2: MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        },
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
            codebook: MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![0.0, 0.0, 0.0, 0.0],
            },
            out_proj: MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
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
