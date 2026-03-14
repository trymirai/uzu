use super::*;
use crate::{
    audio::nanocodec::decoder::NanoCodecDecoderJson,
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    parameters::{ParameterLoader, ParameterTree},
    utils::array_io::read_array_to_f32_vec,
};

#[derive(Debug, Clone, Copy)]
enum TransposeConvWeightLayout {
    Oih,
}

type AudioParameterContext = <Cpu as Backend>::Context;
type AudioParameterTree<'loader> = ParameterTree<'loader, AudioParameterContext>;

fn map_parameter_error(
    err: crate::parameters::ParameterLoaderError<<AudioParameterContext as Context>::Backend>,
) -> AudioError {
    AudioError::Runtime(err.to_string())
}

fn subtree<'loader>(
    tree: &AudioParameterTree<'loader>,
    name: &str,
) -> AudioResult<AudioParameterTree<'loader>> {
    tree.subtree(name).map_err(map_parameter_error)
}

fn read_tensor_f32(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<(Box<[usize]>, Vec<f32>)> {
    let array = tree.leaf_array(name).map_err(map_parameter_error)?;
    let shape = array.shape().iter().copied().collect::<Vec<_>>().into_boxed_slice();
    Ok((shape, read_array_to_f32_vec(&array)?))
}

fn leaf_vector_f32(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = read_tensor_f32(tree, name)?;
    if shape.len() != 1 {
        return Err(AudioError::Runtime(format!("expected rank-1 tensor for '{name}', got shape {shape:?}")));
    }
    Ok(values)
}

fn leaf_tensor_3(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<Tensor3Json> {
    let (shape, values) = read_tensor_f32(tree, name)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{name}', got shape {shape:?}")));
    }
    Ok(Tensor3Json {
        shape: [shape[0], shape[1], shape[2]],
        values,
    })
}

fn leaf_matrix_f32(
    tree: &AudioParameterTree<'_>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = read_tensor_f32(tree, name)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{name}', got shape {shape:?}")));
    }
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {shape:?}"
        )));
    }
    Ok(values)
}

fn leaf_matrix_f32_any(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<([usize; 2], Vec<f32>)> {
    let (shape, values) = read_tensor_f32(tree, name)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{name}', got shape {shape:?}")));
    }
    Ok(([shape[0], shape[1]], values))
}

fn read_causal_conv_1d(
    tree: &AudioParameterTree<'_>,
    dilation: usize,
) -> AudioResult<CausalConv1dJson> {
    Ok(CausalConv1dJson {
        weight: leaf_tensor_3(tree, "weights")?,
        bias: leaf_vector_f32(tree, "biases")?,
        dilation,
    })
}

pub(super) fn convert_lalamo_transpose_weight_oih_to_iog(
    weight: &Tensor3Json,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
) -> AudioResult<Tensor3Json> {
    let [out_channels_in_weight, in_channels_per_group, kernel_size] = weight.shape;
    if out_channels_in_weight != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv out-channel mismatch: expected {out_channels}, got {out_channels_in_weight}"
        )));
    }
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid transpose conv grouping: in_channels={in_channels}, out_channels={out_channels}, groups={groups}"
        )));
    }

    let expected_in_per_group = in_channels / groups;
    if in_channels_per_group != expected_in_per_group {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight shape mismatch: expected in_per_group={expected_in_per_group}, got {in_channels_per_group}"
        )));
    }

    let out_channels_per_group = out_channels / groups;
    let expected_weight_len = checked_product(&[out_channels, in_channels_per_group, kernel_size])?;
    if weight.values.len() != expected_weight_len {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight value length mismatch: expected {expected_weight_len}, got {}",
            weight.values.len()
        )));
    }

    let converted_len = checked_product(&[in_channels, out_channels_per_group, kernel_size])?;
    let mut converted = vec![0.0_f32; converted_len];

    for group in 0..groups {
        let in_base = group * in_channels_per_group;
        let out_base = group * out_channels_per_group;

        for out_idx in 0..out_channels_per_group {
            for in_idx in 0..in_channels_per_group {
                for k in 0..kernel_size {
                    let src_index = ((out_base + out_idx) * in_channels_per_group + in_idx) * kernel_size + k;
                    let dst_index = ((in_base + in_idx) * out_channels_per_group + out_idx) * kernel_size + k;
                    converted[dst_index] = weight.values[src_index];
                }
            }
        }
    }

    Ok(Tensor3Json {
        shape: [in_channels, out_channels_per_group, kernel_size],
        values: converted,
    })
}

pub(super) fn build_nanocodec_decoder_graph_from_lalamo_config(
    config: &NanoCodecAudioDecoderConfig,
    model_weights_path: &Path,
) -> AudioResult<NanoCodecDecoderGraph> {
    let cfg = config;
    let file = File::open(model_weights_path).map_err(|err| {
        AudioError::Runtime(format!("failed to open safetensors file '{}': {err}", model_weights_path.display()))
    })?;
    let host_context = <AudioParameterContext as Context>::new()
        .map_err(|err| AudioError::Runtime(format!("failed to create cpu parameter context: {err}")))?;
    let loader = ParameterLoader::new(&file, host_context.as_ref()).map_err(|err| {
        AudioError::Runtime(format!(
            "failed to parse safetensors metadata from '{}': {err}",
            model_weights_path.display()
        ))
    })?;
    let audio_decoder_tree = subtree(&loader.tree(), "audio_decoder")?;
    let decoder_tree = subtree(&audio_decoder_tree, "decoder")?;

    let pre_conv = read_causal_conv_1d(&subtree(&decoder_tree, "pre_conv")?, 1)?;
    let mut stages = Vec::with_capacity(cfg.up_sample_rates.len());

    let mut current_channels = cfg.base_channels;
    for (stage_index, &stride) in cfg.up_sample_rates.iter().enumerate() {
        if stride == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid upsample stride at stage {stage_index}: stride must be > 0"
            )));
        }
        let out_channels = current_channels
            .checked_div(2)
            .ok_or(AudioError::Runtime("decoder channel progression overflow".to_string()))?;
        if out_channels == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid decoder stage {stage_index}: current_channels={current_channels}, expected > 1"
            )));
        }
        let groups = out_channels;

        let activations_tree = subtree(&decoder_tree, "activations")?;
        let activation_alpha = leaf_vector_f32(
            &subtree(&subtree(&activations_tree, &stage_index.to_string())?, "snake")?,
            "alpha",
        )?;

        let upsample_convs_tree = subtree(&decoder_tree, "upsample_convs")?;
        let upsample_tree = subtree(&upsample_convs_tree, &stage_index.to_string())?;
        let upsample_weight_oih = leaf_tensor_3(&upsample_tree, "weights")?;
        let upsample_bias = leaf_vector_f32(&upsample_tree, "biases")?;
        let upsample_weight =
            convert_lalamo_transpose_weight_oih_to_iog(&upsample_weight_oih, current_channels, out_channels, groups)?;
        let upsample_conv = CausalConvTranspose1dJson {
            weight: upsample_weight,
            bias: upsample_bias,
            stride,
            groups,
        };

        let mut res_blocks = Vec::with_capacity(cfg.resblock_kernel_sizes.len());
        for (res_block_index, _) in cfg.resblock_kernel_sizes.iter().enumerate() {
            let mut residuals = Vec::with_capacity(cfg.resblock_dilations.len());
            for (residual_index, &dilation) in cfg.resblock_dilations.iter().enumerate() {
                let res_layers_tree = subtree(&decoder_tree, "res_layers")?;
                let residual_tree = subtree(
                    &subtree(
                        &subtree(
                            &subtree(&subtree(&res_layers_tree, &stage_index.to_string())?, "res_blocks")?,
                            &res_block_index.to_string(),
                        )?,
                        "res_blocks",
                    )?,
                    &residual_index.to_string(),
                )?;
                residuals.push(NanoCodecResidualBlockJson {
                    input_activation_alpha: leaf_vector_f32(
                        &subtree(&subtree(&residual_tree, "input_activation")?, "snake")?,
                        "alpha",
                    )?,
                    input_conv: read_causal_conv_1d(&subtree(&residual_tree, "input_conv")?, dilation)?,
                    skip_activation_alpha: leaf_vector_f32(
                        &subtree(&subtree(&residual_tree, "skip_activation")?, "snake")?,
                        "alpha",
                    )?,
                    skip_conv: read_causal_conv_1d(&subtree(&residual_tree, "skip_conv")?, 1)?,
                });
            }
            res_blocks.push(NanoCodecHiFiGanResBlockJson {
                res_blocks: residuals,
            });
        }

        stages.push(NanoCodecUpsampleStageJson {
            activation_alpha,
            upsample_conv,
            res_layer: Some(NanoCodecHiFiGanResLayerJson {
                res_blocks,
            }),
        });

        current_channels = out_channels;
    }

    let post_activation_alpha =
        leaf_vector_f32(&subtree(&subtree(&decoder_tree, "post_activation")?, "snake")?, "alpha")?;
    let post_conv = read_causal_conv_1d(&subtree(&decoder_tree, "post_conv")?, 1)?;

    Ok(NanoCodecDecoderJson {
        pre_conv,
        stages,
        post_activation_alpha,
        post_conv,
        negative_slope: cfg
            .decoder_config
            .activation_config
            .leaky_relu_negative_slope
            .unwrap_or_else(default_negative_slope),
        eps: default_decoder_eps(),
    })
    .and_then(NanoCodecDecoderGraph::try_from)
}

pub(super) fn resolve_descript_audio_codec_vocoder_data_type(
    top_level_precision: Option<ConfigDataType>,
    config: &DescriptAudioCodecConfig,
) -> AudioResult<DataType> {
    let mut resolved_precision: Option<ConfigDataType> = None;
    for (field_name, precision) in [
        ("tts_config.activation_precision", top_level_precision),
        ("tts_config.audio_decoder_config.precision", Some(config.precision)),
        ("tts_config.audio_decoder_config.quantizer_config.precision", config.quantizer_config.precision),
    ] {
        if let Some(precision) = precision {
            if let Some(existing) = resolved_precision {
                if existing != precision {
                    return Err(AudioError::Runtime(format!(
                        "conflicting DescriptAudioCodec precision in Lalamo export: {field_name}={precision:?} conflicts with {existing:?}"
                    )));
                }
            } else {
                resolved_precision = Some(precision);
            }
        }
    }

    let precision = resolved_precision.ok_or(AudioError::Runtime(
        "missing DescriptAudioCodec precision in Lalamo export; expected one of tts_config.activation_precision, \
tts_config.audio_decoder_config.precision, or tts_config.audio_decoder_config.quantizer_config.precision"
            .to_string(),
    ))?;
    let data_type: DataType = precision.into();
    if !matches!(data_type, DataType::F32 | DataType::F16 | DataType::BF16) {
        return Err(AudioError::Runtime(format!(
            "unsupported DescriptAudioCodec vocoder precision in Lalamo export: {precision:?} (expected float32/float16/bfloat16)"
        )));
    }
    Ok(data_type)
}

fn read_vector_quantizer(
    tree: &AudioParameterTree<'_>,
    codebook_size: usize,
    output_dim: usize,
) -> AudioResult<StructuredAudioVectorQuantizer> {
    let codebook_tree = subtree(tree, "codebook")?;
    let (shape, codebook) = leaf_matrix_f32_any(&codebook_tree, "weights")?;
    if shape[0] != codebook_size {
        return Err(AudioError::Runtime(format!("codebook rows mismatch: expected {codebook_size}, got {}", shape[0])));
    }
    let code_dim = shape[1];
    if code_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let out_proj_tree = subtree(tree, "out_proj")?;
    let out_proj = leaf_matrix_f32(&out_proj_tree, "weights", output_dim, code_dim)?;
    let out_bias = leaf_vector_f32(&out_proj_tree, "biases")?;
    if out_bias.len() != output_dim {
        return Err(AudioError::Runtime(format!(
            "out_proj bias shape mismatch: expected {output_dim}, got {}",
            out_bias.len()
        )));
    }

    Ok(StructuredAudioVectorQuantizer {
        codebook,
        codebook_dim: code_dim,
        out_proj,
        out_bias,
    })
}

fn read_conv1d_layer(
    tree: &AudioParameterTree<'_>,
    dilation: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConv1dLayer> {
    let weight = leaf_tensor_3(tree, "weights")?;
    let shape = weight.shape;
    let values = weight.values;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for 'weights', got shape {shape:?}")));
    }
    let bias = leaf_vector_f32(tree, "biases")?;
    if bias.len() != shape[0] {
        return Err(AudioError::Runtime(format!(
            "bias shape mismatch for 'biases': expected {}, got {}",
            shape[0],
            bias.len()
        )));
    }
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if shape[0] % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv weights for 'weights': out_channels {} not divisible by groups {groups}",
            shape[0]
        )));
    }

    Ok(StructuredAudioConv1dLayer {
        weight: values,
        bias,
        cin: shape[1].checked_mul(groups).ok_or(AudioError::Runtime("conv input channel overflow".to_string()))?,
        cout: shape[0],
        kernel_size: shape[2],
        dilation,
        groups,
    })
}

fn read_conv_transpose1d_layer(
    tree: &AudioParameterTree<'_>,
    stride: usize,
    groups: usize,
    weight_layout: TransposeConvWeightLayout,
) -> AudioResult<StructuredAudioConvTranspose1dLayer> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight = leaf_tensor_3(tree, "weights")?;
    let bias = leaf_vector_f32(tree, "biases")?;
    let out_channels = weight.shape[0];
    if bias.len() != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv bias mismatch for 'biases': expected {out_channels}, got {}",
            bias.len()
        )));
    }
    let converted = match weight_layout {
        TransposeConvWeightLayout::Oih => {
            let in_channels = weight.shape[1]
                .checked_mul(groups)
                .ok_or(AudioError::Runtime("transpose conv input channel overflow".to_string()))?;
            convert_lalamo_transpose_weight_oih_to_iog(&weight, in_channels, out_channels, groups)?
        },
    };
    Ok(StructuredAudioConvTranspose1dLayer {
        weight: converted.values,
        bias,
        cin: converted.shape[0],
        cout: out_channels,
        kernel_size: converted.shape[2],
        stride,
        groups,
    })
}

fn read_residual_unit_layer(
    tree: &AudioParameterTree<'_>,
    dilation: usize,
) -> AudioResult<StructuredAudioResidualUnitLayer> {
    Ok(StructuredAudioResidualUnitLayer {
        snake1_alpha: leaf_vector_f32(&subtree(tree, "snake1")?, "alpha")?,
        conv1: read_conv1d_layer(&subtree(tree, "conv1")?, dilation, 1)?,
        snake2_alpha: leaf_vector_f32(&subtree(tree, "snake2")?, "alpha")?,
        conv2: read_conv1d_layer(&subtree(tree, "conv2")?, 1, 1)?,
    })
}

pub(super) fn read_norm_layer(
    tree: &AudioParameterTree<'_>,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<StructuredAudioNormLayer> {
    let scales = leaf_vector_f32(tree, "scales")?;
    let biases = if use_bias {
        Some(leaf_vector_f32(tree, "biases")?)
    } else {
        None
    };
    if let Some(biases) = &biases {
        if biases.len() != scales.len() {
            return Err(AudioError::Runtime(format!(
                "norm scale/bias length mismatch at 'scales': {} vs {}",
                scales.len(),
                biases.len()
            )));
        }
    }
    Ok(StructuredAudioNormLayer {
        scales,
        biases,
        epsilon,
        subtract_mean,
    })
}

fn read_convnext_layer(
    tree: &AudioParameterTree<'_>,
    norm_config: &DescriptAudioConvNeXtNormConfig,
) -> AudioResult<StructuredAudioConvNeXtLayer> {
    let depthwise_tree = subtree(tree, "dwconv")?;
    let depthwise_weight = leaf_tensor_3(&depthwise_tree, "weights")?;
    let depthwise_bias = leaf_vector_f32(&depthwise_tree, "biases")?;
    if depthwise_weight.shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at 'weights', got {}",
            depthwise_weight.shape[1]
        )));
    }
    let depthwise_conv = StructuredAudioConv1dLayer {
        weight: depthwise_weight.values,
        bias: depthwise_bias,
        cin: depthwise_weight.shape[0],
        cout: depthwise_weight.shape[0],
        kernel_size: depthwise_weight.shape[2],
        dilation: 1,
        groups: depthwise_weight.shape[0],
    };
    if depthwise_conv.bias.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise bias mismatch at 'biases': expected {}, got {}",
            depthwise_conv.cout,
            depthwise_conv.bias.len()
        )));
    }
    if depthwise_conv.cout == 0 || depthwise_conv.kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if depthwise_conv.cin != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise conv expects cin==cout, got {} vs {} at 'weights'",
            depthwise_conv.cin,
            depthwise_conv.cout,
        )));
    }

    let norm = read_norm_layer(
        &subtree(tree, "norm")?,
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
    )?;
    if norm.scales.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt norm channels mismatch at 'norm': expected {}, got {}",
            depthwise_conv.cout,
            norm.scales.len()
        )));
    }

    let pwconv1_tree = subtree(tree, "pwconv1")?;
    let ([pwconv1_out, pwconv1_in], pwconv1) = leaf_matrix_f32_any(&pwconv1_tree, "weights")?;
    let pwconv1_bias = leaf_vector_f32(&pwconv1_tree, "biases")?;
    if pwconv1_in != depthwise_conv.cout || pwconv1_out != pwconv1_bias.len() {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 shape mismatch at 'weights': weight [{}, {}], bias {}",
            pwconv1_out,
            pwconv1_in,
            pwconv1_bias.len()
        )));
    }

    let pwconv2_tree = subtree(tree, "pwconv2")?;
    let pwconv2 = leaf_matrix_f32(&pwconv2_tree, "weights", depthwise_conv.cout, pwconv1_out)?;
    let pwconv2_bias = leaf_vector_f32(&pwconv2_tree, "biases")?;
    if pwconv2_bias.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv2 bias mismatch at 'biases': expected {}, got {}",
            depthwise_conv.cout,
            pwconv2_bias.len()
        )));
    }

    Ok(StructuredAudioConvNeXtLayer {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv1_hidden_dim: pwconv1_out,
        pwconv1_bias,
        pwconv2,
        pwconv2_bias,
    })
}

fn build_structured_decoder_graph_from_descript_audio_codec_config(
    root: &AudioParameterTree<'_>,
    cfg: &DescriptAudioCodecConfig,
) -> AudioResult<StructuredAudioDecoderGraph> {
    if cfg.decoder_rates.is_empty() {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.upsampler_config.block_configs.len() != cfg.downsample_factor.len() {
        return Err(AudioError::Runtime(format!(
            "FishAudio upsampler config mismatch: {} block configs for {} downsample factors",
            cfg.quantizer_config.upsampler_config.block_configs.len(),
            cfg.downsample_factor.len()
        )));
    }

    let audio_decoder_tree = subtree(root, "audio_decoder")?;
    let quantizer_tree = subtree(&audio_decoder_tree, "quantizer")?;
    let decoder_tree = subtree(&audio_decoder_tree, "decoder")?;
    let mut upsample_blocks = Vec::with_capacity(cfg.downsample_factor.len());
    for (index, &stride) in cfg.downsample_factor.iter().rev().enumerate() {
        let block_tree = subtree(&subtree(&subtree(&quantizer_tree, "upsampler")?, "blocks")?, &index.to_string())?;
        let trans_conv =
            read_conv_transpose1d_layer(&subtree(&block_tree, "trans_conv")?, stride, 1, TransposeConvWeightLayout::Oih)?;
        let convnext = read_convnext_layer(
            &subtree(&block_tree, "convnext")?,
            &cfg.quantizer_config.upsampler_config.block_configs[index].convnext_config.norm_config,
        )?;
        if convnext.depthwise_conv.cin != trans_conv.cout {
            return Err(AudioError::Runtime(format!(
                "FishAudio upsampler convnext channel mismatch at block {index}: trans_conv out {} vs convnext in {}",
                trans_conv.cout, convnext.depthwise_conv.cin
            )));
        }
        upsample_blocks.push((trans_conv, convnext));
    }

    let first_conv = read_conv1d_layer(&subtree(&decoder_tree, "first_conv")?, 1, 1)?;
    let mut decoder_blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
        let block_tree = subtree(&subtree(&decoder_tree, "decoder_blocks")?, &index.to_string())?;
        decoder_blocks.push(StructuredAudioDecoderBlockLayer {
            snake_alpha: leaf_vector_f32(&subtree(&block_tree, "snake")?, "alpha")?,
            trans_conv: read_conv_transpose1d_layer(
                &subtree(&block_tree, "trans_conv")?,
                stride,
                1,
                TransposeConvWeightLayout::Oih,
            )?,
            res_unit1: read_residual_unit_layer(&subtree(&block_tree, "res_unit1")?, 1)?,
            res_unit2: read_residual_unit_layer(&subtree(&block_tree, "res_unit2")?, 3)?,
            res_unit3: read_residual_unit_layer(&subtree(&block_tree, "res_unit3")?, 9)?,
        });
    }

    let final_snake_alpha = leaf_vector_f32(&subtree(&decoder_tree, "final_snake")?, "alpha")?;
    let final_conv = read_conv1d_layer(&subtree(&decoder_tree, "final_conv")?, 1, 1)?;

    let upsample_factor = cfg
        .downsample_factor
        .iter()
        .chain(cfg.decoder_rates.iter())
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("FishAudio decoder upsample factor overflow".to_string()))?;

    Ok(StructuredAudioDecoderGraph {
        first_conv,
        upsample_blocks,
        decoder_blocks,
        final_snake_alpha,
        final_conv,
        upsample_factor,
    })
}

fn build_structured_codec_graph_from_descript_audio_codec_export(
    tts_config: &TtsConfig,
    model_weights_path: &Path,
) -> AudioResult<StructuredAudioCodecGraph> {
    let TtsAudioDecoderConfig::DescriptAudioCodecConfig {
        config: cfg,
    } = &tts_config.audio_decoder_config
    else {
        return Err(AudioError::Runtime("expected DescriptAudioCodecConfig for structured audio runtime".to_string()));
    };
    let file = File::open(model_weights_path).map_err(|err| {
        AudioError::Runtime(format!("failed to open safetensors file '{}': {err}", model_weights_path.display()))
    })?;
    let host_context = <AudioParameterContext as Context>::new()
        .map_err(|err| AudioError::Runtime(format!("failed to create cpu parameter context: {err}")))?;
    let loader = ParameterLoader::new(&file, host_context.as_ref()).map_err(|err| {
        AudioError::Runtime(format!(
            "failed to parse safetensors metadata from '{}': {err}",
            model_weights_path.display()
        ))
    })?;
    let root = loader.tree();
    let audio_decoder_tree = subtree(&root, "audio_decoder")?;
    let quantizer_tree = subtree(&audio_decoder_tree, "quantizer")?;
    if cfg.n_codebooks == 0 || cfg.codebook_size <= 1 || cfg.semantic_codebook_size <= 1 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.post_module_config.model_dim != cfg.input_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio post_module model_dim mismatch: expected {}, got {}",
            cfg.input_dim, cfg.quantizer_config.post_module_config.model_dim
        )));
    }

    let semantic_quantizer = read_vector_quantizer(
        &subtree(&subtree(&subtree(&quantizer_tree, "semantic_quantizer")?, "quantizers")?, "0")?,
        cfg.semantic_codebook_size,
        cfg.input_dim,
    )?;
    let mut residual_quantizers = Vec::with_capacity(cfg.n_codebooks);
    for index in 0..cfg.n_codebooks {
        residual_quantizers.push(read_vector_quantizer(
            &subtree(&subtree(&subtree(&quantizer_tree, "quantizer")?, "quantizers")?, &index.to_string())?,
            cfg.codebook_size,
            cfg.input_dim,
        )?);
    }
    let decoder = build_structured_decoder_graph_from_descript_audio_codec_config(&root, cfg)?;
    let total_codebooks =
        cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime("FishAudio codebook count overflow".to_string()))?;
    let vocoder_data_type = resolve_descript_audio_codec_vocoder_data_type(tts_config.activation_precision, cfg)?;

    Ok(StructuredAudioCodecGraph {
        semantic_quantizer,
        residual_quantizers,
        post_module_model_dim: cfg.quantizer_config.post_module_config.model_dim,
        post_module_transformer_config: cfg.quantizer_config.post_module_config.clone(),
        weights_path: model_weights_path.display().to_string(),
        decoder,
        codebook_size: cfg.codebook_size,
        semantic_codebook_size: cfg.semantic_codebook_size,
        input_dim: cfg.input_dim,
        total_codebooks,
        upsample_factor: cfg
            .downsample_factor
            .iter()
            .chain(cfg.decoder_rates.iter())
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or(AudioError::Runtime("FishAudio upsample factor overflow".to_string()))?,
        vocoder_data_type,
    })
}

pub(super) fn load_audio_runtime_from_tts_config(
    tts_config: &TtsConfig,
    model_path: &Path,
) -> AudioResult<LoadedTtsAudioRuntimeConfig> {
    match &tts_config.audio_decoder_config {
        TtsAudioDecoderConfig::DescriptAudioCodecConfig {
            config: cfg,
        } => {
            let fishaudio_weights = model_path.join("model.safetensors");
            if !fishaudio_weights.is_file() {
                return Err(AudioError::Runtime(format!(
                    "missing exported FishAudio decoder weights '{}'",
                    fishaudio_weights.display()
                )));
            }
            let total_codebooks = cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime(
                "FishAudio codebook count overflow while building runtime config".to_string(),
            ))?;
            let codebook_size_i32 = i32::try_from(cfg.codebook_size)
                .map_err(|_| AudioError::Runtime("FishAudio codebook_size exceeds i32 kernel range".to_string()))?;
            if codebook_size_i32 <= 1 {
                return Err(AudioError::InvalidTokenCardinality);
            }

            let runtime = RuntimeConfigJson {
                sample_rate: cfg.samplerate,
                num_groups: total_codebooks,
                num_levels_per_group: vec![codebook_size_i32],
                eps: default_eps(),
                output_packing: RuntimePacking::CodebookMajor,
            };
            let decoder =
                build_structured_codec_graph_from_descript_audio_codec_export(tts_config, &fishaudio_weights)?;
            Ok(LoadedTtsAudioRuntimeConfig::StructuredDecoder {
                runtime,
                decoder,
            })
        },
        TtsAudioDecoderConfig::NanoCodecConfig {
            config,
        } => Ok(LoadedTtsAudioRuntimeConfig::Standard(config.clone())),
    }
}
