use super::*;
use crate::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    parameters::{ParameterLoader, ParameterTree},
    utils::array_io::read_array_to_f32_vec,
};

type AudioParameterContext = <Cpu as Backend>::Context;
type AudioParameterTree<'loader> = ParameterTree<'loader, AudioParameterContext>;

fn read_f32_tensor<const RANK: usize>(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<([usize; RANK], Vec<f32>)> {
    let array = tree.leaf_array(name)?;
    let shape = array.shape();
    if shape.len() != RANK {
        return Err(AudioError::Runtime(format!("expected rank-{RANK} tensor for '{name}', got shape {shape:?}")));
    }
    let mut dims = [0usize; RANK];
    dims.copy_from_slice(shape);
    Ok((dims, read_array_to_f32_vec(&array)?))
}

fn read_f32_vector(
    tree: &AudioParameterTree<'_>,
    name: &str,
) -> AudioResult<Vec<f32>> {
    Ok(read_f32_tensor::<1>(tree, name)?.1)
}

fn read_f32_matrix_exact(
    tree: &AudioParameterTree<'_>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = read_f32_tensor::<2>(tree, name)?;
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {shape:?}"
        )));
    }
    Ok(values)
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
    let codebook_tree = tree.subtree("codebook")?;
    let (shape, codebook) = read_f32_tensor::<2>(&codebook_tree, "weights")?;
    if shape[0] != codebook_size {
        return Err(AudioError::Runtime(format!("codebook rows mismatch: expected {codebook_size}, got {}", shape[0])));
    }
    let code_dim = shape[1];
    if code_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let out_proj_tree = tree.subtree("out_proj")?;
    let out_proj = read_f32_matrix_exact(&out_proj_tree, "weights", output_dim, code_dim)?;
    let out_bias = read_f32_vector(&out_proj_tree, "biases")?;
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
    let (shape, values) = read_f32_tensor::<3>(tree, "weights")?;
    let bias = read_f32_vector(tree, "biases")?;
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
) -> AudioResult<StructuredAudioConvTranspose1dLayer> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let (weight_shape, weight_values) = read_f32_tensor::<3>(tree, "weights")?;
    let bias = read_f32_vector(tree, "biases")?;
    if groups == 0 || weight_shape[0] == 0 || weight_shape[1] == 0 || weight_shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let out_channels = weight_shape[1]
        .checked_mul(groups)
        .ok_or(AudioError::Runtime("transpose conv output channel overflow".to_string()))?;
    if bias.len() != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv export layout mismatch for 'weights'/'biases': expected weight layout [in_channels, out_channels/groups, kernel_size] with out_channels={}, got weight shape {:?} and bias len {}",
            out_channels,
            weight_shape,
            bias.len()
        )));
    }
    Ok(StructuredAudioConvTranspose1dLayer {
        weight: weight_values,
        bias,
        cin: weight_shape[0],
        cout: out_channels,
        kernel_size: weight_shape[2],
        stride,
        groups,
    })
}

fn read_residual_unit_layer(
    tree: &AudioParameterTree<'_>,
    dilation: usize,
) -> AudioResult<StructuredAudioResidualUnitLayer> {
    Ok(StructuredAudioResidualUnitLayer {
        snake1_alpha: read_f32_vector(&tree.subtree("snake1")?, "alpha")?,
        conv1: read_conv1d_layer(&tree.subtree("conv1")?, dilation, 1)?,
        snake2_alpha: read_f32_vector(&tree.subtree("snake2")?, "alpha")?,
        conv2: read_conv1d_layer(&tree.subtree("conv2")?, 1, 1)?,
    })
}

pub(super) fn read_norm_layer(
    tree: &AudioParameterTree<'_>,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<StructuredAudioNormLayer> {
    let scales = read_f32_vector(tree, "scales")?;
    let biases = if use_bias {
        Some(read_f32_vector(tree, "biases")?)
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
    let depthwise_tree = tree.subtree("dwconv")?;
    let (depthwise_weight_shape, depthwise_weight_values) = read_f32_tensor::<3>(&depthwise_tree, "weights")?;
    let depthwise_bias = read_f32_vector(&depthwise_tree, "biases")?;
    if depthwise_weight_shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at 'weights', got {}",
            depthwise_weight_shape[1]
        )));
    }
    let depthwise_conv = StructuredAudioConv1dLayer {
        weight: depthwise_weight_values,
        bias: depthwise_bias,
        cin: depthwise_weight_shape[0],
        cout: depthwise_weight_shape[0],
        kernel_size: depthwise_weight_shape[2],
        dilation: 1,
        groups: depthwise_weight_shape[0],
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
            depthwise_conv.cin, depthwise_conv.cout,
        )));
    }

    let norm = read_norm_layer(&tree.subtree("norm")?, norm_config.epsilon, norm_config.subtract_mean, norm_config.use_bias)?;
    if norm.scales.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt norm channels mismatch at 'norm': expected {}, got {}",
            depthwise_conv.cout,
            norm.scales.len()
        )));
    }

    let pwconv1_tree = tree.subtree("pwconv1")?;
    let ([pwconv1_out, pwconv1_in], pwconv1) = read_f32_tensor::<2>(&pwconv1_tree, "weights")?;
    let pwconv1_bias = read_f32_vector(&pwconv1_tree, "biases")?;
    if pwconv1_in != depthwise_conv.cout || pwconv1_out != pwconv1_bias.len() {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 shape mismatch at 'weights': weight [{}, {}], bias {}",
            pwconv1_out,
            pwconv1_in,
            pwconv1_bias.len()
        )));
    }

    let pwconv2_tree = tree.subtree("pwconv2")?;
    let pwconv2 = read_f32_matrix_exact(&pwconv2_tree, "weights", depthwise_conv.cout, pwconv1_out)?;
    let pwconv2_bias = read_f32_vector(&pwconv2_tree, "biases")?;
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

    let audio_decoder_tree = root.subtree("audio_decoder")?;
    let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
    let decoder_tree = audio_decoder_tree.subtree("decoder")?;
    let mut upsample_blocks = Vec::with_capacity(cfg.downsample_factor.len());
    for (index, &stride) in cfg.downsample_factor.iter().rev().enumerate() {
        let block_tree = quantizer_tree.subtree("upsampler")?.subtree("blocks")?.subtree(&index.to_string())?;
        let trans_conv = read_conv_transpose1d_layer(&block_tree.subtree("trans_conv")?, stride, 1)?;
        let convnext = read_convnext_layer(
            &block_tree.subtree("convnext")?,
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

    let first_conv = read_conv1d_layer(&decoder_tree.subtree("first_conv")?, 1, 1)?;
    let mut decoder_blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
        let block_tree = decoder_tree.subtree("decoder_blocks")?.subtree(&index.to_string())?;
        decoder_blocks.push(StructuredAudioDecoderBlockLayer {
            snake_alpha: read_f32_vector(&block_tree.subtree("snake")?, "alpha")?,
            trans_conv: read_conv_transpose1d_layer(&block_tree.subtree("trans_conv")?, stride, 1)?,
            res_unit1: read_residual_unit_layer(&block_tree.subtree("res_unit1")?, 1)?,
            res_unit2: read_residual_unit_layer(&block_tree.subtree("res_unit2")?, 3)?,
            res_unit3: read_residual_unit_layer(&block_tree.subtree("res_unit3")?, 9)?,
        });
    }

    let final_snake_alpha = read_f32_vector(&decoder_tree.subtree("final_snake")?, "alpha")?;
    let final_conv = read_conv1d_layer(&decoder_tree.subtree("final_conv")?, 1, 1)?;

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
    let cfg = match &tts_config.audio_decoder_config {
        TtsAudioDecoderConfig::DescriptAudioCodecConfig {
            config,
        } => config,
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
    let audio_decoder_tree = root.subtree("audio_decoder")?;
    let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
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
        &quantizer_tree.subtree("semantic_quantizer")?.subtree("quantizers")?.subtree("0")?,
        cfg.semantic_codebook_size,
        cfg.input_dim,
    )?;
    let mut residual_quantizers = Vec::with_capacity(cfg.n_codebooks);
    for index in 0..cfg.n_codebooks {
        residual_quantizers.push(read_vector_quantizer(
            &quantizer_tree.subtree("quantizer")?.subtree("quantizers")?.subtree(&index.to_string())?,
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
) -> AudioResult<(RuntimeConfigJson, StructuredAudioCodecGraph)> {
    let cfg = match &tts_config.audio_decoder_config {
        TtsAudioDecoderConfig::DescriptAudioCodecConfig {
            config,
        } => config,
    };
    let fishaudio_weights = model_path.join("model.safetensors");
    if !fishaudio_weights.is_file() {
        return Err(AudioError::Runtime(format!(
            "missing exported FishAudio decoder weights '{}'",
            fishaudio_weights.display()
        )));
    }
    let total_codebooks = cfg
        .n_codebooks
        .checked_add(1)
        .ok_or(AudioError::Runtime("FishAudio codebook count overflow while building runtime config".to_string()))?;
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
    let decoder = build_structured_codec_graph_from_descript_audio_codec_export(tts_config, &fishaudio_weights)?;
    Ok((runtime, decoder))
}
