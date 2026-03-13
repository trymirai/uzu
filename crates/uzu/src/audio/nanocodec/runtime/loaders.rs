use super::*;

#[derive(Debug)]
pub(super) struct SafeTensorReader {
    file: File,
    entries: HashMap<String, SafeTensorEntry>,
}

impl SafeTensorReader {
    pub(super) fn open(path: &Path) -> AudioResult<Self> {
        let file = File::open(path).map_err(|err| {
            AudioError::Runtime(format!("failed to open safetensors file '{}': {err}", path.display()))
        })?;
        let (global_offset, metadata) = read_safetensors_metadata(&file).map_err(|err| {
            AudioError::Runtime(format!("failed to parse safetensors metadata from '{}': {err}", path.display()))
        })?;

        let mut entries = HashMap::new();
        for (name, tensor) in metadata.tensors {
            let (local_begin, local_end) = tensor.data_offsets;
            let size = local_end
                .checked_sub(local_begin)
                .ok_or(AudioError::Runtime("invalid tensor data offsets in safetensors metadata".to_string()))?;
            let offset = global_offset
                .checked_add(local_begin)
                .ok_or(AudioError::Runtime("safetensors tensor offset overflow".to_string()))?;

            entries.insert(
                name,
                SafeTensorEntry {
                    data_type: tensor.dtype.into(),
                    shape: tensor.shape,
                    offset,
                    size,
                },
            );
        }

        Ok(Self {
            file,
            entries,
        })
    }

    fn read_tensor_f32(
        &self,
        key: &str,
    ) -> AudioResult<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| AudioError::Runtime(format!("missing tensor '{key}' in model.safetensors")))?;

        let num_elements = checked_product(&entry.shape)?;
        let expected_size = num_elements
            .checked_mul(entry.data_type.size_in_bytes())
            .ok_or(AudioError::Runtime(format!("tensor '{key}' byte-size overflow")))?;
        if entry.size != expected_size {
            return Err(AudioError::Runtime(format!(
                "tensor '{key}' size mismatch: expected {expected_size} bytes from shape {:?} and dtype {:?}, got {}",
                entry.shape, entry.data_type, entry.size
            )));
        }

        let mut bytes = vec![0_u8; entry.size];
        self.file.read_exact_at(&mut bytes, entry.offset as u64).map_err(|err| {
            AudioError::Runtime(format!("failed reading tensor '{key}' from model.safetensors: {err}"))
        })?;

        let values = match entry.data_type {
            DataType::F32 => decode_f32_bytes(&bytes),
            DataType::F16 => decode_f16_bytes(&bytes).into_iter().map(f16::to_f32).collect::<Vec<f32>>(),
            DataType::BF16 => decode_bf16_bytes(&bytes).into_iter().map(bf16::to_f32).collect::<Vec<f32>>(),
            other => {
                return Err(AudioError::Runtime(format!(
                    "unsupported tensor dtype for '{key}': {other:?} (expected F32/F16/BF16)"
                )));
            },
        };

        if values.len() != num_elements {
            return Err(AudioError::Runtime(format!(
                "decoded tensor '{key}' element count mismatch: expected {num_elements}, got {}",
                values.len()
            )));
        }

        Ok((entry.shape.clone(), values))
    }
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn decode_f16_bytes(bytes: &[u8]) -> Vec<f16> {
    bytes.chunks_exact(2).map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn decode_bf16_bytes(bytes: &[u8]) -> Vec<bf16> {
    bytes.chunks_exact(2).map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn read_tensor_1d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 1 {
        return Err(AudioError::Runtime(format!("expected rank-1 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(values)
}

fn read_tensor_3d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Tensor3Json> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(Tensor3Json {
        shape: [shape[0], shape[1], shape[2]],
        values,
    })
}

fn read_causal_conv_1d(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<CausalConv1dJson> {
    Ok(CausalConv1dJson {
        weight: read_tensor_3d(reader, &format!("{prefix}.weights"))?,
        bias: read_tensor_1d(reader, &format!("{prefix}.biases"))?,
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
    let reader = SafeTensorReader::open(model_weights_path)?;

    let pre_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.pre_conv", 1)?;
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

        let activation_alpha =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.activations.{stage_index}.snake.alpha"))?;

        let upsample_weight_oih =
            read_tensor_3d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.weights"))?;
        let upsample_bias =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.biases"))?;
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
                let prefix = format!(
                    "audio_decoder.decoder.res_layers.{stage_index}.res_blocks.{res_block_index}.res_blocks.{residual_index}"
                );
                residuals.push(NanoCodecResidualBlockJson {
                    input_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.input_activation.snake.alpha"))?,
                    input_conv: read_causal_conv_1d(&reader, &format!("{prefix}.input_conv"), dilation)?,
                    skip_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.skip_activation.snake.alpha"))?,
                    skip_conv: read_causal_conv_1d(&reader, &format!("{prefix}.skip_conv"), 1)?,
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

    let post_activation_alpha = read_tensor_1d(&reader, "audio_decoder.decoder.post_activation.snake.alpha")?;
    let post_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.post_conv", 1)?;

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

pub(super) fn resolve_fishaudio_vocoder_data_type(
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
                        "conflicting FishAudio precision in Lalamo export: {field_name}={precision:?} conflicts with {existing:?}"
                    )));
                }
            } else {
                resolved_precision = Some(precision);
            }
        }
    }

    let precision = resolved_precision.ok_or(AudioError::Runtime(
        "missing FishAudio precision in Lalamo export; expected one of tts_config.activation_precision, \
tts_config.audio_decoder_config.precision, or tts_config.audio_decoder_config.quantizer_config.precision"
            .to_string(),
    ))?;
    let data_type: DataType = precision.into();
    if !matches!(data_type, DataType::F32 | DataType::F16 | DataType::BF16) {
        return Err(AudioError::Runtime(format!(
            "unsupported FishAudio vocoder precision in Lalamo export: {precision:?} (expected float32/float16/bfloat16)"
        )));
    }
    Ok(data_type)
}

pub(super) fn read_matrix_f32(
    reader: &SafeTensorReader,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(AudioError::Runtime(format!(
            "tensor '{key}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {shape:?}"
        )));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_matrix_f32_any(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_fishaudio_vector_quantizer(
    reader: &SafeTensorReader,
    prefix: &str,
    codebook_size: usize,
    output_dim: usize,
) -> AudioResult<StructuredAudioVectorQuantizer> {
    let codebook_key = format!("{prefix}.codebook.weights");
    let (shape, values) = reader.read_tensor_f32(&codebook_key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{codebook_key}', got shape {shape:?}")));
    }
    if shape[0] != codebook_size {
        return Err(AudioError::Runtime(format!(
            "codebook rows mismatch for '{codebook_key}': expected {codebook_size}, got {}",
            shape[0]
        )));
    }
    let code_dim = shape[1];
    if code_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let out_proj = read_matrix_f32(reader, &format!("{prefix}.out_proj.weights"), output_dim, code_dim)?;
    let out_bias = read_tensor_1d(reader, &format!("{prefix}.out_proj.biases"))?;
    if out_bias.len() != output_dim {
        return Err(AudioError::Runtime(format!(
            "out_proj bias shape mismatch for '{prefix}': expected {output_dim}, got {}",
            out_bias.len()
        )));
    }

    Ok(StructuredAudioVectorQuantizer {
        codebook: MatrixF32 {
            rows: shape[0],
            cols: shape[1],
            values,
        },
        out_proj,
        out_bias,
    })
}

fn read_fishaudio_conv1d_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConv1dLayer> {
    let weight_key = format!("{prefix}.weights");
    let bias_key = format!("{prefix}.biases");
    let (shape, values) = reader.read_tensor_f32(&weight_key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{weight_key}', got shape {shape:?}")));
    }
    let bias = read_tensor_1d(reader, &bias_key)?;
    if bias.len() != shape[0] {
        return Err(AudioError::Runtime(format!(
            "bias shape mismatch for '{bias_key}': expected {}, got {}",
            shape[0],
            bias.len()
        )));
    }
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if shape[0] % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv weights for '{weight_key}': out_channels {} not divisible by groups {groups}",
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

fn read_fishaudio_transpose_conv_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    stride: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConvTranspose1dLayer> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight_oih = read_tensor_3d(reader, &format!("{prefix}.weights"))?;
    let bias = read_tensor_1d(reader, &format!("{prefix}.biases"))?;
    let out_channels = weight_oih.shape[0];
    if bias.len() != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv bias mismatch for '{prefix}.biases': expected {out_channels}, got {}",
            bias.len()
        )));
    }
    let in_channels = weight_oih.shape[1]
        .checked_mul(groups)
        .ok_or(AudioError::Runtime("transpose conv input channel overflow".to_string()))?;
    let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, in_channels, out_channels, groups)?;
    Ok(StructuredAudioConvTranspose1dLayer {
        weight: converted.values,
        bias,
        cin: in_channels,
        cout: out_channels,
        kernel_size: converted.shape[2],
        stride,
        groups,
    })
}

fn read_fishaudio_residual_unit_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<StructuredAudioResidualUnitLayer> {
    Ok(StructuredAudioResidualUnitLayer {
        snake1_alpha: read_tensor_1d(reader, &format!("{prefix}.snake1.alpha"))?,
        conv1: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv1"), dilation, 1)?,
        snake2_alpha: read_tensor_1d(reader, &format!("{prefix}.snake2.alpha"))?,
        conv2: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv2"), 1, 1)?,
    })
}

pub(super) fn read_fishaudio_norm_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<StructuredAudioNormLayer> {
    let scales = read_tensor_1d(reader, &format!("{prefix}.scales"))?;
    let biases = if use_bias {
        Some(read_tensor_1d(reader, &format!("{prefix}.biases"))?)
    } else {
        None
    };
    if let Some(biases) = &biases {
        if biases.len() != scales.len() {
            return Err(AudioError::Runtime(format!(
                "norm scale/bias length mismatch at '{prefix}': {} vs {}",
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

fn read_fishaudio_convnext_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    norm_config: &DescriptAudioConvNeXtNormConfig,
) -> AudioResult<StructuredAudioConvNeXtLayer> {
    let depthwise_weight = read_tensor_3d(reader, &format!("{prefix}.dwconv.weights"))?;
    let depthwise_bias = read_tensor_1d(reader, &format!("{prefix}.dwconv.biases"))?;
    if depthwise_weight.shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at {prefix}, got {}",
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
            "ConvNeXt depthwise bias mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            depthwise_conv.bias.len()
        )));
    }
    if depthwise_conv.cout == 0 || depthwise_conv.kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if depthwise_conv.cin != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise conv expects cin==cout, got {} vs {} at {prefix}",
            depthwise_conv.cin, depthwise_conv.cout
        )));
    }

    let norm = read_fishaudio_norm_layer(
        reader,
        &format!("{prefix}.norm"),
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
    )?;
    if norm.scales.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt norm channels mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            norm.scales.len()
        )));
    }

    let pwconv1 = read_matrix_f32_any(reader, &format!("{prefix}.pwconv1.weights"))?;
    let pwconv1_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv1.biases"))?;
    if pwconv1.cols != depthwise_conv.cout || pwconv1.rows != pwconv1_bias.len() {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 shape mismatch at {prefix}: weight [{}, {}], bias {}",
            pwconv1.rows,
            pwconv1.cols,
            pwconv1_bias.len()
        )));
    }

    let pwconv2 = read_matrix_f32(reader, &format!("{prefix}.pwconv2.weights"), depthwise_conv.cout, pwconv1.rows)?;
    let pwconv2_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv2.biases"))?;
    if pwconv2_bias.len() != pwconv2.rows {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv2 bias mismatch at {prefix}: expected {}, got {}",
            pwconv2.rows,
            pwconv2_bias.len()
        )));
    }

    Ok(StructuredAudioConvNeXtLayer {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv1_bias,
        pwconv2,
        pwconv2_bias,
    })
}

fn build_structured_decoder_graph_from_fishaudio_config(
    reader: &SafeTensorReader,
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

    let mut upsample_blocks = Vec::with_capacity(cfg.downsample_factor.len());
    for (index, &stride) in cfg.downsample_factor.iter().rev().enumerate() {
        let trans_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.trans_conv");
        let convnext_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.convnext");
        let trans_conv = read_fishaudio_transpose_conv_layer(reader, &trans_prefix, stride, 1)?;
        let convnext = read_fishaudio_convnext_layer(
            reader,
            &convnext_prefix,
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

    let first_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.first_conv", 1, 1)?;
    let mut decoder_blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
        let base = format!("audio_decoder.decoder.decoder_blocks.{index}");
        decoder_blocks.push(StructuredAudioDecoderBlockLayer {
            snake_alpha: read_tensor_1d(reader, &format!("{base}.snake.alpha"))?,
            trans_conv: read_fishaudio_transpose_conv_layer(reader, &format!("{base}.trans_conv"), stride, 1)?,
            res_unit1: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit1"), 1)?,
            res_unit2: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit2"), 3)?,
            res_unit3: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit3"), 9)?,
        });
    }

    let final_snake_alpha = read_tensor_1d(reader, "audio_decoder.decoder.final_snake.alpha")?;
    let final_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.final_conv", 1, 1)?;

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

fn build_structured_codec_graph_from_fishaudio_export(
    tts_config: &TtsConfig,
    model_weights_path: &Path,
) -> AudioResult<StructuredAudioCodecGraph> {
    let TtsAudioDecoderConfig::DescriptAudioCodecConfig {
        config: cfg,
    } = &tts_config.audio_decoder_config
    else {
        return Err(AudioError::Runtime("expected DescriptAudioCodecConfig for structured audio runtime".to_string()));
    };
    let reader = SafeTensorReader::open(model_weights_path)?;
    if cfg.n_codebooks == 0 || cfg.codebook_size <= 1 || cfg.semantic_codebook_size <= 1 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.post_module_config.model_dim != cfg.input_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio post_module model_dim mismatch: expected {}, got {}",
            cfg.input_dim, cfg.quantizer_config.post_module_config.model_dim
        )));
    }

    let semantic_quantizer = read_fishaudio_vector_quantizer(
        &reader,
        "audio_decoder.quantizer.semantic_quantizer.quantizers.0",
        cfg.semantic_codebook_size,
        cfg.input_dim,
    )?;
    let mut residual_quantizers = Vec::with_capacity(cfg.n_codebooks);
    for index in 0..cfg.n_codebooks {
        let prefix = format!("audio_decoder.quantizer.quantizer.quantizers.{index}");
        residual_quantizers.push(read_fishaudio_vector_quantizer(&reader, &prefix, cfg.codebook_size, cfg.input_dim)?);
    }
    let decoder = build_structured_decoder_graph_from_fishaudio_config(&reader, cfg)?;
    let total_codebooks =
        cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime("FishAudio codebook count overflow".to_string()))?;
    let vocoder_data_type = resolve_fishaudio_vocoder_data_type(tts_config.activation_precision, cfg)?;

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
                r#type: Some("nanocodec_fsq".to_string()),
                sample_rate: cfg.samplerate,
                num_groups: total_codebooks,
                num_levels_per_group: vec![codebook_size_i32],
                eps: default_eps(),
                output_packing: RuntimePacking::CodebookMajor,
                decoder: None,
            };
            let decoder = build_structured_codec_graph_from_fishaudio_export(tts_config, &fishaudio_weights)?;
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
