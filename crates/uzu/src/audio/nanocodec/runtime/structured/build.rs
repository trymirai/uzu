use super::*;

pub(super) type BackendParameterTree<'loader, B> = crate::parameters::ParameterTree<'loader, <B as Backend>::Context>;

pub(super) fn read_float_allocation<B: Backend, const RANK: usize>(
    tree: &BackendParameterTree<'_, B>,
    name: &str,
    expected_data_type: DataType,
) -> AudioResult<([usize; RANK], crate::backends::common::Allocation<B>)> {
    let leaf = tree.leaf(name)?;
    if leaf.data_type() != expected_data_type {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' dtype mismatch: expected {expected_data_type:?}, got {:?}",
            leaf.data_type()
        )));
    }
    if leaf.shape().len() != RANK {
        return Err(AudioError::Runtime(format!(
            "expected rank-{RANK} tensor for '{name}', got rank {}",
            leaf.shape().len()
        )));
    }
    let mut dims = [0usize; RANK];
    dims.copy_from_slice(leaf.shape());
    Ok((dims, leaf.read_allocation()?))
}

pub(super) fn read_float_vector_exact<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    name: &str,
    expected_len: usize,
    expected_data_type: DataType,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let (shape, allocation) = read_float_allocation::<B, 1>(tree, name, expected_data_type)?;
    if shape[0] != expected_len {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_len}], got {:?}",
            shape
        )));
    }
    Ok(allocation)
}

pub(super) fn read_float_matrix_exact<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
    expected_data_type: DataType,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let (shape, allocation) = read_float_allocation::<B, 2>(tree, name, expected_data_type)?;
    if shape != [expected_rows, expected_cols] {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {:?}",
            shape
        )));
    }
    Ok(allocation)
}

pub(super) fn outer_axis_view<B: Backend>(
    allocation: &crate::backends::common::Allocation<B>,
    index: usize,
    slice_shape: &[usize],
    data_type: DataType,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let slice_bytes = size_for_shape(slice_shape, data_type);
    let offset = index
        .checked_mul(slice_bytes)
        .ok_or(AudioError::Runtime("allocation outer-axis view offset overflow".to_string()))?;
    Ok(allocation.slice(offset..offset + slice_bytes))
}

pub(super) fn read_conv1d_layer<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    dilation: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConv1d<B>> {
    let (shape, weight) = read_float_allocation::<B, 3>(tree, "weights", data_type)?;
    let bias = read_float_vector_exact::<B>(tree, "biases", shape[0], data_type)?;
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if shape[0] % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv weights for 'weights': out_channels {} not divisible by groups {groups}",
            shape[0]
        )));
    }
    Ok(StructuredAudioConv1d {
        weight,
        bias,
        cin: shape[1].checked_mul(groups).ok_or(AudioError::Runtime("conv input channel overflow".to_string()))?,
        cout: shape[0],
        kernel_size: shape[2],
        dilation,
        groups,
    })
}

pub(super) fn read_conv_transpose1d_layer<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    stride: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConvTranspose1d<B>> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let (shape, weight) = read_float_allocation::<B, 3>(tree, "weights", data_type)?;
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let export_cout = shape[0];
    let export_cin =
        shape[1].checked_mul(groups).ok_or(AudioError::Runtime("transpose conv input channel overflow".to_string()))?;
    if export_cout % groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let bias = read_float_vector_exact::<B>(tree, "biases", export_cout, data_type)?;
    let expected_weight_shape = [export_cout, export_cin / groups, shape[2]];
    if shape != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "transpose conv export-layout weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape, shape
        )));
    }

    Ok(StructuredAudioConvTranspose1d {
        weight,
        bias,
        cin: export_cin,
        cout: export_cout,
        kernel_size: shape[2],
        stride,
        groups,
    })
}

pub(super) fn read_pointwise_conv_layer<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    expected_out_dim: usize,
    expected_in_dim: usize,
) -> AudioResult<StructuredAudioPointwiseConv<B>> {
    let weight = read_float_matrix_exact::<B>(tree, "weights", expected_out_dim, expected_in_dim, data_type)?;
    let bias = read_float_vector_exact::<B>(tree, "biases", expected_out_dim, data_type)?;
    Ok(StructuredAudioPointwiseConv {
        weight,
        bias,
        cin: expected_in_dim,
        cout: expected_out_dim,
    })
}

pub(super) fn read_norm_layer<B: Backend>(
    context: &Rc<B::Context>,
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    channels: usize,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
    _label_prefix: &str,
) -> AudioResult<StructuredAudioNorm<B>> {
    let scales = read_float_vector_exact::<B>(tree, "scales", channels, data_type)?;
    let bias = if use_bias {
        read_float_vector_exact::<B>(tree, "biases", channels, data_type)?
    } else {
        crate::backends::common::allocation_helpers::create_zeroed_allocation(context.as_ref(), &[channels], data_type)
    };
    Ok(StructuredAudioNorm {
        scales,
        bias,
        epsilon,
        subtract_mean,
    })
}

pub(super) fn read_convnext_layer<B: Backend>(
    context: &Rc<B::Context>,
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    norm_config: &DescriptAudioConvNeXtNormConfig,
) -> AudioResult<StructuredAudioConvNeXt<B>> {
    let depthwise_tree = tree.subtree("dwconv")?;
    let (depthwise_shape, depthwise_weight) = read_float_allocation::<B, 3>(&depthwise_tree, "weights", data_type)?;
    if depthwise_shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at 'dwconv.weights', got {}",
            depthwise_shape[1]
        )));
    }
    let depthwise_conv = StructuredAudioConv1d {
        weight: depthwise_weight,
        bias: read_float_vector_exact::<B>(&depthwise_tree, "biases", depthwise_shape[0], data_type)?,
        cin: depthwise_shape[0],
        cout: depthwise_shape[0],
        kernel_size: depthwise_shape[2],
        dilation: 1,
        groups: depthwise_shape[0],
    };
    if depthwise_conv.cout == 0 || depthwise_conv.kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if depthwise_conv.cin != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise conv expects cin==cout, got {} vs {} at 'dwconv.weights'",
            depthwise_conv.cin, depthwise_conv.cout
        )));
    }
    let channels = depthwise_conv.cout;
    let norm = read_norm_layer::<B>(
        context,
        &tree.subtree("norm")?,
        data_type,
        channels,
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
        "structured_audio_convnext_norm",
    )?;
    let (pwconv1_shape, _) = read_float_allocation::<B, 2>(&tree.subtree("pwconv1")?, "weights", data_type)?;
    if pwconv1_shape[1] != channels {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 input mismatch at 'pwconv1.weights': expected {}, got {}",
            channels, pwconv1_shape[1]
        )));
    }
    let pwconv1 = read_pointwise_conv_layer::<B>(&tree.subtree("pwconv1")?, data_type, pwconv1_shape[0], channels)?;
    let pwconv2 = read_pointwise_conv_layer::<B>(&tree.subtree("pwconv2")?, data_type, channels, pwconv1_shape[0])?;
    Ok(StructuredAudioConvNeXt {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv2,
    })
}

pub(super) fn read_residual_unit_layer<B: Backend>(
    tree: &BackendParameterTree<'_, B>,
    data_type: DataType,
    dilation: usize,
) -> AudioResult<StructuredAudioResidualUnit<B>> {
    let snake1_alpha = {
        let snake1_tree = tree.subtree("snake1")?;
        let (shape, alpha) = read_float_allocation::<B, 1>(&snake1_tree, "alpha", data_type)?;
        if shape[0] == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        alpha
    };
    let conv1 = read_conv1d_layer::<B>(&tree.subtree("conv1")?, data_type, dilation, 1)?;
    let snake2_alpha = {
        let snake2_tree = tree.subtree("snake2")?;
        let (shape, alpha) = read_float_allocation::<B, 1>(&snake2_tree, "alpha", data_type)?;
        if shape[0] != conv1.cout {
            return Err(AudioError::Runtime(format!(
                "residual snake2 alpha mismatch: expected {}, got {}",
                conv1.cout, shape[0]
            )));
        }
        alpha
    };
    let conv2 = read_conv1d_layer::<B>(&tree.subtree("conv2")?, data_type, 1, 1)?;
    Ok(StructuredAudioResidualUnit {
        snake1_alpha,
        conv1,
        snake2_alpha,
        conv2,
    })
}

pub(super) fn build_vocoder_graph_from_tree<B: Backend>(
    context: &Rc<B::Context>,
    root: &BackendParameterTree<'_, B>,
    config: &DescriptAudioCodecConfig,
    data_type: DataType,
) -> AudioResult<StructuredAudioDecoderGraph<B>> {
    let audio_decoder_tree = root.subtree("audio_decoder")?;
    let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
    let decoder_tree = audio_decoder_tree.subtree("decoder")?;

    let first_conv = read_conv1d_layer::<B>(&decoder_tree.subtree("first_conv")?, data_type, 1, 1)?;
    let final_conv = read_conv1d_layer::<B>(&decoder_tree.subtree("final_conv")?, data_type, 1, 1)?;
    let final_snake_alpha =
        read_float_vector_exact::<B>(&decoder_tree.subtree("final_snake")?, "alpha", final_conv.cin, data_type)?;

    let mut upsample_blocks = Vec::with_capacity(config.downsample_factor.len());
    for (index, &stride) in config.downsample_factor.iter().rev().enumerate() {
        let block_tree = quantizer_tree.subtree("upsampler")?.subtree("blocks")?.subtree(&index.to_string())?;
        let trans_conv = read_conv_transpose1d_layer::<B>(&block_tree.subtree("trans_conv")?, data_type, stride, 1)?;
        let convnext = read_convnext_layer::<B>(
            context,
            &block_tree.subtree("convnext")?,
            data_type,
            &config.quantizer_config.upsampler_config.block_configs[index].convnext_config.norm_config,
        )?;
        if convnext.depthwise_conv.cin != trans_conv.cout {
            return Err(AudioError::Runtime(format!(
                "structured audio upsampler convnext channel mismatch at block {index}: trans_conv out {} vs convnext in {}",
                trans_conv.cout, convnext.depthwise_conv.cin
            )));
        }
        upsample_blocks.push((trans_conv, convnext));
    }

    let mut decoder_blocks = Vec::with_capacity(config.decoder_rates.len());
    for (index, &stride) in config.decoder_rates.iter().enumerate() {
        let block_tree = decoder_tree.subtree("decoder_blocks")?.subtree(&index.to_string())?;
        let trans_conv = read_conv_transpose1d_layer::<B>(&block_tree.subtree("trans_conv")?, data_type, stride, 1)?;
        let snake_alpha =
            read_float_vector_exact::<B>(&block_tree.subtree("snake")?, "alpha", trans_conv.cin, data_type)?;
        let channels = trans_conv.cout;
        let res_unit1 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit1")?, data_type, 1)?;
        let res_unit2 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit2")?, data_type, 3)?;
        let res_unit3 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit3")?, data_type, 9)?;
        if res_unit1.conv1.cin != channels || res_unit2.conv1.cin != channels || res_unit3.conv1.cin != channels {
            return Err(AudioError::Runtime(format!(
                "structured audio decoder block {index} channel mismatch in exported weights"
            )));
        }
        decoder_blocks.push(StructuredAudioDecoderBlock {
            snake_alpha,
            trans_conv,
            res_unit1,
            res_unit2,
            res_unit3,
        });
    }

    Ok(StructuredAudioDecoderGraph {
        first_conv,
        upsample_blocks,
        decoder_blocks,
        final_snake_alpha,
        final_conv,
    })
}
