type MetalParameterTree<'loader> = crate::parameters::ParameterTree<'loader, <Metal as Backend>::Context>;

fn read_float_array<const RANK: usize>(
    tree: &MetalParameterTree<'_>,
    name: &str,
    expected_data_type: DataType,
) -> AudioResult<([usize; RANK], Array<Metal>)> {
    let array = tree.leaf_array(name)?;
    if array.data_type() != expected_data_type {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' dtype mismatch: expected {expected_data_type:?}, got {:?}",
            array.data_type()
        )));
    }
    if array.shape().len() != RANK {
        return Err(AudioError::Runtime(format!(
            "expected rank-{RANK} tensor for '{name}', got shape {:?}",
            array.shape()
        )));
    }
    let mut dims = [0usize; RANK];
    dims.copy_from_slice(array.shape());
    Ok((dims, array))
}

fn read_float_vector_exact(
    tree: &MetalParameterTree<'_>,
    name: &str,
    expected_len: usize,
    expected_data_type: DataType,
) -> AudioResult<Array<Metal>> {
    let (shape, array) = read_float_array::<1>(tree, name, expected_data_type)?;
    if shape[0] != expected_len {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_len}], got {:?}",
            shape
        )));
    }
    Ok(array)
}

fn read_float_matrix_exact(
    tree: &MetalParameterTree<'_>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
    expected_data_type: DataType,
) -> AudioResult<Array<Metal>> {
    let (shape, array) = read_float_array::<2>(tree, name, expected_data_type)?;
    if shape != [expected_rows, expected_cols] {
        return Err(AudioError::Runtime(format!(
            "tensor '{name}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {:?}",
            shape
        )));
    }
    Ok(array)
}

fn outer_axis_view(
    array: &Array<Metal>,
    index: usize,
    slice_shape: &[usize],
) -> AudioResult<Array<Metal>> {
    let slice_bytes = size_for_shape(slice_shape, array.data_type());
    let offset = index
        .checked_mul(slice_bytes)
        .and_then(|value| value.checked_add(array.offset()))
        .ok_or(AudioError::Runtime("array outer-axis view offset overflow".to_string()))?;
    Ok(unsafe { Array::from_parts(array.buffer(), offset, slice_shape, array.data_type()) })
}

fn read_conv1d_gpu_layer(
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    dilation: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConv1d> {
    let (shape, weight) = read_float_array::<3>(tree, "weights", data_type)?;
    let bias = read_float_vector_exact(tree, "biases", shape[0], data_type)?;
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

fn read_conv_transpose1d_gpu_layer(
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    stride: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConvTranspose1d> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let (shape, weight) = read_float_array::<3>(tree, "weights", data_type)?;
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let out_channels = shape[1]
        .checked_mul(groups)
        .ok_or(AudioError::Runtime("transpose conv output channel overflow".to_string()))?;
    let bias = read_float_vector_exact(tree, "biases", out_channels, data_type)?;
    Ok(StructuredAudioConvTranspose1d {
        weight,
        bias,
        cin: shape[0],
        cout: out_channels,
        kernel_size: shape[2],
        stride,
        groups,
    })
}

fn read_pointwise_conv_gpu_layer(
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    expected_out_dim: usize,
    expected_in_dim: usize,
) -> AudioResult<StructuredAudioPointwiseConv> {
    let weight = read_float_matrix_exact(tree, "weights", expected_out_dim, expected_in_dim, data_type)?;
    let bias = read_float_vector_exact(tree, "biases", expected_out_dim, data_type)?;
    Ok(StructuredAudioPointwiseConv {
        weight: weight.view(&[expected_out_dim, expected_in_dim, 1]),
        bias,
        cin: expected_in_dim,
        cout: expected_out_dim,
    })
}

fn read_norm_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    channels: usize,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
    label_prefix: &str,
) -> AudioResult<StructuredAudioNorm> {
    let scales = read_float_vector_exact(tree, "scales", channels, data_type)?;
    let bias = if use_bias {
        read_float_vector_exact(tree, "biases", channels, data_type)?
    } else {
        context.create_array(&[channels], data_type, &format!("{label_prefix}_bias"))
    };
    Ok(StructuredAudioNorm {
        scales,
        bias,
        epsilon,
        subtract_mean,
    })
}

fn read_convnext_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    norm_config: &DescriptAudioConvNeXtNormConfig,
) -> AudioResult<StructuredAudioConvNeXt> {
    let depthwise_tree = tree.subtree("dwconv")?;
    let (depthwise_shape, depthwise_weight) = read_float_array::<3>(&depthwise_tree, "weights", data_type)?;
    if depthwise_shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at 'dwconv.weights', got {}",
            depthwise_shape[1]
        )));
    }
    let depthwise_conv = StructuredAudioConv1d {
        weight: depthwise_weight,
        bias: read_float_vector_exact(&depthwise_tree, "biases", depthwise_shape[0], data_type)?,
        cin: depthwise_shape[0],
        cout: depthwise_shape[0],
        kernel_size: depthwise_shape[2],
        dilation: 1,
        groups: depthwise_shape[0],
    };
    if depthwise_conv.bias.shape()[0] != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise bias mismatch at 'dwconv.biases': expected {}, got {}",
            depthwise_conv.cout,
            depthwise_conv.bias.shape()[0]
        )));
    }
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
    let norm = read_norm_gpu_layer(
        context,
        &tree.subtree("norm")?,
        data_type,
        channels,
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
        "structured_audio_convnext_norm",
    )?;
    let (pwconv1_shape, _) = read_float_array::<2>(&tree.subtree("pwconv1")?, "weights", data_type)?;
    if pwconv1_shape[1] != channels {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 input mismatch at 'pwconv1.weights': expected {}, got {}",
            channels, pwconv1_shape[1]
        )));
    }
    let pwconv1 = read_pointwise_conv_gpu_layer(&tree.subtree("pwconv1")?, data_type, pwconv1_shape[0], channels)?;
    let pwconv2 = read_pointwise_conv_gpu_layer(&tree.subtree("pwconv2")?, data_type, channels, pwconv1_shape[0])?;
    Ok(StructuredAudioConvNeXt {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv2,
    })
}

fn read_residual_unit_gpu_layer(
    tree: &MetalParameterTree<'_>,
    data_type: DataType,
    dilation: usize,
) -> AudioResult<StructuredAudioResidualUnit> {
    let snake1_alpha = {
        let snake1_tree = tree.subtree("snake1")?;
        let (shape, alpha) = read_float_array::<1>(&snake1_tree, "alpha", data_type)?;
        if shape[0] == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        alpha
    };
    let conv1 = read_conv1d_gpu_layer(&tree.subtree("conv1")?, data_type, dilation, 1)?;
    let snake2_alpha = {
        let snake2_tree = tree.subtree("snake2")?;
        let (shape, alpha) = read_float_array::<1>(&snake2_tree, "alpha", data_type)?;
        if shape[0] != conv1.cout {
            return Err(AudioError::Runtime(format!(
                "residual snake2 alpha mismatch: expected {}, got {}",
                conv1.cout, shape[0]
            )));
        }
        alpha
    };
    if snake1_alpha.shape()[0] != conv1.cin {
        return Err(AudioError::Runtime(format!(
            "residual snake1 alpha mismatch: expected {}, got {}",
            conv1.cin,
            snake1_alpha.shape()[0]
        )));
    }
    let conv2 = read_conv1d_gpu_layer(&tree.subtree("conv2")?, data_type, 1, 1)?;
    Ok(StructuredAudioResidualUnit {
        snake1_alpha,
        conv1,
        snake2_alpha,
        conv2,
    })
}

fn build_vocoder_gpu_graph_from_tree(
    context: &Rc<<Metal as Backend>::Context>,
    root: &MetalParameterTree<'_>,
    config: &DescriptAudioCodecConfig,
    data_type: DataType,
) -> AudioResult<StructuredAudioDecoderGraph> {
    let audio_decoder_tree = root.subtree("audio_decoder")?;
    let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
    let decoder_tree = audio_decoder_tree.subtree("decoder")?;

    let first_conv = read_conv1d_gpu_layer(&decoder_tree.subtree("first_conv")?, data_type, 1, 1)?;
    let final_conv = read_conv1d_gpu_layer(&decoder_tree.subtree("final_conv")?, data_type, 1, 1)?;
    let final_snake_alpha =
        read_float_vector_exact(&decoder_tree.subtree("final_snake")?, "alpha", final_conv.cin, data_type)?;

    let mut upsample_blocks = Vec::with_capacity(config.downsample_factor.len());
    for (index, &stride) in config.downsample_factor.iter().rev().enumerate() {
        let block_tree = quantizer_tree.subtree("upsampler")?.subtree("blocks")?.subtree(&index.to_string())?;
        let trans_conv = read_conv_transpose1d_gpu_layer(&block_tree.subtree("trans_conv")?, data_type, stride, 1)?;
        let convnext = read_convnext_gpu_layer(
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
        let trans_conv = read_conv_transpose1d_gpu_layer(&block_tree.subtree("trans_conv")?, data_type, stride, 1)?;
        let snake_alpha =
            read_float_vector_exact(&block_tree.subtree("snake")?, "alpha", trans_conv.cin, data_type)?;
        let channels = trans_conv.cout;
        let res_unit1 = read_residual_unit_gpu_layer(&block_tree.subtree("res_unit1")?, data_type, 1)?;
        let res_unit2 = read_residual_unit_gpu_layer(&block_tree.subtree("res_unit2")?, data_type, 3)?;
        let res_unit3 = read_residual_unit_gpu_layer(&block_tree.subtree("res_unit3")?, data_type, 9)?;
        if snake_alpha.shape()[0] != trans_conv.cin
            || res_unit1.conv1.cin != channels
            || res_unit2.conv1.cin != channels
            || res_unit3.conv1.cin != channels
        {
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
