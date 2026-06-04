use super::*;
use crate::{array::Array, backends::common::gpu_types::ActivationType, parameters::ParameterTree};

pub(super) fn copy_to_outer_axis_slice<B: Backend>(
    destination: &mut Array<B>,
    index: usize,
    source: &Array<B>,
    slice_shape: &[usize],
) -> AudioResult<()> {
    let slice_bytes = size_for_shape(slice_shape, source.data_type());
    let offset =
        index.checked_mul(slice_bytes).ok_or(AudioError::Runtime("outer-axis slice offset overflow".to_string()))?;
    let end =
        offset.checked_add(slice_bytes).ok_or(AudioError::Runtime("outer-axis slice end overflow".to_string()))?;
    if end > destination.size() {
        return Err(AudioError::Runtime(format!(
            "outer-axis slice exceeds destination: end {end}, destination size {}",
            destination.size()
        )));
    }
    destination.as_bytes_mut()[offset..end].copy_from_slice(source.as_bytes());
    Ok(())
}

pub(super) fn read_conv1d_layer<B: Backend>(
    tree: &ParameterTree<B>,
    data_type: DataType,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConv1d<B>> {
    if groups == 0 || cin == 0 || cout == 0 || kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if !cin.is_multiple_of(groups) || !cout.is_multiple_of(groups) {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv dimensions: cin {cin}, cout {cout}, groups {groups}"
        )));
    }
    let weight = tree.leaf("weights")?.validate(&[cout, cin / groups, kernel_size], data_type)?.read_array()?;
    let bias = tree.leaf("biases")?.validate(&[cout], data_type)?.read_array()?;
    Ok(StructuredAudioConv1d {
        weight,
        bias,
        cin,
        cout,
        kernel_size,
        dilation,
        groups,
    })
}

pub(super) fn read_conv_transpose1d_layer<B: Backend>(
    tree: &ParameterTree<B>,
    data_type: DataType,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
) -> AudioResult<StructuredAudioConvTranspose1d<B>> {
    if stride == 0 || groups == 0 || cin == 0 || cout == 0 || kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if !cin.is_multiple_of(groups) || !cout.is_multiple_of(groups) {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight = tree.leaf("weights")?.validate(&[cout, cin / groups, kernel_size], data_type)?.read_array()?;
    let bias = tree.leaf("biases")?.validate(&[cout], data_type)?.read_array()?;
    Ok(StructuredAudioConvTranspose1d {
        weight,
        bias,
        cin,
        cout,
        kernel_size,
        stride,
        groups,
    })
}

pub(super) fn read_pointwise_conv_layer<B: Backend>(
    tree: &ParameterTree<B>,
    data_type: DataType,
    expected_out_dim: usize,
    expected_in_dim: usize,
) -> AudioResult<StructuredAudioPointwiseConv<B>> {
    let weight =
        tree.leaf("weights.weights")?.validate(&[expected_out_dim, expected_in_dim], data_type)?.read_array()?;
    let bias = tree.leaf("biases")?.validate(&[expected_out_dim], data_type)?.read_array()?;
    Ok(StructuredAudioPointwiseConv {
        weight,
        bias,
        cin: expected_in_dim,
        cout: expected_out_dim,
    })
}

pub(super) fn read_norm_layer<B: Backend>(
    context: &Rc<B::Context>,
    tree: &ParameterTree<B>,
    channels: usize,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<StructuredAudioNorm<B>> {
    let norm_data_type = DataType::F32;
    let scales = tree.leaf("scales")?.validate(&[channels], norm_data_type)?.read_array()?;
    let bias = if use_bias {
        tree.leaf("biases")?.validate(&[channels], norm_data_type)?.read_array()?
    } else {
        context.create_array_zeros(&[channels], norm_data_type)
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
    tree: &ParameterTree<B>,
    data_type: DataType,
    activation_type: ActivationType,
    norm_config: &NormalizationConfig,
    dim: usize,
) -> AudioResult<StructuredAudioConvNeXt<B>> {
    let hidden_dim = dim.checked_mul(4).ok_or(AudioError::Runtime("ConvNeXt hidden dimension overflow".to_string()))?;
    let depthwise_conv = read_conv1d_layer::<B>(&tree.subtree("depthwise_conv")?, data_type, dim, dim, 7, 1, dim)?;
    let norm = read_norm_layer::<B>(
        context,
        &tree.subtree("norm")?,
        dim,
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.has_biases,
    )?;
    let pwconv1 = read_pointwise_conv_layer::<B>(&tree.subtree("pointwise_conv_step1")?, data_type, hidden_dim, dim)?;
    let pwconv2 = read_pointwise_conv_layer::<B>(&tree.subtree("pointwise_conv_step2")?, data_type, dim, hidden_dim)?;
    Ok(StructuredAudioConvNeXt {
        activation_type,
        depthwise_conv,
        norm,
        pwconv1,
        pwconv2,
    })
}

pub(super) fn read_residual_unit_layer<B: Backend>(
    tree: &ParameterTree<B>,
    data_type: DataType,
    dilation: usize,
    dim: usize,
) -> AudioResult<StructuredAudioResidualUnit<B>> {
    let alpha_shape = [dim];
    let snake1_alpha = tree.leaf("snake1.alpha")?.validate(&alpha_shape, data_type)?.read_array()?;
    let conv1 = read_conv1d_layer::<B>(&tree.subtree("conv1")?, data_type, dim, dim, 7, dilation, 1)?;
    let snake2_alpha = tree.leaf("snake2.alpha")?.validate(&alpha_shape, data_type)?.read_array()?;
    let conv2 = read_conv1d_layer::<B>(&tree.subtree("conv2")?, data_type, dim, dim, 1, 1, 1)?;
    Ok(StructuredAudioResidualUnit {
        snake1_alpha,
        conv1,
        snake2_alpha,
        conv2,
    })
}

pub(super) fn build_vocoder_graph_from_tree<B: Backend>(
    context: &Rc<B::Context>,
    root: &ParameterTree<B>,
    config: &DescriptAudioCodecConfig,
    data_type: DataType,
) -> AudioResult<StructuredAudioDecoderGraph<B>> {
    let audio_decoder_tree = root.subtree("audio_decoder")?;
    let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
    let decoder_tree = audio_decoder_tree.subtree("decoder")?;

    let latent_dim = config.encoder_dim * (1usize << config.encoder_rates.len());
    let final_dim = config.decoder_dim / (1usize << config.decoder_rates.len());

    let first_conv = read_conv1d_layer::<B>(
        &decoder_tree.subtree("first_conv")?,
        data_type,
        latent_dim,
        config.decoder_dim,
        7,
        1,
        1,
    )?;
    let final_conv = read_conv1d_layer::<B>(&decoder_tree.subtree("final_conv")?, data_type, final_dim, 1, 7, 1, 1)?;
    let final_snake_alpha = decoder_tree.leaf("final_snake.alpha")?.validate(&[final_dim], data_type)?.read_array()?;

    let mut upsample_blocks = Vec::with_capacity(config.downsample_factor.len());
    for (index, &stride) in config.downsample_factor.iter().rev().enumerate() {
        let block_tree = quantizer_tree.subtree("upsampler")?.subtree("blocks")?.subtree(&index.to_string())?;
        let trans_conv = read_conv_transpose1d_layer::<B>(
            &block_tree.subtree("trans_conv")?,
            data_type,
            config.input_dim,
            config.input_dim,
            stride,
            stride,
            1,
        )?;
        let convnext = read_convnext_layer::<B>(
            context,
            &block_tree.subtree("convnext")?,
            data_type,
            config.quantizer_config.upsampler_config.block_configs[index].convnext_config.activation.act_type(),
            &config.quantizer_config.upsampler_config.block_configs[index].convnext_config.norm_config,
            config.input_dim,
        )?;
        upsample_blocks.push((trans_conv, convnext));
    }

    let mut decoder_blocks = Vec::with_capacity(config.decoder_rates.len());
    for (index, &stride) in config.decoder_rates.iter().enumerate() {
        let block_tree = decoder_tree.subtree("decoder_blocks")?.subtree(&index.to_string())?;
        let input_dim = config.decoder_dim / (1usize << index);
        let output_dim = config.decoder_dim / (1usize << (index + 1));
        let trans_conv = read_conv_transpose1d_layer::<B>(
            &block_tree.subtree("trans_conv")?,
            data_type,
            input_dim,
            output_dim,
            2 * stride,
            stride,
            1,
        )?;
        let snake_alpha = block_tree.leaf("snake.alpha")?.validate(&[input_dim], data_type)?.read_array()?;
        let res_unit1 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit1")?, data_type, 1, output_dim)?;
        let res_unit2 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit2")?, data_type, 3, output_dim)?;
        let res_unit3 = read_residual_unit_layer::<B>(&block_tree.subtree("res_unit3")?, data_type, 9, output_dim)?;
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
