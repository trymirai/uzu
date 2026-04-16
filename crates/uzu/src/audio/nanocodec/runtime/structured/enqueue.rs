use super::*;
use crate::backends::common::gpu_types::ActivationType;

pub(super) fn snake1d_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    alpha: &crate::backends::common::Allocation<B>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let expected_input = size_for_shape(&[batch_size, channels, seq_len], data_type);
    if input.as_buffer_range().1.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.as_buffer_range().1.len(),
        });
    }
    if alpha.as_buffer_range().1.len() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: channels,
            actual_tokens: alpha.as_buffer_range().1.len() / data_type.size_in_bytes(),
        });
    }

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let mut output = output;
    kernels.half_snake.encode(
        input,
        alpha,
        &mut output,
        channels_i32,
        seq_len_i32,
        channels_i32,
        0.0,
        1e-9,
        batch_i32,
        encoder,
    );

    Ok(output)
}

fn validate_grouped_conv_params<B: Backend>(
    input: &crate::backends::common::Allocation<B>,
    data_type: DataType,
    layer: &StructuredAudioConv1d<B>,
    lengths: &[i32],
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<()> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cout, layer.cin / layer.groups, layer.kernel_size];
    if layer.weight.as_buffer_range().1.len() != size_for_shape(&expected_weight_shape, data_type) {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}",
            expected_weight_shape
        )));
    }
    if layer.bias.as_buffer_range().1.len() != size_for_shape(&[layer.cout], data_type) {
        return Err(AudioError::Runtime(format!("causal conv1d bias shape mismatch: expected [{}]", layer.cout)));
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len], data_type);
    if input.as_buffer_range().1.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.as_buffer_range().1.len(),
        });
    }
    Ok(())
}

pub(super) fn causal_conv1d_grouped_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    layer: &StructuredAudioConv1d<B>,
    input_layout: SequenceLayout,
    lengths: &[i32],
    lengths_array: &crate::backends::common::Allocation<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    validate_grouped_conv_params(input, data_type, layer, lengths, batch_size, seq_len)?;

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let input_layout_i32 = input_layout.as_i32();
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let mut output = output;
    if layer.groups == 1 {
        kernels.causal_conv1d.encode(
            input,
            &layer.weight,
            &layer.bias,
            &mut output,
            lengths_array,
            cin_i32,
            cout_i32,
            seq_len_i32,
            kernel_size_i32,
            dilation_i32,
            input_layout_i32,
            batch_i32,
            encoder,
        );
    } else {
        let groups_i32 = usize_to_i32(layer.groups, "groups")?;
        kernels.causal_conv1d_grouped.encode(
            input,
            &layer.weight,
            &layer.bias,
            &mut output,
            lengths_array,
            cin_i32,
            cout_i32,
            seq_len_i32,
            kernel_size_i32,
            dilation_i32,
            groups_i32,
            input_layout_i32,
            batch_i32,
            encoder,
        );
    }

    Ok(output)
}

pub(super) fn causal_conv1d_grouped_residual_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    residual: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    layer: &StructuredAudioConv1d<B>,
    lengths: &[i32],
    lengths_array: &crate::backends::common::Allocation<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    validate_grouped_conv_params(input, data_type, layer, lengths, batch_size, seq_len)?;
    let expected_residual = size_for_shape(&[batch_size, layer.cout, seq_len], data_type);
    if residual.as_buffer_range().1.len() != expected_residual {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_residual,
            actual_tokens: residual.as_buffer_range().1.len(),
        });
    }
    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let mut output = output;
    kernels.causal_conv1d_grouped_residual.encode(
        input,
        residual,
        &layer.weight,
        &layer.bias,
        &mut output,
        lengths_array,
        cin_i32,
        cout_i32,
        seq_len_i32,
        kernel_size_i32,
        dilation_i32,
        groups_i32,
        batch_i32,
        encoder,
    );

    Ok(output)
}

pub(super) fn causal_conv_transpose1d_causal_pad_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    layer: &StructuredAudioConvTranspose1d<B>,
    lengths: &[i32],
    batch_size: usize,
    seq_len_in: usize,
    seq_len_out: usize,
    input_layout: SequenceLayout,
    lengths_array: &crate::backends::common::Allocation<B>,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len_in], data_type);
    if input.as_buffer_range().1.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.as_buffer_range().1.len(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cout, layer.cin / layer.groups, layer.kernel_size];
    if layer.weight.as_buffer_range().1.len() != size_for_shape(&expected_weight_shape, data_type) {
        return Err(AudioError::Runtime(format!(
            "causal transpose weight shape mismatch: expected {:?}",
            expected_weight_shape
        )));
    }
    if layer.bias.as_buffer_range().1.len() != size_for_shape(&[layer.cout], data_type) {
        return Err(AudioError::Runtime(format!("causal transpose bias shape mismatch: expected [{}]", layer.cout)));
    }

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_in_i32 = usize_to_i32(seq_len_in, "seq_len_in")?;
    let seq_out_i32 = usize_to_i32(seq_len_out, "seq_len_out")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let stride_i32 = usize_to_i32(layer.stride, "stride")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let input_layout_i32 = input_layout.as_i32();
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let mut output = output;
    kernels.causal_conv_transpose1d_causal_pad.encode(
        input,
        &layer.weight,
        &layer.bias,
        &mut output,
        lengths_array,
        cin_i32,
        cout_i32,
        seq_in_i32,
        seq_out_i32,
        kernel_size_i32,
        stride_i32,
        groups_i32,
        input_layout_i32,
        batch_i32,
        encoder,
    );

    Ok(output)
}

pub(super) fn conv1d_pointwise_ncs_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    layer: &StructuredAudioPointwiseConv<B>,
    lengths: &[i32],
    lengths_array: &crate::backends::common::Allocation<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len], data_type);
    if input.as_buffer_range().1.len() != expected_input {
        return Err(AudioError::Runtime(format!(
            "pointwise conv input shape mismatch: expected {expected_input} bytes ([batch={batch_size}, cin={}, seq_len={seq_len}]), got {}",
            layer.cin,
            input.as_buffer_range().1.len()
        )));
    }
    if layer.weight.as_buffer_range().1.len() != size_for_shape(&[layer.cout, layer.cin], data_type) {
        return Err(AudioError::Runtime(format!(
            "pointwise conv weight shape mismatch: expected [{}, {}]",
            layer.cout, layer.cin
        )));
    }
    if layer.bias.as_buffer_range().1.len() != size_for_shape(&[layer.cout], data_type) {
        return Err(AudioError::Runtime(format!("pointwise conv bias shape mismatch: expected [{}]", layer.cout)));
    }

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let mut output = output;
    kernels.conv1d.encode(
        input,
        &layer.weight,
        &layer.bias,
        &mut output,
        lengths_array,
        cin_i32,
        cout_i32,
        seq_len_i32,
        seq_len_i32,
        1,
        1,
        1,
        0,
        0,
        batch_i32,
        encoder,
    );

    Ok(output)
}

pub(super) fn norm_ncs_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    norm: &StructuredAudioNorm<B>,
    lengths: &[i32],
    lengths_array: &crate::backends::common::Allocation<B>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if norm.scales.as_buffer_range().1.len() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::Runtime(format!("norm scale shape mismatch: expected [{channels}]")));
    }
    if norm.bias.as_buffer_range().1.len() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::Runtime(format!("norm bias shape mismatch: expected [{channels}]")));
    }

    let expected_input = size_for_shape(&[batch_size, channels, seq_len], data_type);
    if input.as_buffer_range().1.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.as_buffer_range().1.len(),
        });
    }

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let subtract_mean = if norm.subtract_mean {
        1
    } else {
        0
    };
    let mut output = output;
    kernels.norm_ncs.encode(
        input,
        &norm.scales,
        &norm.bias,
        &mut output,
        lengths_array,
        channels_i32,
        seq_len_i32,
        norm.epsilon,
        subtract_mean,
        batch_i32,
        encoder,
    );

    Ok(output)
}

pub(super) fn gelu_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let n_u32 = u32::try_from(input.as_buffer_range().1.len() / data_type.size_in_bytes())
        .map_err(|_| AudioError::Runtime("gelu element count exceeds u32 range".to_string()))?;
    let mut output = output;
    kernels.activation.encode(Some(input), &mut output, n_u32, ActivationType::GELU, encoder);
    Ok(output)
}

pub(super) fn add_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    a: &crate::backends::common::Allocation<B>,
    b: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    if a.as_buffer_range().1.len() != b.as_buffer_range().1.len() {
        return Err(AudioError::Runtime(format!(
            "elementwise add shape mismatch: {} vs {}",
            a.as_buffer_range().1.len(),
            b.as_buffer_range().1.len()
        )));
    }
    let n_i32 = usize_to_i32(a.as_buffer_range().1.len() / data_type.size_in_bytes(), "n")?;
    let mut output = output;
    kernels.add.encode(a, b, &mut output, n_i32, encoder);
    Ok(output)
}

pub(super) fn tanh_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &crate::backends::common::Allocation<B>,
    output: crate::backends::common::Allocation<B>,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let n_u32 = u32::try_from(input.as_buffer_range().1.len() / data_type.size_in_bytes())
        .map_err(|_| AudioError::Runtime("tanh element count exceeds u32 range".to_string()))?;
    let mut output = output;
    kernels.activation.encode(Some(input), &mut output, n_u32, ActivationType::TANH, encoder);
    Ok(output)
}
