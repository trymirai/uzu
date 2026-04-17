use super::*;
use crate::{array::Array, backends::common::gpu_types::ActivationType};

pub(super) fn snake1d_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    alpha: &Array<B>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    let expected_input = size_for_shape(&[batch_size, channels, seq_len], data_type);
    if input.size() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.size(),
        });
    }
    if alpha.size() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: channels,
            actual_tokens: alpha.num_elements(),
        });
    }

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.half_snake.encode(
        input.allocation(),
        alpha.allocation(),
        &mut output,
        channels_i32,
        seq_len_i32,
        channels_i32,
        0.0,
        1e-9,
        batch_i32,
        encoder,
    );

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

fn validate_grouped_conv_params<B: Backend>(
    input: &Array<B>,
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
    if layer.weight.size() != size_for_shape(&expected_weight_shape, data_type) {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}",
            expected_weight_shape
        )));
    }
    if layer.bias.size() != size_for_shape(&[layer.cout], data_type) {
        return Err(AudioError::Runtime(format!("causal conv1d bias shape mismatch: expected [{}]", layer.cout)));
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len], data_type);
    if input.size() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.size(),
        });
    }
    Ok(())
}

pub(super) fn causal_conv1d_grouped_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    layer: &StructuredAudioConv1d<B>,
    input_layout: SequenceLayout,
    lengths: &[i32],
    lengths_array: &Array<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    validate_grouped_conv_params(input, data_type, layer, lengths, batch_size, seq_len)?;

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let input_layout_i32 = input_layout.as_i32();
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    if layer.groups == 1 {
        kernels.causal_conv1d.encode(
            input.allocation(),
            layer.weight.allocation(),
            layer.bias.allocation(),
            &mut output,
            lengths_array.allocation(),
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
            input.allocation(),
            layer.weight.allocation(),
            layer.bias.allocation(),
            &mut output,
            lengths_array.allocation(),
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

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn causal_conv1d_grouped_residual_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    residual: &Array<B>,
    output: Array<B>,
    layer: &StructuredAudioConv1d<B>,
    lengths: &[i32],
    lengths_array: &Array<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    validate_grouped_conv_params(input, data_type, layer, lengths, batch_size, seq_len)?;
    let expected_residual = size_for_shape(&[batch_size, layer.cout, seq_len], data_type);
    if residual.size() != expected_residual {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_residual,
            actual_tokens: residual.size(),
        });
    }
    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.causal_conv1d_grouped_residual.encode(
        input.allocation(),
        residual.allocation(),
        layer.weight.allocation(),
        layer.bias.allocation(),
        &mut output,
        lengths_array.allocation(),
        cin_i32,
        cout_i32,
        seq_len_i32,
        kernel_size_i32,
        dilation_i32,
        groups_i32,
        batch_i32,
        encoder,
    );

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn causal_conv_transpose1d_causal_pad_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    layer: &StructuredAudioConvTranspose1d<B>,
    lengths: &[i32],
    batch_size: usize,
    seq_len_in: usize,
    seq_len_out: usize,
    input_layout: SequenceLayout,
    lengths_array: &Array<B>,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len_in], data_type);
    if input.size() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.size(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cout, layer.cin / layer.groups, layer.kernel_size];
    if layer.weight.size() != size_for_shape(&expected_weight_shape, data_type) {
        return Err(AudioError::Runtime(format!(
            "causal transpose weight shape mismatch: expected {:?}",
            expected_weight_shape
        )));
    }
    if layer.bias.size() != size_for_shape(&[layer.cout], data_type) {
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
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.causal_conv_transpose1d_causal_pad.encode(
        input.allocation(),
        layer.weight.allocation(),
        layer.bias.allocation(),
        &mut output,
        lengths_array.allocation(),
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

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn conv1d_pointwise_ncs_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    layer: &StructuredAudioPointwiseConv<B>,
    lengths: &[i32],
    lengths_array: &Array<B>,
    batch_size: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = size_for_shape(&[batch_size, layer.cin, seq_len], data_type);
    if input.size() != expected_input {
        return Err(AudioError::Runtime(format!(
            "pointwise conv input shape mismatch: expected {expected_input} bytes ([batch={batch_size}, cin={}, seq_len={seq_len}]), got {}",
            layer.cin,
            input.size()
        )));
    }
    if layer.weight.size() != size_for_shape(&[layer.cout, layer.cin], data_type) {
        return Err(AudioError::Runtime(format!(
            "pointwise conv weight shape mismatch: expected [{}, {}]",
            layer.cout, layer.cin
        )));
    }
    if layer.bias.size() != size_for_shape(&[layer.cout], data_type) {
        return Err(AudioError::Runtime(format!("pointwise conv bias shape mismatch: expected [{}]", layer.cout)));
    }

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.conv1d.encode(
        input.allocation(),
        layer.weight.allocation(),
        layer.bias.allocation(),
        &mut output,
        lengths_array.allocation(),
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

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn norm_ncs_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    norm: &StructuredAudioNorm<B>,
    lengths: &[i32],
    lengths_array: &Array<B>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
    data_type: DataType,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if norm.scales.size() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::Runtime(format!("norm scale shape mismatch: expected [{channels}]")));
    }
    if norm.bias.size() != size_for_shape(&[channels], data_type) {
        return Err(AudioError::Runtime(format!("norm bias shape mismatch: expected [{channels}]")));
    }

    let expected_input = size_for_shape(&[batch_size, channels, seq_len], data_type);
    if input.size() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.size(),
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
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.norm_ncs.encode(
        input.allocation(),
        norm.scales.allocation(),
        norm.bias.allocation(),
        &mut output,
        lengths_array.allocation(),
        channels_i32,
        seq_len_i32,
        norm.epsilon,
        subtract_mean,
        batch_i32,
        encoder,
    );

    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn gelu_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("gelu element count exceeds u32 range".to_string()))?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.activation.encode(Some(input.allocation()), &mut output, n_u32, ActivationType::GELU, encoder);
    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn add_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    a: &Array<B>,
    b: &Array<B>,
    output: Array<B>,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    if a.size() != b.size() {
        return Err(AudioError::Runtime(format!("elementwise add shape mismatch: {} vs {}", a.size(), b.size())));
    }
    let n_i32 = usize_to_i32(a.num_elements(), "n")?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.add.encode(a.allocation(), b.allocation(), &mut output, n_i32, encoder);
    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}

pub(super) fn tanh_enqueue<B: Backend>(
    encoder: &mut Encoder<B>,
    input: &Array<B>,
    output: Array<B>,
    kernels: &StructuredAudioKernelCache<B>,
) -> AudioResult<Array<B>> {
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("tanh element count exceeds u32 range".to_string()))?;
    let output_shape = output.shape().to_vec();
    let output_data_type = output.data_type();
    let mut output = output.into_allocation();
    kernels.activation.encode(Some(input.allocation()), &mut output, n_u32, ActivationType::TANH, encoder);
    Ok(unsafe { Array::from_allocation(output, 0, &output_shape, output_data_type) })
}
