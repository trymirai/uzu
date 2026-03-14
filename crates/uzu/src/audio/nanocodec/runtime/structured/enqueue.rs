fn transpose_nsc_to_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
    channels: usize,
) -> AudioResult<Array<Metal>> {
    let expected = checked_product(&[batch_size, seq_len, channels])?;
    if input.num_elements() != expected {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected,
            actual_tokens: input.num_elements(),
        });
    }

    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_ncs_output");
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let channels_i32 = usize_to_i32(channels, "channels")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.transpose_nsc_to_ncs.encode(
            &*input_buffer,
            &mut *output_buffer,
            seq_len_i32,
            channels_i32,
            batch_i32,
            compute_encoder,
        );
    });
    Ok(output)
}

fn snake1d_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    alpha: &Array<Metal>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    if alpha.shape() != [channels] {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: channels,
            actual_tokens: alpha.num_elements(),
        });
    }

    let data_type = input.data_type();
    if alpha.data_type() != data_type {
        return Err(AudioError::Runtime(format!(
            "snake alpha dtype mismatch: expected {:?}, got {:?}",
            data_type,
            alpha.data_type()
        )));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_snake_output");

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let alpha_buffer = alpha.buffer();
    let alpha_buffer = alpha_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.half_snake.encode(
            &*input_buffer,
            &*alpha_buffer,
            &mut *output_buffer,
            channels_i32,
            seq_len_i32,
            channels_i32,
            0.0,
            1e-9,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn causal_conv1d_grouped_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    layer: &StructuredAudioConv1dGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
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
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal conv1d bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("causal conv1d dtype mismatch".to_string()));
    }
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_causal_conv1d_output");

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let kernels = fishaudio_kernels(context, data_type)?;
    if layer.groups == 1 {
        let input_buffer = input.buffer();
        let input_buffer = input_buffer.borrow();
        let weight_buffer = layer.weight.buffer();
        let weight_buffer = weight_buffer.borrow();
        let bias_buffer = layer.bias.buffer();
        let bias_buffer = bias_buffer.borrow();
        let output_buffer = output.buffer();
        let mut output_buffer = output_buffer.borrow_mut();
        let lengths_buffer = lengths_array.buffer();
        let lengths_buffer = lengths_buffer.borrow();
        command_buffer.with_compute_encoder(|compute_encoder| {
            kernels.causal_conv1d.encode(
                &*input_buffer,
                &*weight_buffer,
                &*bias_buffer,
                &mut *output_buffer,
                &*lengths_buffer,
                cin_i32,
                cout_i32,
                seq_len_i32,
                kernel_size_i32,
                dilation_i32,
                batch_i32,
                compute_encoder,
            );
        });
    } else {
        let groups_i32 = usize_to_i32(layer.groups, "groups")?;
        let input_buffer = input.buffer();
        let input_buffer = input_buffer.borrow();
        let weight_buffer = layer.weight.buffer();
        let weight_buffer = weight_buffer.borrow();
        let bias_buffer = layer.bias.buffer();
        let bias_buffer = bias_buffer.borrow();
        let output_buffer = output.buffer();
        let mut output_buffer = output_buffer.borrow_mut();
        let lengths_buffer = lengths_array.buffer();
        let lengths_buffer = lengths_buffer.borrow();
        command_buffer.with_compute_encoder(|compute_encoder| {
            kernels.causal_conv1d_grouped.encode(
                &*input_buffer,
                &*weight_buffer,
                &*bias_buffer,
                &mut *output_buffer,
                &*lengths_buffer,
                cin_i32,
                cout_i32,
                seq_len_i32,
                kernel_size_i32,
                dilation_i32,
                groups_i32,
                batch_i32,
                compute_encoder,
            );
        });
    }

    Ok(output)
}

fn causal_conv1d_grouped_residual_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    residual: &Array<Metal>,
    layer: &StructuredAudioConv1dGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
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
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal conv1d weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal conv1d bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    let expected_residual = checked_product(&[batch_size, layer.cout, seq_len])?;
    if residual.num_elements() != expected_residual {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_residual,
            actual_tokens: residual.num_elements(),
        });
    }
    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let dilation_i32 = usize_to_i32(layer.dilation, "dilation")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let data_type = input.data_type();
    if residual.data_type() != data_type || layer.weight.data_type() != data_type || layer.bias.data_type() != data_type
    {
        return Err(AudioError::Runtime("causal conv1d residual dtype mismatch".to_string()));
    }
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_causal_conv1d_output");
    let kernels = fishaudio_kernels(context, data_type)?;

    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let residual_buffer = residual.buffer();
    let residual_buffer = residual_buffer.borrow();
    let weight_buffer = layer.weight.buffer();
    let weight_buffer = weight_buffer.borrow();
    let bias_buffer = layer.bias.buffer();
    let bias_buffer = bias_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    let lengths_buffer = lengths_array.buffer();
    let lengths_buffer = lengths_buffer.borrow();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.causal_conv1d_grouped_residual.encode(
            &*input_buffer,
            &*residual_buffer,
            &*weight_buffer,
            &*bias_buffer,
            &mut *output_buffer,
            &*lengths_buffer,
            cin_i32,
            cout_i32,
            seq_len_i32,
            kernel_size_i32,
            dilation_i32,
            groups_i32,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn causal_conv_transpose1d_causal_pad_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    layer: &StructuredAudioConvTranspose1dGpuLayer,
    lengths: &[i32],
    batch_size: usize,
    seq_len_in: usize,
    seq_len_out: usize,
    input_layout: SequenceLayout,
    lengths_array: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len_in])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let expected_weight_shape = [layer.cin, layer.cout / layer.groups, layer.kernel_size];
    if layer.weight.shape() != expected_weight_shape {
        return Err(AudioError::Runtime(format!(
            "causal transpose weight shape mismatch: expected {:?}, got {:?}",
            expected_weight_shape,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "causal transpose bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }

    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("causal transpose dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(
        &[batch_size, layer.cout, seq_len_out],
        data_type,
        "fishaudio_causal_conv_transpose_output",
    );

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_in_i32 = usize_to_i32(seq_len_in, "seq_len_in")?;
    let seq_out_i32 = usize_to_i32(seq_len_out, "seq_len_out")?;
    let kernel_size_i32 = usize_to_i32(layer.kernel_size, "kernel_size")?;
    let stride_i32 = usize_to_i32(layer.stride, "stride")?;
    let groups_i32 = usize_to_i32(layer.groups, "groups")?;
    let input_layout_i32 = input_layout.as_i32();
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;

    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let weight_buffer = layer.weight.buffer();
    let weight_buffer = weight_buffer.borrow();
    let bias_buffer = layer.bias.buffer();
    let bias_buffer = bias_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    let lengths_buffer = lengths_array.buffer();
    let lengths_buffer = lengths_buffer.borrow();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.causal_conv_transpose1d_causal_pad.encode(
            &*input_buffer,
            &*weight_buffer,
            &*bias_buffer,
            &mut *output_buffer,
            &*lengths_buffer,
            cin_i32,
            cout_i32,
            seq_in_i32,
            seq_out_i32,
            kernel_size_i32,
            stride_i32,
            groups_i32,
            input_layout_i32,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn conv1d_pointwise_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    layer: &StructuredAudioPointwiseConvGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::Runtime(format!(
            "pointwise conv input shape mismatch: expected {expected_input} elements ([batch={batch_size}, cin={}, seq_len={seq_len}]), got {}",
            layer.cin,
            input.num_elements()
        )));
    }
    if layer.weight.shape() != [layer.cout, layer.cin, 1] {
        return Err(AudioError::Runtime(format!(
            "pointwise conv weight shape mismatch: expected [{}, {}, 1], got {:?}",
            layer.cout,
            layer.cin,
            layer.weight.shape()
        )));
    }
    if layer.bias.shape() != [layer.cout] {
        return Err(AudioError::Runtime(format!(
            "pointwise conv bias shape mismatch: expected [{}], got {:?}",
            layer.cout,
            layer.bias.shape()
        )));
    }

    let data_type = input.data_type();
    if layer.weight.data_type() != data_type || layer.bias.data_type() != data_type {
        return Err(AudioError::Runtime("pointwise conv dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, layer.cout, seq_len], data_type, "fishaudio_pwconv_output");

    let cin_i32 = usize_to_i32(layer.cin, "cin")?;
    let cout_i32 = usize_to_i32(layer.cout, "cout")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let weight_buffer = layer.weight.buffer();
    let weight_buffer = weight_buffer.borrow();
    let bias_buffer = layer.bias.buffer();
    let bias_buffer = bias_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    let lengths_buffer = lengths_array.buffer();
    let lengths_buffer = lengths_buffer.borrow();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.conv1d.encode(
            &*input_buffer,
            &*weight_buffer,
            &*bias_buffer,
            &mut *output_buffer,
            &*lengths_buffer,
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
            compute_encoder,
        );
    });

    Ok(output)
}

fn norm_ncs_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
    norm: &StructuredAudioNormGpuLayer,
    lengths: &[i32],
    lengths_array: &Array<Metal>,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
) -> AudioResult<Array<Metal>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if norm.scales.shape() != [channels] {
        return Err(AudioError::Runtime(format!(
            "norm scale shape mismatch: expected [{channels}], got {:?}",
            norm.scales.shape()
        )));
    }
    if norm.bias.shape() != [channels] {
        return Err(AudioError::Runtime(format!(
            "norm bias shape mismatch: expected [{channels}], got {:?}",
            norm.bias.shape()
        )));
    }

    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.num_elements() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.num_elements(),
        });
    }

    let data_type = input.data_type();
    if norm.scales.data_type() != data_type || norm.bias.data_type() != data_type {
        return Err(AudioError::Runtime("norm dtype mismatch".to_string()));
    }
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(&[batch_size, channels, seq_len], data_type, "fishaudio_norm_output");

    let channels_i32 = usize_to_i32(channels, "channels")?;
    let seq_len_i32 = usize_to_i32(seq_len, "seq_len")?;
    let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
    let subtract_mean = if norm.subtract_mean {
        1
    } else {
        0
    };
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let scales_buffer = norm.scales.buffer();
    let scales_buffer = scales_buffer.borrow();
    let bias_buffer = norm.bias.buffer();
    let bias_buffer = bias_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    let lengths_buffer = lengths_array.buffer();
    let lengths_buffer = lengths_buffer.borrow();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.norm_ncs.encode(
            &*input_buffer,
            &*scales_buffer,
            &*bias_buffer,
            &mut *output_buffer,
            &*lengths_buffer,
            channels_i32,
            seq_len_i32,
            norm.epsilon,
            subtract_mean,
            batch_i32,
            compute_encoder,
        );
    });

    Ok(output)
}

fn gelu_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(input.shape(), data_type, "fishaudio_gelu_output");
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("gelu element count exceeds u32 range".to_string()))?;
    let gelu_id = 1_u32;
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.activation.encode(Some(&*input_buffer), &mut *output_buffer, n_u32, gelu_id, compute_encoder);
    });
    Ok(output)
}

fn add_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    a: &Array<Metal>,
    b: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    if a.num_elements() != b.num_elements() {
        return Err(AudioError::Runtime(format!(
            "elementwise add shape mismatch: {} vs {}",
            a.num_elements(),
            b.num_elements()
        )));
    }
    if a.data_type() != b.data_type() {
        return Err(AudioError::Runtime(format!(
            "elementwise add dtype mismatch: {:?} vs {:?}",
            a.data_type(),
            b.data_type()
        )));
    }
    let data_type = a.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(a.shape(), data_type, "fishaudio_add_out");
    let n_i32 = usize_to_i32(a.num_elements(), "n")?;
    let a_buffer = a.buffer();
    let a_buffer = a_buffer.borrow();
    let b_buffer = b.buffer();
    let b_buffer = b_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.add.encode(&*a_buffer, &*b_buffer, &mut *output_buffer, n_i32, compute_encoder);
    });
    Ok(output)
}

fn tanh_enqueue(
    context: &Rc<<Metal as Backend>::Context>,
    command_buffer: &mut MetalCommandBuffer,
    input: &Array<Metal>,
) -> AudioResult<Array<Metal>> {
    let data_type = input.data_type();
    let kernels = fishaudio_kernels(context, data_type)?;
    let output = context.create_array(input.shape(), data_type, "fishaudio_tanh_output");
    let n_u32 = u32::try_from(input.num_elements())
        .map_err(|_| AudioError::Runtime("tanh element count exceeds u32 range".to_string()))?;
    let tanh_id = 2_u32;
    let input_buffer = input.buffer();
    let input_buffer = input_buffer.borrow();
    let output_buffer = output.buffer();
    let mut output_buffer = output_buffer.borrow_mut();
    command_buffer.with_compute_encoder(|compute_encoder| {
        kernels.activation.encode(Some(&*input_buffer), &mut *output_buffer, n_u32, tanh_id, compute_encoder);
    });
    Ok(output)
}

