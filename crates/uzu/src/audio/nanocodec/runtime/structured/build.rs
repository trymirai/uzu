fn create_data_array(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    shape: &[usize],
    values: &[f32],
    label: &str,
) -> AudioResult<Array<Metal>> {
    let expected = checked_product(shape)?;
    if expected != values.len() {
        return Err(AudioError::Runtime(format!(
            "tensor '{label}' shape/value mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    let mut array = context.create_array(shape, data_type, label);
    write_f32_slice_to_array(&mut array, values)?;
    Ok(array)
}

fn create_alpha_gpu_array(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    channels: usize,
    alpha: &[f32],
    label: &str,
) -> AudioResult<Array<Metal>> {
    if alpha.len() != channels {
        return Err(AudioError::Runtime(format!(
            "alpha length mismatch for '{label}': expected {channels}, got {}",
            alpha.len()
        )));
    }
    create_data_array(context, data_type, &[channels], alpha, label)
}

fn create_conv1d_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &StructuredAudioConv1dLayer,
    label_prefix: &str,
) -> AudioResult<StructuredAudioConv1dGpuLayer> {
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let cin_per_group = layer.cin / layer.groups;
    let weight_shape = [layer.cout, cin_per_group, layer.kernel_size];
    let weight =
        create_data_array(context, data_type, &weight_shape, &layer.weight, &format!("{label_prefix}_weight"))?;
    let bias = create_data_array(context, data_type, &[layer.cout], &layer.bias, &format!("{label_prefix}_bias"))?;
    Ok(StructuredAudioConv1dGpuLayer {
        weight,
        bias,
        cin: layer.cin,
        cout: layer.cout,
        kernel_size: layer.kernel_size,
        dilation: layer.dilation,
        groups: layer.groups,
    })
}

fn create_conv_transpose1d_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &StructuredAudioConvTranspose1dLayer,
    label_prefix: &str,
) -> AudioResult<StructuredAudioConvTranspose1dGpuLayer> {
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight_plane = checked_product(&[layer.cin, layer.cout / layer.groups])?;
    if weight_plane == 0 || layer.weight.len() % weight_plane != 0 {
        return Err(AudioError::Runtime(format!("transpose layer '{label_prefix}' has invalid weight shape")));
    }
    let kernel_size = layer.kernel_size;
    if kernel_size == 0
        || layer.weight.len()
            != weight_plane
                .checked_mul(kernel_size)
                .ok_or(AudioError::Runtime(format!("transpose layer '{label_prefix}' kernel shape overflow")))?
    {
        return Err(AudioError::Runtime(format!(
            "transpose layer '{label_prefix}' has invalid kernel_size={kernel_size} for weight len {}",
            layer.weight.len()
        )));
    }
    let weight = create_data_array(
        context,
        data_type,
        &[layer.cin, layer.cout / layer.groups, kernel_size],
        &layer.weight,
        &format!("{label_prefix}_weight"),
    )?;
    let bias = create_data_array(context, data_type, &[layer.cout], &layer.bias, &format!("{label_prefix}_bias"))?;
    Ok(StructuredAudioConvTranspose1dGpuLayer {
        weight,
        bias,
        cin: layer.cin,
        cout: layer.cout,
        kernel_size,
        stride: layer.stride,
        groups: layer.groups,
    })
}

fn create_pointwise_conv_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    weight: &MatrixF32,
    bias: &[f32],
    label_prefix: &str,
) -> AudioResult<StructuredAudioPointwiseConvGpuLayer> {
    if bias.len() != weight.rows {
        return Err(AudioError::Runtime(format!(
            "pointwise layer '{label_prefix}' bias mismatch: expected {}, got {}",
            weight.rows,
            bias.len()
        )));
    }
    let weight_array = create_data_array(
        context,
        data_type,
        &[weight.rows, weight.cols, 1],
        &weight.values,
        &format!("{label_prefix}_weight"),
    )?;
    let bias_array = create_data_array(context, data_type, &[weight.rows], bias, &format!("{label_prefix}_bias"))?;
    Ok(StructuredAudioPointwiseConvGpuLayer {
        weight: weight_array,
        bias: bias_array,
        cin: weight.cols,
        cout: weight.rows,
    })
}

fn create_norm_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &StructuredAudioNormLayer,
    channels: usize,
    label_prefix: &str,
) -> AudioResult<StructuredAudioNormGpuLayer> {
    if layer.scales.len() != channels {
        return Err(AudioError::Runtime(format!(
            "norm layer '{label_prefix}' scale mismatch: expected {channels}, got {}",
            layer.scales.len()
        )));
    }
    let mut bias = vec![0.0_f32; channels];
    if let Some(bias_values) = &layer.biases {
        if bias_values.len() != channels {
            return Err(AudioError::Runtime(format!(
                "norm layer '{label_prefix}' bias mismatch: expected {channels}, got {}",
                bias_values.len()
            )));
        }
        bias.copy_from_slice(bias_values);
    }
    Ok(StructuredAudioNormGpuLayer {
        scales: create_data_array(context, data_type, &[channels], &layer.scales, &format!("{label_prefix}_scales"))?,
        bias: create_data_array(context, data_type, &[channels], &bias, &format!("{label_prefix}_bias"))?,
        epsilon: layer.epsilon,
        subtract_mean: layer.subtract_mean,
    })
}

fn create_convnext_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &StructuredAudioConvNeXtLayer,
    label_prefix: &str,
) -> AudioResult<StructuredAudioConvNeXtGpuLayer> {
    let channels = layer.depthwise_conv.cout;
    Ok(StructuredAudioConvNeXtGpuLayer {
        depthwise_conv: create_conv1d_gpu_layer(
            context,
            data_type,
            &layer.depthwise_conv,
            &format!("{label_prefix}_dwconv"),
        )?,
        norm: create_norm_gpu_layer(context, data_type, &layer.norm, channels, &format!("{label_prefix}_norm"))?,
        pwconv1: create_pointwise_conv_gpu_layer(
            context,
            data_type,
            &layer.pwconv1,
            &layer.pwconv1_bias,
            &format!("{label_prefix}_pwconv1"),
        )?,
        pwconv2: create_pointwise_conv_gpu_layer(
            context,
            data_type,
            &layer.pwconv2,
            &layer.pwconv2_bias,
            &format!("{label_prefix}_pwconv2"),
        )?,
    })
}

fn create_residual_unit_gpu_layer(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
    layer: &StructuredAudioResidualUnitLayer,
    channels: usize,
    label_prefix: &str,
) -> AudioResult<StructuredAudioResidualUnitGpuLayer> {
    Ok(StructuredAudioResidualUnitGpuLayer {
        snake1_alpha: create_alpha_gpu_array(
            context,
            data_type,
            channels,
            &layer.snake1_alpha,
            &format!("{label_prefix}_snake1"),
        )?,
        conv1: create_conv1d_gpu_layer(context, data_type, &layer.conv1, &format!("{label_prefix}_conv1"))?,
        snake2_alpha: create_alpha_gpu_array(
            context,
            data_type,
            channels,
            &layer.snake2_alpha,
            &format!("{label_prefix}_snake2"),
        )?,
        conv2: create_conv1d_gpu_layer(context, data_type, &layer.conv2, &format!("{label_prefix}_conv2"))?,
    })
}

fn fishaudio_kernels(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
) -> AudioResult<Rc<StructuredAudioKernelCache>> {
    let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(data_type));
    FISHAUDIO_KERNEL_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Ok(existing.clone());
        }

        let created = Rc::new(StructuredAudioKernelCache {
            transpose_nsc_to_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioTransposeNscToNcsKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize NSC->NCS transpose kernel: {err}")))?,
            half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize snake1d kernel: {err}")))?,
            causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped_residual:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize grouped residual conv1d kernel: {err}"))
                })?,
            causal_conv_transpose1d_causal_pad:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize causal-pad conv transpose kernel: {err}"))
                })?,
            conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize pointwise conv1d kernel: {err}")))?,
            norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize norm kernel: {err}")))?,
            activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(
                context.as_ref(),
                data_type,
                false,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize activation kernel: {err}")))?,
            add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize add kernel: {err}")))?,
        });

        cache.borrow_mut().insert(key, created.clone());
        Ok(created)
    })
}

