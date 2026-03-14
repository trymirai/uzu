fn checked_add_usize(
    a: usize,
    b: usize,
    label: &str,
) -> AudioResult<usize> {
    a.checked_add(b).ok_or_else(|| AudioError::Runtime(format!("{label} overflow")))
}

fn conv1d_estimated_macs(
    batch_size: usize,
    seq_len: usize,
    layer: &StructuredAudioConv1d,
) -> AudioResult<usize> {
    let cin_per_group = layer
        .cin
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len, layer.cout, cin_per_group, layer.kernel_size])
}

fn convtranspose_estimated_macs(
    batch_size: usize,
    seq_len_in: usize,
    layer: &StructuredAudioConvTranspose1d,
) -> AudioResult<usize> {
    let cout_per_group = layer
        .cout
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped transpose-conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len_in, layer.cin, cout_per_group, layer.kernel_size])
}

fn residual_unit_estimated_macs(
    batch_size: usize,
    seq_len: usize,
    unit: &StructuredAudioResidualUnit,
) -> AudioResult<usize> {
    let conv1 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv1)?;
    let conv2 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv2)?;
    checked_add_usize(conv1, conv2, "residual-unit estimated MACs")
}

fn array_batch_view(
    array: &Array<Metal>,
    batch_index: usize,
    frames: usize,
    channels: usize,
    active_frames: usize,
) -> AudioResult<Array<Metal>> {
    let batch_stride_bytes = size_for_shape(&[frames, channels], array.data_type());
    let batch_offset = batch_index
        .checked_mul(batch_stride_bytes)
        .and_then(|value| value.checked_add(array.offset()))
        .ok_or(AudioError::Runtime("array batch view offset overflow".to_string()))?;
    if active_frames > frames {
        return Err(AudioError::Runtime("array batch view active_frames exceeds frames".to_string()));
    }
    Ok(unsafe { Array::from_parts(array.buffer(), batch_offset, &[active_frames, channels], array.data_type()) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceLayout {
    Ncs,
    Nsc,
}

impl SequenceLayout {
    fn as_i32(self) -> i32 {
        match self {
            Self::Ncs => 0,
            Self::Nsc => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioCodecGraph {
    config: DescriptAudioCodecConfig,
    weights_path: String,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    total_codebooks: usize,
    upsample_factor: usize,
    vocoder_data_type: DataType,
}

#[derive(Debug, Clone)]
struct StructuredAudioConv1d {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioConvTranspose1d {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioPointwiseConv {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioNorm {
    scales: Array<Metal>,
    bias: Array<Metal>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone)]
struct StructuredAudioConvNeXt {
    depthwise_conv: StructuredAudioConv1d,
    norm: StructuredAudioNorm,
    pwconv1: StructuredAudioPointwiseConv,
    pwconv2: StructuredAudioPointwiseConv,
}

#[derive(Debug, Clone)]
struct StructuredAudioResidualUnit {
    snake1_alpha: Array<Metal>,
    conv1: StructuredAudioConv1d,
    snake2_alpha: Array<Metal>,
    conv2: StructuredAudioConv1d,
}

#[derive(Debug, Clone)]
struct StructuredAudioDecoderBlock {
    snake_alpha: Array<Metal>,
    trans_conv: StructuredAudioConvTranspose1d,
    res_unit1: StructuredAudioResidualUnit,
    res_unit2: StructuredAudioResidualUnit,
    res_unit3: StructuredAudioResidualUnit,
}

#[derive(Debug, Clone)]
struct StructuredAudioDecoderGraph {
    first_conv: StructuredAudioConv1d,
    upsample_blocks: Vec<(StructuredAudioConvTranspose1d, StructuredAudioConvNeXt)>,
    decoder_blocks: Vec<StructuredAudioDecoderBlock>,
    final_snake_alpha: Array<Metal>,
    final_conv: StructuredAudioConv1d,
}
