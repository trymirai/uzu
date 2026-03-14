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
    layer: &StructuredAudioConv1dGpuLayer,
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
    layer: &StructuredAudioConvTranspose1dGpuLayer,
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
    unit: &StructuredAudioResidualUnitGpuLayer,
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
struct StructuredAudioVectorQuantizer {
    codebook: Vec<f32>,
    codebook_dim: usize,
    out_proj: Vec<f32>,
    out_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioConv1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioConvTranspose1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioNormLayer {
    scales: Vec<f32>,
    biases: Option<Vec<f32>>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioConvNeXtLayer {
    depthwise_conv: StructuredAudioConv1dLayer,
    norm: StructuredAudioNormLayer,
    pwconv1: Vec<f32>,
    pwconv1_hidden_dim: usize,
    pwconv1_bias: Vec<f32>,
    pwconv2: Vec<f32>,
    pwconv2_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioResidualUnitLayer {
    snake1_alpha: Vec<f32>,
    conv1: StructuredAudioConv1dLayer,
    snake2_alpha: Vec<f32>,
    conv2: StructuredAudioConv1dLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioDecoderBlockLayer {
    snake_alpha: Vec<f32>,
    trans_conv: StructuredAudioConvTranspose1dLayer,
    res_unit1: StructuredAudioResidualUnitLayer,
    res_unit2: StructuredAudioResidualUnitLayer,
    res_unit3: StructuredAudioResidualUnitLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioDecoderGraph {
    first_conv: StructuredAudioConv1dLayer,
    upsample_blocks: Vec<(StructuredAudioConvTranspose1dLayer, StructuredAudioConvNeXtLayer)>,
    decoder_blocks: Vec<StructuredAudioDecoderBlockLayer>,
    final_snake_alpha: Vec<f32>,
    final_conv: StructuredAudioConv1dLayer,
    upsample_factor: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredAudioCodecGraph {
    semantic_quantizer: StructuredAudioVectorQuantizer,
    residual_quantizers: Vec<StructuredAudioVectorQuantizer>,
    post_module_transformer_config: crate::config::TransformerConfig,
    weights_path: String,
    decoder: StructuredAudioDecoderGraph,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    total_codebooks: usize,
    upsample_factor: usize,
    vocoder_data_type: DataType,
}

#[derive(Debug, Clone)]
struct StructuredAudioConv1dGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioConvTranspose1dGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioPointwiseConvGpuLayer {
    weight: Array<Metal>,
    bias: Array<Metal>,
    cin: usize,
    cout: usize,
}

#[derive(Debug, Clone)]
struct StructuredAudioNormGpuLayer {
    scales: Array<Metal>,
    bias: Array<Metal>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone)]
struct StructuredAudioConvNeXtGpuLayer {
    depthwise_conv: StructuredAudioConv1dGpuLayer,
    norm: StructuredAudioNormGpuLayer,
    pwconv1: StructuredAudioPointwiseConvGpuLayer,
    pwconv2: StructuredAudioPointwiseConvGpuLayer,
}

#[derive(Debug, Clone)]
struct StructuredAudioResidualUnitGpuLayer {
    snake1_alpha: Array<Metal>,
    conv1: StructuredAudioConv1dGpuLayer,
    snake2_alpha: Array<Metal>,
    conv2: StructuredAudioConv1dGpuLayer,
}

#[derive(Debug, Clone)]
struct StructuredAudioDecoderBlockGpuLayer {
    snake_alpha: Array<Metal>,
    trans_conv: StructuredAudioConvTranspose1dGpuLayer,
    res_unit1: StructuredAudioResidualUnitGpuLayer,
    res_unit2: StructuredAudioResidualUnitGpuLayer,
    res_unit3: StructuredAudioResidualUnitGpuLayer,
}

#[derive(Debug, Clone)]
struct StructuredAudioDecoderGpuGraph {
    first_conv: StructuredAudioConv1dGpuLayer,
    upsample_blocks: Vec<(StructuredAudioConvTranspose1dGpuLayer, StructuredAudioConvNeXtGpuLayer)>,
    decoder_blocks: Vec<StructuredAudioDecoderBlockGpuLayer>,
    final_snake_alpha: Array<Metal>,
    final_conv: StructuredAudioConv1dGpuLayer,
}
