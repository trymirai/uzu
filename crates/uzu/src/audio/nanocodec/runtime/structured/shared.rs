use super::*;

pub(super) fn checked_add_usize(
    a: usize,
    b: usize,
    label: &str,
) -> AudioResult<usize> {
    a.checked_add(b).ok_or_else(|| AudioError::Runtime(format!("{label} overflow")))
}

pub(super) fn conv1d_estimated_macs<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    layer: &StructuredAudioConv1d<B>,
) -> AudioResult<usize> {
    let cin_per_group = layer
        .cin
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len, layer.cout, cin_per_group, layer.kernel_size])
}

pub(super) fn convtranspose_estimated_macs<B: Backend>(
    batch_size: usize,
    seq_len_in: usize,
    layer: &StructuredAudioConvTranspose1d<B>,
) -> AudioResult<usize> {
    let cout_per_group = layer
        .cout
        .checked_div(layer.groups)
        .ok_or_else(|| AudioError::Runtime("invalid grouped transpose-conv channel count".to_string()))?;
    checked_product(&[batch_size, seq_len_in, layer.cin, cout_per_group, layer.kernel_size])
}

pub(super) fn residual_unit_estimated_macs<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    unit: &StructuredAudioResidualUnit<B>,
) -> AudioResult<usize> {
    let conv1 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv1)?;
    let conv2 = conv1d_estimated_macs(batch_size, seq_len, &unit.conv2)?;
    checked_add_usize(conv1, conv2, "residual-unit estimated MACs")
}

pub(super) fn array_batch_view<B: Backend>(
    array: &Array<B>,
    batch_index: usize,
    frames: usize,
    channels: usize,
    active_frames: usize,
) -> AudioResult<Array<B>> {
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
pub(super) enum SequenceLayout {
    Ncs,
    Nsc,
}

impl SequenceLayout {
    pub(super) fn as_i32(self) -> i32 {
        match self {
            Self::Ncs => 0,
            Self::Nsc => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::audio::nanocodec::runtime) struct StructuredAudioCodecGraph {
    pub(in crate::audio::nanocodec::runtime) config: DescriptAudioCodecConfig,
    pub(in crate::audio::nanocodec::runtime) weights_path: String,
    pub(in crate::audio::nanocodec::runtime) codebook_size: usize,
    pub(in crate::audio::nanocodec::runtime) semantic_codebook_size: usize,
    pub(in crate::audio::nanocodec::runtime) input_dim: usize,
    pub(in crate::audio::nanocodec::runtime) total_codebooks: usize,
    pub(in crate::audio::nanocodec::runtime) upsample_factor: usize,
    pub(in crate::audio::nanocodec::runtime) vocoder_data_type: DataType,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioConv1d<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) dilation: usize,
    pub(super) groups: usize,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioConvTranspose1d<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) groups: usize,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioPointwiseConv<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioNorm<B: Backend> {
    pub(super) scales: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) epsilon: f32,
    pub(super) subtract_mean: bool,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioConvNeXt<B: Backend> {
    pub(super) depthwise_conv: StructuredAudioConv1d<B>,
    pub(super) norm: StructuredAudioNorm<B>,
    pub(super) pwconv1: StructuredAudioPointwiseConv<B>,
    pub(super) pwconv2: StructuredAudioPointwiseConv<B>,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioResidualUnit<B: Backend> {
    pub(super) snake1_alpha: Array<B>,
    pub(super) conv1: StructuredAudioConv1d<B>,
    pub(super) snake2_alpha: Array<B>,
    pub(super) conv2: StructuredAudioConv1d<B>,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioDecoderBlock<B: Backend> {
    pub(super) snake_alpha: Array<B>,
    pub(super) trans_conv: StructuredAudioConvTranspose1d<B>,
    pub(super) res_unit1: StructuredAudioResidualUnit<B>,
    pub(super) res_unit2: StructuredAudioResidualUnit<B>,
    pub(super) res_unit3: StructuredAudioResidualUnit<B>,
}

#[derive(Debug, Clone)]
pub(super) struct StructuredAudioDecoderGraph<B: Backend> {
    pub(super) first_conv: StructuredAudioConv1d<B>,
    pub(super) upsample_blocks: Vec<(StructuredAudioConvTranspose1d<B>, StructuredAudioConvNeXt<B>)>,
    pub(super) decoder_blocks: Vec<StructuredAudioDecoderBlock<B>>,
    pub(super) final_snake_alpha: Array<B>,
    pub(super) final_conv: StructuredAudioConv1d<B>,
}
