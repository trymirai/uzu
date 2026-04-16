use super::*;

pub(super) fn allocation_batch_view<B: Backend>(
    allocation: &crate::backends::common::Allocation<B>,
    batch_index: usize,
    frames: usize,
    channels: usize,
    active_frames: usize,
    data_type: DataType,
) -> AudioResult<crate::backends::common::Allocation<B>> {
    let batch_stride_bytes = size_for_shape(&[frames, channels], data_type);
    let batch_offset = batch_index
        .checked_mul(batch_stride_bytes)
        .ok_or(AudioError::Runtime("allocation batch view offset overflow".to_string()))?;
    if active_frames > frames {
        return Err(AudioError::Runtime("allocation batch view active_frames exceeds frames".to_string()));
    }
    Ok(allocation.slice(batch_offset..batch_offset + size_for_shape(&[active_frames, channels], data_type)))
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
pub struct StructuredAudioCodecGraph {
    pub(in crate::audio::nanocodec::runtime) config: DescriptAudioCodecConfig,
    pub(in crate::audio::nanocodec::runtime) weights_path: String,
    pub(in crate::audio::nanocodec::runtime) codebook_size: usize,
    pub(in crate::audio::nanocodec::runtime) semantic_codebook_size: usize,
    pub(in crate::audio::nanocodec::runtime) input_dim: usize,
    pub(in crate::audio::nanocodec::runtime) total_codebooks: usize,
    pub(in crate::audio::nanocodec::runtime) upsample_factor: usize,
    pub(in crate::audio::nanocodec::runtime) vocoder_data_type: DataType,
}

#[derive(Clone)]
pub(super) struct StructuredAudioConv1d<B: Backend> {
    pub(super) weight: crate::backends::common::Allocation<B>,
    pub(super) bias: crate::backends::common::Allocation<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) dilation: usize,
    pub(super) groups: usize,
}

#[derive(Clone)]
pub(super) struct StructuredAudioConvTranspose1d<B: Backend> {
    pub(super) weight: crate::backends::common::Allocation<B>,
    pub(super) bias: crate::backends::common::Allocation<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) groups: usize,
}

#[derive(Clone)]
pub(super) struct StructuredAudioPointwiseConv<B: Backend> {
    pub(super) weight: crate::backends::common::Allocation<B>,
    pub(super) bias: crate::backends::common::Allocation<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
}

#[derive(Clone)]
pub(super) struct StructuredAudioNorm<B: Backend> {
    pub(super) scales: crate::backends::common::Allocation<B>,
    pub(super) bias: crate::backends::common::Allocation<B>,
    pub(super) epsilon: f32,
    pub(super) subtract_mean: bool,
}

#[derive(Clone)]
pub(super) struct StructuredAudioConvNeXt<B: Backend> {
    pub(super) depthwise_conv: StructuredAudioConv1d<B>,
    pub(super) norm: StructuredAudioNorm<B>,
    pub(super) pwconv1: StructuredAudioPointwiseConv<B>,
    pub(super) pwconv2: StructuredAudioPointwiseConv<B>,
}

#[derive(Clone)]
pub(super) struct StructuredAudioResidualUnit<B: Backend> {
    pub(super) snake1_alpha: crate::backends::common::Allocation<B>,
    pub(super) conv1: StructuredAudioConv1d<B>,
    pub(super) snake2_alpha: crate::backends::common::Allocation<B>,
    pub(super) conv2: StructuredAudioConv1d<B>,
}

#[derive(Clone)]
pub(super) struct StructuredAudioDecoderBlock<B: Backend> {
    pub(super) snake_alpha: crate::backends::common::Allocation<B>,
    pub(super) trans_conv: StructuredAudioConvTranspose1d<B>,
    pub(super) res_unit1: StructuredAudioResidualUnit<B>,
    pub(super) res_unit2: StructuredAudioResidualUnit<B>,
    pub(super) res_unit3: StructuredAudioResidualUnit<B>,
}

#[derive(Clone)]
pub(super) struct StructuredAudioDecoderGraph<B: Backend> {
    pub(super) first_conv: StructuredAudioConv1d<B>,
    pub(super) upsample_blocks: Vec<(StructuredAudioConvTranspose1d<B>, StructuredAudioConvNeXt<B>)>,
    pub(super) decoder_blocks: Vec<StructuredAudioDecoderBlock<B>>,
    pub(super) final_snake_alpha: crate::backends::common::Allocation<B>,
    pub(super) final_conv: StructuredAudioConv1d<B>,
}
