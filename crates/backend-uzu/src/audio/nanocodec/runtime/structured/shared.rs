use std::ops::Range;

use super::*;
use crate::array::Array;

pub(super) fn array_batch_byte_range<B: Backend>(
    array: &Array<B>,
    batch_index: usize,
    frames: usize,
    channels: usize,
    active_frames: usize,
) -> AudioResult<Range<usize>> {
    let batch_stride_bytes = size_for_shape(&[frames, channels], array.data_type());
    let batch_offset = batch_index
        .checked_mul(batch_stride_bytes)
        .and_then(|value| value.checked_add(array.offset()))
        .ok_or(AudioError::Runtime("array batch range offset overflow".to_string()))?;
    if active_frames > frames {
        return Err(AudioError::Runtime("array batch range active_frames exceeds frames".to_string()));
    }
    let byte_len = size_for_shape(&[active_frames, channels], array.data_type());
    let end =
        batch_offset.checked_add(byte_len).ok_or(AudioError::Runtime("array batch range end overflow".to_string()))?;
    Ok(batch_offset..end)
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

#[derive(Debug)]
pub(super) struct StructuredAudioConv1d<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) dilation: usize,
    pub(super) groups: usize,
}

#[derive(Debug)]
pub(super) struct StructuredAudioConvTranspose1d<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) groups: usize,
}

#[derive(Debug)]
pub(super) struct StructuredAudioPointwiseConv<B: Backend> {
    pub(super) weight: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) cin: usize,
    pub(super) cout: usize,
}

#[derive(Debug)]
pub(super) struct StructuredAudioNorm<B: Backend> {
    pub(super) scales: Array<B>,
    pub(super) bias: Array<B>,
    pub(super) epsilon: f32,
    pub(super) subtract_mean: bool,
}

#[derive(Debug)]
pub(super) struct StructuredAudioConvNeXt<B: Backend> {
    pub(super) depthwise_conv: StructuredAudioConv1d<B>,
    pub(super) norm: StructuredAudioNorm<B>,
    pub(super) pwconv1: StructuredAudioPointwiseConv<B>,
    pub(super) pwconv2: StructuredAudioPointwiseConv<B>,
}

#[derive(Debug)]
pub(super) struct StructuredAudioResidualUnit<B: Backend> {
    pub(super) snake1_alpha: Array<B>,
    pub(super) conv1: StructuredAudioConv1d<B>,
    pub(super) snake2_alpha: Array<B>,
    pub(super) conv2: StructuredAudioConv1d<B>,
}

#[derive(Debug)]
pub(super) struct StructuredAudioDecoderBlock<B: Backend> {
    pub(super) snake_alpha: Array<B>,
    pub(super) trans_conv: StructuredAudioConvTranspose1d<B>,
    pub(super) res_unit1: StructuredAudioResidualUnit<B>,
    pub(super) res_unit2: StructuredAudioResidualUnit<B>,
    pub(super) res_unit3: StructuredAudioResidualUnit<B>,
}

#[derive(Debug)]
pub(super) struct StructuredAudioDecoderGraph<B: Backend> {
    pub(super) first_conv: StructuredAudioConv1d<B>,
    pub(super) upsample_blocks: Vec<(StructuredAudioConvTranspose1d<B>, StructuredAudioConvNeXt<B>)>,
    pub(super) decoder_blocks: Vec<StructuredAudioDecoderBlock<B>>,
    pub(super) final_snake_alpha: Array<B>,
    pub(super) final_conv: StructuredAudioConv1d<B>,
}
