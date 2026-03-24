use crate::backends::common::{Backend, Encoder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPadMode {
    Zeros,
    Replicate,
}

pub struct AudioFsqDecodeArguments<'a, B: Backend> {
    pub tokens: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: &'a [i32],
}

pub struct AudioFsqEncodeArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub tokens: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: &'a [i32],
    pub dim_base_index: &'a [i32],
    pub eps: f32,
}

pub struct AudioConv1dArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub weight: &'a B::Buffer,
    pub bias: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub pad_mode: AudioPadMode,
}

pub struct AudioCausalConv1dArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub weight: &'a B::Buffer,
    pub bias: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

pub struct AudioCausalConvTranspose1dArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub weight: &'a B::Buffer,
    pub bias: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub lengths: &'a B::Buffer,
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub stride: usize,
    pub groups: usize,
}

pub struct AudioHalfSnakeArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub alpha: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub batch_size: usize,
    pub channels: usize,
    pub seq_len: usize,
    pub snake_channels: usize,
    pub negative_slope: f32,
    pub eps: f32,
}

pub struct AudioElementwiseArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub n: usize,
}

pub struct AudioAddArguments<'a, B: Backend> {
    pub a: &'a B::Buffer,
    pub b: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub n: usize,
}

pub struct AudioScaleArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub n: usize,
    pub scale: f32,
}

pub struct AudioClampArguments<'a, B: Backend> {
    pub input: &'a B::Buffer,
    pub output: &'a B::Buffer,
    pub n: usize,
    pub min_value: f32,
    pub max_value: f32,
}

pub trait AudioKernelRuntime<B: Backend> {
    fn encode_fsq_decode(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioFsqDecodeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_fsq_encode(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioFsqEncodeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_conv1d(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioConv1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_causal_conv1d(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioCausalConv1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_causal_conv_transpose1d(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioCausalConvTranspose1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_half_snake(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioHalfSnakeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_leaky_relu(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioElementwiseArguments<'_, B>,
        negative_slope: f32,
    ) -> Result<(), B::Error>;

    fn encode_tanh(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioElementwiseArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_add(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioAddArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_scale(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioScaleArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_clamp(
        &self,
        encoder: &mut Encoder<B>,
        arguments: AudioClampArguments<'_, B>,
    ) -> Result<(), B::Error>;
}
