use crate::backends::common::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPadMode {
    Zeros,
    Replicate,
}

pub struct AudioFsqDecodeArguments<'a, B: Backend> {
    pub tokens: &'a B::NativeBuffer,
    pub lengths: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: &'a [i32],
}

pub struct AudioFsqEncodeArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub tokens: &'a B::NativeBuffer,
    pub lengths: &'a B::NativeBuffer,
    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: &'a [i32],
    pub dim_base_index: &'a [i32],
    pub eps: f32,
}

pub struct AudioConv1dArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub weight: &'a B::NativeBuffer,
    pub bias: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub lengths: &'a B::NativeBuffer,
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
    pub input: &'a B::NativeBuffer,
    pub weight: &'a B::NativeBuffer,
    pub bias: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub lengths: &'a B::NativeBuffer,
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

pub struct AudioCausalConvTranspose1dArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub weight: &'a B::NativeBuffer,
    pub bias: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub lengths: &'a B::NativeBuffer,
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub stride: usize,
    pub groups: usize,
}

pub struct AudioHalfSnakeArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub alpha: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub batch_size: usize,
    pub channels: usize,
    pub seq_len: usize,
    pub snake_channels: usize,
    pub negative_slope: f32,
    pub eps: f32,
}

pub struct AudioElementwiseArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub n: usize,
}

pub struct AudioAddArguments<'a, B: Backend> {
    pub a: &'a B::NativeBuffer,
    pub b: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub n: usize,
}

pub struct AudioScaleArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub n: usize,
    pub scale: f32,
}

pub struct AudioClampArguments<'a, B: Backend> {
    pub input: &'a B::NativeBuffer,
    pub output: &'a B::NativeBuffer,
    pub n: usize,
    pub min_value: f32,
    pub max_value: f32,
}

pub trait AudioKernelRuntime<B: Backend> {
    fn encode_fsq_decode(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioFsqDecodeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_fsq_encode(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioFsqEncodeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_conv1d(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioConv1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_causal_conv1d(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioCausalConv1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_causal_conv_transpose1d(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioCausalConvTranspose1dArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_half_snake(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioHalfSnakeArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_leaky_relu(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioElementwiseArguments<'_, B>,
        negative_slope: f32,
    ) -> Result<(), B::Error>;

    fn encode_tanh(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioElementwiseArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_add(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioAddArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_scale(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioScaleArguments<'_, B>,
    ) -> Result<(), B::Error>;

    fn encode_clamp(
        &self,
        compute_encoder: &B::ComputeEncoder,
        arguments: AudioClampArguments<'_, B>,
    ) -> Result<(), B::Error>;
}
