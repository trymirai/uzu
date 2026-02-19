use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                AudioAddKernel, AudioCausalConv1dKernel, AudioCausalConvTranspose1dKernel, AudioClampKernel,
                AudioConv1dKernel, AudioFsqDecodeKernel, AudioFsqEncodeKernel, AudioHalfSnakeKernel,
                AudioLeakyReluKernel, AudioScaleKernel, AudioTanhKernel,
                audio::{
                    AudioAddArguments, AudioCausalConv1dArguments, AudioCausalConvTranspose1dArguments,
                    AudioClampArguments, AudioConv1dArguments, AudioElementwiseArguments, AudioFsqDecodeArguments,
                    AudioFsqEncodeArguments, AudioHalfSnakeArguments, AudioKernelRuntime, AudioPadMode,
                    AudioScaleArguments,
                },
            },
        },
        metal::{
            Metal, MetalContext, MetalError,
            kernel::dsl::{
                AudioAddMetalKernel, AudioCausalConv1dMetalKernel, AudioCausalConvTranspose1dMetalKernel,
                AudioClampMetalKernel, AudioConv1dMetalKernel, AudioFsqDecodeMetalKernel, AudioFsqEncodeMetalKernel,
                AudioHalfSnakeMetalKernel, AudioLeakyReluMetalKernel, AudioScaleMetalKernel, AudioTanhMetalKernel,
            },
        },
    },
};

pub struct MetalAudioKernelRuntime {
    fsq_decode: AudioFsqDecodeMetalKernel,
    fsq_encode: AudioFsqEncodeMetalKernel,
    conv1d: AudioConv1dMetalKernel,
    causal_conv1d: AudioCausalConv1dMetalKernel,
    causal_conv_transpose1d: AudioCausalConvTranspose1dMetalKernel,
    half_snake: AudioHalfSnakeMetalKernel,
    leaky_relu: AudioLeakyReluMetalKernel,
    tanh: AudioTanhMetalKernel,
    add: AudioAddMetalKernel,
    scale: AudioScaleMetalKernel,
    clamp: AudioClampMetalKernel,
}

impl MetalAudioKernelRuntime {
    pub fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            fsq_decode: AudioFsqDecodeMetalKernel::new(context, data_type)?,
            fsq_encode: AudioFsqEncodeMetalKernel::new(context, data_type)?,
            conv1d: AudioConv1dMetalKernel::new(context, data_type)?,
            causal_conv1d: AudioCausalConv1dMetalKernel::new(context, data_type)?,
            causal_conv_transpose1d: AudioCausalConvTranspose1dMetalKernel::new(context, data_type)?,
            half_snake: AudioHalfSnakeMetalKernel::new(context, data_type)?,
            leaky_relu: AudioLeakyReluMetalKernel::new(context, data_type)?,
            tanh: AudioTanhMetalKernel::new(context, data_type)?,
            add: AudioAddMetalKernel::new(context, data_type)?,
            scale: AudioScaleMetalKernel::new(context, data_type)?,
            clamp: AudioClampMetalKernel::new(context, data_type)?,
        })
    }
}

impl AudioKernelRuntime<Metal> for MetalAudioKernelRuntime {
    fn encode_fsq_decode(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioFsqDecodeArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.fsq_decode.encode(
            arguments.tokens,
            arguments.output,
            arguments.lengths,
            arguments.num_groups as i32,
            arguments.seq_len as i32,
            arguments.codebook_dim_per_group as i32,
            arguments.num_levels_per_group,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_fsq_encode(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioFsqEncodeArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.fsq_encode.encode(
            arguments.input,
            arguments.tokens,
            arguments.lengths,
            arguments.num_groups as i32,
            arguments.seq_len as i32,
            arguments.codebook_dim_per_group as i32,
            arguments.num_levels_per_group,
            arguments.dim_base_index,
            arguments.eps,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_conv1d(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioConv1dArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        let pad_mode = match arguments.pad_mode {
            AudioPadMode::Zeros => 0_i32,
            AudioPadMode::Replicate => 1_i32,
        };

        self.conv1d.encode(
            arguments.input,
            arguments.weight,
            arguments.bias,
            arguments.output,
            arguments.lengths,
            arguments.cin as i32,
            arguments.cout as i32,
            arguments.seq_len_in as i32,
            arguments.seq_len_out as i32,
            arguments.kernel_size as i32,
            arguments.stride as i32,
            arguments.dilation as i32,
            arguments.padding as i32,
            pad_mode,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_causal_conv1d(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioCausalConv1dArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.causal_conv1d.encode(
            arguments.input,
            arguments.weight,
            arguments.bias,
            arguments.output,
            arguments.lengths,
            arguments.cin as i32,
            arguments.cout as i32,
            arguments.seq_len as i32,
            arguments.kernel_size as i32,
            arguments.dilation as i32,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_causal_conv_transpose1d(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioCausalConvTranspose1dArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.causal_conv_transpose1d.encode(
            arguments.input,
            arguments.weight,
            arguments.bias,
            arguments.output,
            arguments.lengths,
            arguments.cin as i32,
            arguments.cout as i32,
            arguments.seq_len_in as i32,
            arguments.seq_len_out as i32,
            arguments.stride as i32,
            arguments.groups as i32,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_half_snake(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioHalfSnakeArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.half_snake.encode(
            arguments.input,
            arguments.alpha,
            arguments.output,
            arguments.channels as i32,
            arguments.seq_len as i32,
            arguments.snake_channels as i32,
            arguments.negative_slope,
            arguments.eps,
            arguments.batch_size as i32,
            compute_encoder,
        );

        Ok(())
    }

    fn encode_leaky_relu(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioElementwiseArguments<'_, Metal>,
        negative_slope: f32,
    ) -> Result<(), MetalError> {
        self.leaky_relu.encode(arguments.input, arguments.output, arguments.n as i32, negative_slope, compute_encoder);

        Ok(())
    }

    fn encode_tanh(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioElementwiseArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.tanh.encode(arguments.input, arguments.output, arguments.n as i32, compute_encoder);

        Ok(())
    }

    fn encode_add(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioAddArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.add.encode(arguments.a, arguments.b, arguments.output, arguments.n as i32, compute_encoder);

        Ok(())
    }

    fn encode_scale(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioScaleArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.scale.encode(arguments.input, arguments.output, arguments.n as i32, arguments.scale, compute_encoder);

        Ok(())
    }

    fn encode_clamp(
        &self,
        compute_encoder: &<Metal as Backend>::ComputeEncoder,
        arguments: AudioClampArguments<'_, Metal>,
    ) -> Result<(), MetalError> {
        self.clamp.encode(
            arguments.input,
            arguments.output,
            arguments.n as i32,
            arguments.min_value,
            arguments.max_value,
            compute_encoder,
        );

        Ok(())
    }
}
