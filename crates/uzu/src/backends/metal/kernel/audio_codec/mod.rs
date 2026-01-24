use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState,
    MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

#[derive(Debug, thiserror::Error)]
pub enum AudioCodecKernelError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(&'static str),
}

pub struct FsqDecodeKernel {
    pipeline: ComputePipelineState,
}

pub struct FsqDecodeArguments<'a> {
    pub tokens: &'a MTLBuffer,  // [B, G, T] int32
    pub lengths: &'a MTLBuffer, // [B] int32
    pub out: &'a MTLBuffer,     // [B, G*D, T] float/half/bfloat

    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: Box<[i32]>,
}

impl FsqDecodeKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!("fsq_decode_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: FsqDecodeArguments<'_>,
    ) -> Result<(), AudioCodecKernelError> {
        if args.batch_size == 0 || args.num_groups == 0 || args.seq_len == 0 {
            return Ok(());
        }
        if args.codebook_dim_per_group == 0 {
            return Err(AudioCodecKernelError::InvalidConfig(
                "codebook_dim_per_group must be > 0",
            ));
        }
        if args.num_levels_per_group.len() != args.codebook_dim_per_group {
            return Err(AudioCodecKernelError::InvalidConfig(
                "num_levels_per_group length must equal codebook_dim_per_group",
            ));
        }

        encoder.set_label("FSQ Decode");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(args.tokens), 0);
        encoder.set_buffer(1, Some(args.out), 0);
        encoder.set_buffer(2, Some(args.lengths), 0);

        let num_groups_i32: i32 = args.num_groups as i32;
        let seq_len_i32: i32 = args.seq_len as i32;
        let codebook_dim_i32: i32 = args.codebook_dim_per_group as i32;

        encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &num_groups_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &seq_len_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &codebook_dim_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            (args.num_levels_per_group.len() * size_of::<i32>()) as u64,
            args.num_levels_per_group.as_ptr() as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let threads_per_grid = MTLSize {
            width: args.seq_len as u64,
            height: args.num_groups as u64,
            depth: args.batch_size as u64,
        };

        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        Ok(())
    }
}

pub struct FsqEncodeKernel {
    pipeline: ComputePipelineState,
}

pub struct FsqEncodeArguments<'a> {
    pub input: &'a MTLBuffer,   // [B, G*D, T] float
    pub tokens: &'a MTLBuffer,  // [B, G, T] int32
    pub lengths: &'a MTLBuffer, // [B] int32

    pub batch_size: usize,
    pub num_groups: usize,
    pub seq_len: usize,
    pub codebook_dim_per_group: usize,
    pub num_levels_per_group: Box<[i32]>,
    pub dim_base_index: Box<[i32]>,
    pub eps: f32,
}

impl FsqEncodeKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!("fsq_encode_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: FsqEncodeArguments<'_>,
    ) -> Result<(), AudioCodecKernelError> {
        if args.batch_size == 0 || args.num_groups == 0 || args.seq_len == 0 {
            return Ok(());
        }
        if args.codebook_dim_per_group == 0 {
            return Err(AudioCodecKernelError::InvalidConfig(
                "codebook_dim_per_group must be > 0",
            ));
        }
        if args.num_levels_per_group.len() != args.codebook_dim_per_group {
            return Err(AudioCodecKernelError::InvalidConfig(
                "num_levels_per_group length must equal codebook_dim_per_group",
            ));
        }
        if args.dim_base_index.len() != args.codebook_dim_per_group {
            return Err(AudioCodecKernelError::InvalidConfig(
                "dim_base_index length must equal codebook_dim_per_group",
            ));
        }

        encoder.set_label("FSQ Encode");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(args.input), 0);
        encoder.set_buffer(1, Some(args.tokens), 0);
        encoder.set_buffer(2, Some(args.lengths), 0);

        let num_groups_i32: i32 = args.num_groups as i32;
        let seq_len_i32: i32 = args.seq_len as i32;
        let codebook_dim_i32: i32 = args.codebook_dim_per_group as i32;

        encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &num_groups_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &seq_len_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &codebook_dim_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            (args.num_levels_per_group.len() * size_of::<i32>()) as u64,
            args.num_levels_per_group.as_ptr() as *const _,
        );
        encoder.set_bytes(
            7,
            (args.dim_base_index.len() * size_of::<i32>()) as u64,
            args.dim_base_index.as_ptr() as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.eps as *const f32 as *const _,
        );

        encoder.dispatch_threads(
            MTLSize::new(args.seq_len as u64, args.num_groups as u64, args.batch_size as u64),
            MTLSize::new(32, 1, 1),
        );
        Ok(())
    }
}

pub struct LeakyReluKernel {
    pipeline: ComputePipelineState,
}

impl LeakyReluKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!(
            "audio_codec_leaky_relu_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &MTLBuffer,
        output: &MTLBuffer,
        n: usize,
        negative_slope: f32,
    ) -> Result<(), AudioCodecKernelError> {
        if n == 0 {
            return Ok(());
        }
        encoder.set_label("Audio LeakyReLU");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &negative_slope as *const f32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }
}

pub struct TanhKernel {
    pipeline: ComputePipelineState,
}

impl TanhKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name =
            format!("audio_codec_tanh_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &MTLBuffer,
        output: &MTLBuffer,
        n: usize,
    ) -> Result<(), AudioCodecKernelError> {
        if n == 0 {
            return Ok(());
        }
        encoder.set_label("Audio Tanh");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }
}

pub struct AddKernel {
    pipeline: ComputePipelineState,
}

impl AddKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name =
            format!("audio_codec_add_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a: &MTLBuffer,
        b: &MTLBuffer,
        out: &MTLBuffer,
        n: usize,
    ) -> Result<(), AudioCodecKernelError> {
        if n == 0 {
            return Ok(());
        }
        encoder.set_label("Audio Add");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }
}

pub struct ScaleKernel {
    pipeline: ComputePipelineState,
}

impl ScaleKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name =
            format!("audio_codec_scale_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &MTLBuffer,
        output: &MTLBuffer,
        n: usize,
        scale: f32,
    ) -> Result<(), AudioCodecKernelError> {
        if n == 0 {
            return Ok(());
        }
        encoder.set_label("Audio Scale");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &scale as *const f32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }
}

pub struct CausalConv1dKernel {
    pipeline: ComputePipelineState,
}

pub struct CausalConv1dArguments<'a> {
    pub input: &'a MTLBuffer,   // [B, Cin, T]
    pub weight: &'a MTLBuffer,  // [Cout, Cin, K]
    pub bias: &'a MTLBuffer,    // [Cout]
    pub output: &'a MTLBuffer,  // [B, Cout, T]
    pub lengths: &'a MTLBuffer, // [B]

    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

impl CausalConv1dKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!(
            "audio_codec_causal_conv1d_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: CausalConv1dArguments<'_>,
    ) -> Result<(), AudioCodecKernelError> {
        if args.batch_size == 0 || args.cout == 0 || args.seq_len == 0 {
            return Ok(());
        }
        if args.cin == 0 || args.kernel_size == 0 || args.dilation == 0 {
            return Err(AudioCodecKernelError::InvalidConfig(
                "cin/kernel_size/dilation must be > 0",
            ));
        }

        encoder.set_label("Audio CausalConv1d");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(args.input), 0);
        encoder.set_buffer(1, Some(args.weight), 0);
        encoder.set_buffer(2, Some(args.bias), 0);
        encoder.set_buffer(3, Some(args.output), 0);
        encoder.set_buffer(4, Some(args.lengths), 0);

        let cin_i32 = args.cin as i32;
        let cout_i32 = args.cout as i32;
        let seq_len_i32 = args.seq_len as i32;
        let kernel_size_i32 = args.kernel_size as i32;
        let dilation_i32 = args.dilation as i32;

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &cin_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &cout_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &seq_len_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &kernel_size_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &dilation_i32 as *const i32 as *const _,
        );

        encoder.dispatch_threads(
            MTLSize::new(args.seq_len as u64, args.cout as u64, args.batch_size as u64),
            MTLSize::new(32, 1, 1),
        );
        Ok(())
    }
}

pub struct CausalConvTranspose1dKernel {
    pipeline: ComputePipelineState,
}

pub struct CausalConvTranspose1dArguments<'a> {
    pub input: &'a MTLBuffer,   // [B, Cin, Tin]
    pub weight: &'a MTLBuffer,  // [Cin, Cout/groups, 2*stride]
    pub bias: &'a MTLBuffer,    // [Cout]
    pub output: &'a MTLBuffer,  // [B, Cout, Tout]
    pub lengths: &'a MTLBuffer, // [B] output lengths

    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub stride: usize,
    pub groups: usize,
}

impl CausalConvTranspose1dKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!(
            "audio_codec_causal_conv_transpose1d_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: CausalConvTranspose1dArguments<'_>,
    ) -> Result<(), AudioCodecKernelError> {
        if args.batch_size == 0 || args.cout == 0 || args.seq_len_out == 0 {
            return Ok(());
        }
        if args.cin == 0
            || args.seq_len_in == 0
            || args.stride == 0
            || args.groups == 0
        {
            return Err(AudioCodecKernelError::InvalidConfig(
                "cin/seq_len_in/stride/groups must be > 0",
            ));
        }
        if args.cin % args.groups != 0 || args.cout % args.groups != 0 {
            return Err(AudioCodecKernelError::InvalidConfig(
                "cin and cout must be divisible by groups",
            ));
        }

        encoder.set_label("Audio CausalConvTranspose1d");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(args.input), 0);
        encoder.set_buffer(1, Some(args.weight), 0);
        encoder.set_buffer(2, Some(args.bias), 0);
        encoder.set_buffer(3, Some(args.output), 0);
        encoder.set_buffer(4, Some(args.lengths), 0);

        let cin_i32 = args.cin as i32;
        let cout_i32 = args.cout as i32;
        let seq_len_in_i32 = args.seq_len_in as i32;
        let seq_len_out_i32 = args.seq_len_out as i32;
        let stride_i32 = args.stride as i32;
        let groups_i32 = args.groups as i32;

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &cin_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &cout_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &seq_len_in_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &seq_len_out_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &stride_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &groups_i32 as *const i32 as *const _,
        );

        encoder.dispatch_threads(
            MTLSize::new(args.seq_len_out as u64, args.cout as u64, args.batch_size as u64),
            MTLSize::new(32, 1, 1),
        );
        Ok(())
    }
}

pub struct HalfSnakeKernel {
    pipeline: ComputePipelineState,
}

impl HalfSnakeKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!(
            "audio_codec_half_snake_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &MTLBuffer,
        alpha: &MTLBuffer,
        output: &MTLBuffer,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
        snake_channels: usize,
        negative_slope: f32,
        eps: f32,
    ) -> Result<(), AudioCodecKernelError> {
        if batch_size == 0 || channels == 0 || seq_len == 0 {
            return Ok(());
        }
        if snake_channels > channels {
            return Err(AudioCodecKernelError::InvalidConfig(
                "snake_channels must be <= channels",
            ));
        }

        encoder.set_label("Audio HalfSnake");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(alpha), 0);
        encoder.set_buffer(2, Some(output), 0);

        let channels_i32 = channels as i32;
        let seq_len_i32 = seq_len as i32;
        let snake_channels_i32 = snake_channels as i32;

        encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &channels_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &seq_len_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &snake_channels_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<f32>() as u64,
            &negative_slope as *const f32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        encoder.dispatch_threads(
            MTLSize::new(seq_len as u64, channels as u64, batch_size as u64),
            MTLSize::new(32, 1, 1),
        );
        Ok(())
    }
}

pub struct ClampKernel {
    pipeline: ComputePipelineState,
}

pub struct Conv1dKernel {
    pipeline: ComputePipelineState,
}

pub struct Conv1dArguments<'a> {
    pub input: &'a MTLBuffer,   // [B, Cin, Tin]
    pub weight: &'a MTLBuffer,  // [Cout, Cin, K]
    pub bias: &'a MTLBuffer,    // [Cout]
    pub output: &'a MTLBuffer,  // [B, Cout, Tout]
    pub lengths: &'a MTLBuffer, // [B] Tout lengths

    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    /// 0 = zeros, 1 = replicate
    pub pad_mode: usize,
}

impl Conv1dKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name = format!(
            "audio_codec_conv1d_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: Conv1dArguments<'_>,
    ) -> Result<(), AudioCodecKernelError> {
        if args.batch_size == 0 || args.cout == 0 || args.seq_len_out == 0 {
            return Ok(());
        }
        if args.cin == 0
            || args.seq_len_in == 0
            || args.kernel_size == 0
            || args.stride == 0
            || args.dilation == 0
        {
            return Err(AudioCodecKernelError::InvalidConfig(
                "cin/seq_len_in/kernel_size/stride/dilation must be > 0",
            ));
        }
        if args.pad_mode > 1 {
            return Err(AudioCodecKernelError::InvalidConfig(
                "pad_mode must be 0 (zeros) or 1 (replicate)",
            ));
        }

        encoder.set_label("Audio Conv1d");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(args.input), 0);
        encoder.set_buffer(1, Some(args.weight), 0);
        encoder.set_buffer(2, Some(args.bias), 0);
        encoder.set_buffer(3, Some(args.output), 0);
        encoder.set_buffer(4, Some(args.lengths), 0);

        let cin_i32 = args.cin as i32;
        let cout_i32 = args.cout as i32;
        let seq_len_in_i32 = args.seq_len_in as i32;
        let seq_len_out_i32 = args.seq_len_out as i32;
        let kernel_size_i32 = args.kernel_size as i32;
        let stride_i32 = args.stride as i32;
        let dilation_i32 = args.dilation as i32;
        let padding_i32 = args.padding as i32;
        let pad_mode_i32 = args.pad_mode as i32;

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &cin_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &cout_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &seq_len_in_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &seq_len_out_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &kernel_size_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &stride_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            11,
            size_of::<i32>() as u64,
            &dilation_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            12,
            size_of::<i32>() as u64,
            &padding_i32 as *const i32 as *const _,
        );
        encoder.set_bytes(
            13,
            size_of::<i32>() as u64,
            &pad_mode_i32 as *const i32 as *const _,
        );

        encoder.dispatch_threads(
            MTLSize::new(args.seq_len_out as u64, args.cout as u64, args.batch_size as u64),
            MTLSize::new(32, 1, 1),
        );
        Ok(())
    }
}

impl ClampKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AudioCodecKernelError> {
        let fn_name =
            format!("audio_codec_clamp_{}", data_type.function_name_suffix());
        let (pipeline, _reflection) =
            context.compute_pipeline_state_with_reflection(&fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &MTLBuffer,
        output: &MTLBuffer,
        n: usize,
        min_value: f32,
        max_value: f32,
    ) -> Result<(), AudioCodecKernelError> {
        if n == 0 {
            return Ok(());
        }
        encoder.set_label("Audio Clamp");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &min_value as *const f32 as *const _,
        );
        encoder.set_bytes(
            4,
            size_of::<f32>() as u64,
            &max_value as *const f32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }
}

