use std::mem::size_of;

use crate::backends::metal::{
    BufferRef, ComputeCommandEncoderRef, ComputeEncoderLegacy,
    ComputePipelineState, FunctionConstantValues, FunctionConstantValuesLegacy,
    KernelDataType, MTLContext, MTLDataType, MTLSize,
};

use objc2::rc::Retained;

use super::{SSMKernelError, fn_suffix};
use crate::config::Activation;

const CONV1D_SCAN_THREADGROUP_SIZE: usize = 32;

const ACTIVATION_IDENTITY: i32 = 0;
const ACTIVATION_SILU: i32 = 1;
const ACTIVATION_GELU: i32 = 2;

fn activation_to_int(activation: &Activation) -> i32 {
    match activation {
        Activation::Identity => ACTIVATION_IDENTITY,
        Activation::SiLU {
            ..
        } => ACTIVATION_SILU,
        Activation::Gelu => ACTIVATION_GELU,
    }
}

fn make_function_constants(
    activation: &Activation,
    has_bias: bool,
) -> Retained<FunctionConstantValues> {
    let function_constants = FunctionConstantValues::new();
    let activation_type = activation_to_int(activation);

    function_constants.set_constant_value_at_index(
        &activation_type as *const i32 as *const std::ffi::c_void,
        MTLDataType::Int,
        0,
    );

    function_constants.set_constant_value_at_index(
        &has_bias as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        1,
    );

    function_constants
}

pub struct Conv1dScanKernel {
    pipeline_no_bias: ComputePipelineState,
    pipeline_with_bias: ComputePipelineState,
    pack_pipeline: ComputePipelineState,
    decode_pipeline_no_bias: ComputePipelineState,
    decode_pipeline_with_bias: ComputePipelineState,
}

pub struct Conv1dScanArguments<'a> {
    pub padded: BufferRef<'a>, // buffer(0) [prefix+suffix, channels]
    pub w: BufferRef<'a>,      // buffer(1) [channels, kernel]
    pub b: Option<BufferRef<'a>>, // buffer(2) [channels]
    pub x_out: BufferRef<'a>,  // buffer(3) [suffix, inner_dim]
    pub b_out: BufferRef<'a>,  // buffer(4) [suffix, proj_dim]
    pub c_out: BufferRef<'a>,  // buffer(5) [suffix, proj_dim]
    pub state_out: BufferRef<'a>, // buffer(6) [channels, kernel-1]
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub row_stride: usize,
    pub state_stride: usize,
    pub channels: usize,
    pub inner_dim: usize,
    pub proj_dim: usize,
}

pub struct Conv1dPackArguments<'a> {
    pub state_in: BufferRef<'a>,
    pub x: BufferRef<'a>,
    pub padded: BufferRef<'a>,
    pub state_stride: usize,
    pub row_stride: usize,
    pub suffix_len: usize,
    pub channels: usize,
}

pub struct Conv1dDecodeArguments<'a> {
    pub x: BufferRef<'a>,
    pub w: BufferRef<'a>,
    pub b: Option<BufferRef<'a>>,
    pub state: BufferRef<'a>,
    pub x_out: BufferRef<'a>,
    pub b_out: BufferRef<'a>,
    pub c_out: BufferRef<'a>,
    pub next_state: BufferRef<'a>,
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub row_stride: usize,
    pub state_stride: usize,
    pub channels: usize,
    pub inner_dim: usize,
    pub proj_dim: usize,
}

impl Conv1dScanKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        activation: &Activation,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("conv1d_scan_kernel_{}", fn_suffix(data_type));
        let pack_name =
            format!("conv1d_pack_prefix_kernel_{}", fn_suffix(data_type));
        let decode_name =
            format!("conv1d_decode_kernel_{}", fn_suffix(data_type));
        let activation_id = activation_to_int(activation);
        let pipeline_no_bias = {
            let function_constants = make_function_constants(activation, false);
            let cache_key = format!("{}_act_{}_bias_0", fn_name, activation_id);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &fn_name,
                    Some(&function_constants),
                )
                .map_err(SSMKernelError::MetalError)?
        };
        let pipeline_with_bias = {
            let function_constants = make_function_constants(activation, true);
            let cache_key = format!("{}_act_{}_bias_1", fn_name, activation_id);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &fn_name,
                    Some(&function_constants),
                )
                .map_err(SSMKernelError::MetalError)?
        };
        let pack_pipeline = context
            .compute_pipeline_state(&pack_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let decode_pipeline_no_bias = {
            let function_constants = make_function_constants(activation, false);
            let cache_key =
                format!("{}_act_{}_bias_0", decode_name, activation_id);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &decode_name,
                    Some(&function_constants),
                )
                .map_err(SSMKernelError::MetalError)?
        };
        let decode_pipeline_with_bias = {
            let function_constants = make_function_constants(activation, true);
            let cache_key =
                format!("{}_act_{}_bias_1", decode_name, activation_id);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &decode_name,
                    Some(&function_constants),
                )
                .map_err(SSMKernelError::MetalError)?
        };
        Ok(Self {
            pipeline_no_bias,
            pipeline_with_bias,
            pack_pipeline,
            decode_pipeline_no_bias,
            decode_pipeline_with_bias,
        })
    }

    pub fn encode_pack(
        &self,
        compute_encoder: ComputeCommandEncoderRef<'_>,
        args: Conv1dPackArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.state_stride == 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.pack_pipeline);
        compute_encoder.set_buffer(0, Some(args.state_in), 0);
        compute_encoder.set_buffer(1, Some(args.x), 0);
        compute_encoder.set_buffer(2, Some(args.padded), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<usize>() as u64,
            &args.row_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &(args.channels as u32) as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: CONV1D_SCAN_THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        };
        let padded_rows = args.state_stride + args.suffix_len;
        let total_threads = MTLSize {
            width: args.channels,
            height: padded_rows,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode_decode(
        &self,
        compute_encoder: ComputeCommandEncoderRef<'_>,
        args: Conv1dDecodeArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.suffix_len == 0 || args.kernel_size <= 0 {
            return Ok(());
        }

        let has_bias = args.b.is_some();
        let pipeline = if has_bias {
            &self.decode_pipeline_with_bias
        } else {
            &self.decode_pipeline_no_bias
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        if has_bias {
            compute_encoder.set_buffer(2, args.b, 0);
        }
        compute_encoder.set_buffer(3, Some(args.state), 0);
        compute_encoder.set_buffer(4, Some(args.x_out), 0);
        compute_encoder.set_buffer(5, Some(args.b_out), 0);
        compute_encoder.set_buffer(6, Some(args.c_out), 0);
        compute_encoder.set_buffer(7, Some(args.next_state), 0);
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.row_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &(args.channels as u32) as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            13,
            size_of::<u32>() as u64,
            &(args.inner_dim as u32) as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            14,
            size_of::<u32>() as u64,
            &(args.proj_dim as u32) as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: CONV1D_SCAN_THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.suffix_len,
            height: args.channels,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode(
        &self,
        compute_encoder: ComputeCommandEncoderRef<'_>,
        args: Conv1dScanArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.suffix_len == 0 || args.kernel_size <= 0 {
            return Ok(());
        }

        let has_bias = args.b.is_some();
        let pipeline = if has_bias {
            &self.pipeline_with_bias
        } else {
            &self.pipeline_no_bias
        };
        compute_encoder.set_compute_pipeline_state(pipeline);

        compute_encoder.set_buffer(0, Some(args.padded), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        if has_bias {
            compute_encoder.set_buffer(2, args.b, 0);
        }
        compute_encoder.set_buffer(3, Some(args.x_out), 0);
        compute_encoder.set_buffer(4, Some(args.b_out), 0);
        compute_encoder.set_buffer(5, Some(args.c_out), 0);
        compute_encoder.set_buffer(6, Some(args.state_out), 0);

        compute_encoder.set_bytes(
            7,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.row_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &(args.channels as u32) as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            size_of::<u32>() as u64,
            &(args.inner_dim as u32) as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            13,
            size_of::<u32>() as u64,
            &(args.proj_dim as u32) as *const u32 as *const _,
        );

        let suffix = args.suffix_len;
        let tap_count = args.kernel_size.saturating_sub(1) as usize;
        let work_len = suffix + tap_count;
        let channels = args.channels;
        let threads_per_threadgroup = MTLSize {
            width: CONV1D_SCAN_THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: work_len,
            height: channels,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
