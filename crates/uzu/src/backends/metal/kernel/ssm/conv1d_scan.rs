use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::{
    backends::metal::{KernelDataType, MTLContext},
    config::Activation,
};

const CONV1D_SCAN_THREADGROUP_SIZE: u64 = 32;

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

fn make_function_constants(activation: &Activation) -> FunctionConstantValues {
    let function_constants = FunctionConstantValues::new();
    let activation_type = activation_to_int(activation);

    function_constants.set_constant_value_at_index(
        &activation_type as *const i32 as *const std::ffi::c_void,
        MTLDataType::Int,
        0,
    );

    function_constants
}

pub struct Conv1dScanKernel {
    pipeline: MTLComputePipelineState,
    pack_pipeline: MTLComputePipelineState,
}

pub struct Conv1dScanArguments<'a> {
    pub padded: &'a MTLBuffer,    // buffer(0) [prefix+suffix, channels]
    pub w: &'a MTLBuffer,         // buffer(1) [channels, kernel]
    pub b: Option<&'a MTLBuffer>, // buffer(2) [channels]
    pub y: &'a MTLBuffer,         // buffer(3) [suffix, channels]
    pub state_out: &'a MTLBuffer, // buffer(4) [channels, kernel-1]
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub row_stride: usize,
    pub state_stride: usize,
    pub channels: usize,
}

pub struct Conv1dPackArguments<'a> {
    pub state_in: &'a MTLBuffer,
    pub x: &'a MTLBuffer,
    pub padded: &'a MTLBuffer,
    pub state_stride: usize,
    pub row_stride: usize,
    pub suffix_len: usize,
    pub channels: usize,
}

impl Conv1dScanKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        activation: &Activation,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("conv1d_scan_kernel_{}", fn_suffix(data_type));
        let pack_name = format!(
            "conv1d_pack_prefix_kernel_{}",
            fn_suffix(data_type)
        );
        let function_constants = make_function_constants(activation);
        let pipeline = context
            .compute_pipeline_state(&fn_name, Some(&function_constants))
            .map_err(SSMKernelError::MetalError)?;
        let pack_pipeline = context
            .compute_pipeline_state(&pack_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            pipeline,
            pack_pipeline,
        })
    }

    pub fn encode_pack(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
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
            width: args.channels as u64,
            height: padded_rows as u64,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: Conv1dScanArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.suffix_len == 0 || args.kernel_size <= 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.padded), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        if let Some(bias) = args.b {
            compute_encoder.set_buffer(2, Some(bias), 0);
        } else {
            compute_encoder.set_buffer(2, None, 0);
        }
        compute_encoder.set_buffer(3, Some(args.y), 0);
        compute_encoder.set_buffer(4, Some(args.state_out), 0);

        compute_encoder.set_bytes(
            5,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<usize>() as u64,
            &args.row_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &(args.channels as u32) as *const u32 as *const _,
        );

        let suffix = args.suffix_len as u64;
        let tap_count = args.kernel_size.saturating_sub(1) as u64;
        let work_len = suffix + tap_count;
        let channels = args.channels as u64;
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
