use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use super::{SSMKernelError, fn_suffix};
use crate::{
    backends::metal::{
        KernelDataType, MTLBuffer, MTLComputePipelineState, MTLContext, MTLDataType,
        MTLFunctionConstantValues, MTLSize, ProtocolObject, Retained,
    },
    config::Activation,
};

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
) -> Retained<MTLFunctionConstantValues> {
    let function_constants = MTLFunctionConstantValues::new();
    let activation_type = activation_to_int(activation);

    function_constants.set_constant_value_type_at_index(
        NonNull::from(&activation_type).cast(),
        MTLDataType::Int,
        0,
    );

    function_constants.set_constant_value_type_at_index(
        NonNull::from(&has_bias).cast(),
        MTLDataType::Bool,
        1,
    );

    function_constants
}

pub struct Conv1dScanKernel {
    pipeline_no_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_with_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pack_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    decode_pipeline_no_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    decode_pipeline_with_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct Conv1dScanArguments<'a> {
    pub padded: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0) [prefix+suffix, channels]
    pub w: &'a ProtocolObject<dyn MTLBuffer>, // buffer(1) [channels, kernel]
    pub b: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(2) [channels]
    pub x_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(3) [suffix, inner_dim]
    pub b_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(4) [suffix, proj_dim]
    pub c_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(5) [suffix, proj_dim]
    pub state_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(6) [channels, kernel-1]
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub row_stride: usize,
    pub state_stride: usize,
    pub channels: usize,
    pub inner_dim: usize,
    pub proj_dim: usize,
}

pub struct Conv1dPackArguments<'a> {
    pub state_in: &'a ProtocolObject<dyn MTLBuffer>,
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    pub padded: &'a ProtocolObject<dyn MTLBuffer>,
    pub state_stride: usize,
    pub row_stride: usize,
    pub suffix_len: usize,
    pub channels: usize,
}

pub struct Conv1dDecodeArguments<'a> {
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    pub w: &'a ProtocolObject<dyn MTLBuffer>,
    pub b: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    pub state: &'a ProtocolObject<dyn MTLBuffer>,
    pub x_out: &'a ProtocolObject<dyn MTLBuffer>,
    pub b_out: &'a ProtocolObject<dyn MTLBuffer>,
    pub c_out: &'a ProtocolObject<dyn MTLBuffer>,
    pub next_state: &'a ProtocolObject<dyn MTLBuffer>,
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
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: Conv1dPackArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.state_stride == 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.pack_pipeline);
        compute_encoder.set_buffer(Some(args.state_in), 0, 0);
        compute_encoder.set_buffer(Some(args.x), 0, 1);
        compute_encoder.set_buffer(Some(args.padded), 0, 2);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                3,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.row_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                4,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                5,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.channels as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                6,
            );
        }

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
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
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
        compute_encoder.set_buffer(Some(args.x), 0, 0);
        compute_encoder.set_buffer(Some(args.w), 0, 1);
        if has_bias {
            compute_encoder.set_buffer(args.b, 0, 2);
        }
        compute_encoder.set_buffer(Some(args.state), 0, 3);
        compute_encoder.set_buffer(Some(args.x_out), 0, 4);
        compute_encoder.set_buffer(Some(args.b_out), 0, 5);
        compute_encoder.set_buffer(Some(args.c_out), 0, 6);
        compute_encoder.set_buffer(Some(args.next_state), 0, 7);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.kernel_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                8,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.row_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                9,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                10,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.channels as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                11,
            );
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                12,
            );
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.inner_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                13,
            );
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.proj_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                14,
            );
        }

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
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
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

        compute_encoder.set_buffer(Some(args.padded), 0, 0);
        compute_encoder.set_buffer(Some(args.w), 0, 1);
        if has_bias {
            compute_encoder.set_buffer(args.b, 0, 2);
        }
        compute_encoder.set_buffer(Some(args.x_out), 0, 3);
        compute_encoder.set_buffer(Some(args.b_out), 0, 4);
        compute_encoder.set_buffer(Some(args.c_out), 0, 5);
        compute_encoder.set_buffer(Some(args.state_out), 0, 6);

        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                7,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.kernel_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                8,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.row_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                9,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                10,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.channels as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                11,
            );
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.inner_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                12,
            );
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.proj_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                13,
            );
        }

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
