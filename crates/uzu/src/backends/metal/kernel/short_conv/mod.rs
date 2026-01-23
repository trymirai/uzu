use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    ComputePipelineState, FunctionConstantValues,
    FunctionConstantValuesLegacy, KernelDataType, MTLBuffer,
    MTLContext,
    MTLDataType, MTLError, MTLSize, ProtocolObject,
};

#[derive(Debug, thiserror::Error)]
pub enum ShortConvKernelError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

use objc2::rc::Retained;

fn fn_suffix(dt: KernelDataType) -> &'static str {
    dt.function_name_suffix()
}

fn make_function_constants(has_bias: bool) -> Retained<FunctionConstantValues> {
    let function_constants = FunctionConstantValues::new();
    function_constants.set_constant_value_at_index(
        &has_bias as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        0,
    );
    function_constants
}

pub struct ShortConvKernel {
    pack_pipeline: ComputePipelineState,
    prefill_pipeline_no_bias: ComputePipelineState,
    prefill_pipeline_with_bias: ComputePipelineState,
    decode_pipeline_no_bias: ComputePipelineState,
    decode_pipeline_with_bias: ComputePipelineState,
}

pub struct ShortConvPackArguments<'a> {
    pub state_in: &'a ProtocolObject<dyn MTLBuffer>,
    pub in_proj: &'a ProtocolObject<dyn MTLBuffer>,
    pub padded: &'a ProtocolObject<dyn MTLBuffer>,
    pub state_stride: usize,
    pub suffix_len: usize,
    pub in_proj_stride: usize,
    pub model_dim: usize,
}

pub struct ShortConvPrefillArguments<'a> {
    pub padded: &'a ProtocolObject<dyn MTLBuffer>,
    pub in_proj: &'a ProtocolObject<dyn MTLBuffer>,
    pub w: &'a ProtocolObject<dyn MTLBuffer>,
    pub b: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    pub out: &'a ProtocolObject<dyn MTLBuffer>,
    pub state_out: &'a ProtocolObject<dyn MTLBuffer>,
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub in_proj_stride: usize,
    pub state_stride: usize,
    pub model_dim: usize,
}

pub struct ShortConvDecodeArguments<'a> {
    pub in_proj: &'a ProtocolObject<dyn MTLBuffer>,
    pub w: &'a ProtocolObject<dyn MTLBuffer>,
    pub b: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    pub state: &'a ProtocolObject<dyn MTLBuffer>,
    pub out: &'a ProtocolObject<dyn MTLBuffer>,
    pub next_state: &'a ProtocolObject<dyn MTLBuffer>,
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub in_proj_stride: usize,
    pub state_stride: usize,
    pub model_dim: usize,
}

impl ShortConvKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, ShortConvKernelError> {
        let pack_name =
            format!("short_conv_pack_kernel_{}", fn_suffix(data_type));
        let prefill_name =
            format!("short_conv_prefill_kernel_{}", fn_suffix(data_type));
        let decode_name =
            format!("short_conv_decode_kernel_{}", fn_suffix(data_type));

        let pack_pipeline = context
            .compute_pipeline_state(&pack_name, None)
            .map_err(ShortConvKernelError::MetalError)?;

        let prefill_pipeline_no_bias = {
            let function_constants = make_function_constants(false);
            let cache_key = format!("{}_has_bias_0", prefill_name);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &prefill_name,
                    Some(&function_constants),
                )
                .map_err(ShortConvKernelError::MetalError)?
        };
        let prefill_pipeline_with_bias = {
            let function_constants = make_function_constants(true);
            let cache_key = format!("{}_has_bias_1", prefill_name);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &prefill_name,
                    Some(&function_constants),
                )
                .map_err(ShortConvKernelError::MetalError)?
        };
        let decode_pipeline_no_bias = {
            let function_constants = make_function_constants(false);
            let cache_key = format!("{}_has_bias_0", decode_name);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &decode_name,
                    Some(&function_constants),
                )
                .map_err(ShortConvKernelError::MetalError)?
        };
        let decode_pipeline_with_bias = {
            let function_constants = make_function_constants(true);
            let cache_key = format!("{}_has_bias_1", decode_name);
            context
                .compute_pipeline_state_cached(
                    &cache_key,
                    &decode_name,
                    Some(&function_constants),
                )
                .map_err(ShortConvKernelError::MetalError)?
        };

        Ok(Self {
            pack_pipeline,
            prefill_pipeline_no_bias,
            prefill_pipeline_with_bias,
            decode_pipeline_no_bias,
            decode_pipeline_with_bias,
        })
    }

    pub fn encode_pack(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: ShortConvPackArguments,
    ) -> Result<(), ShortConvKernelError> {
        if args.model_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.pack_pipeline);
        compute_encoder.set_buffer(Some(args.state_in), 0, 0);
        compute_encoder.set_buffer(Some(args.in_proj), 0, 1);
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
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                4,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &args.in_proj_stride as *const usize as *mut c_void,
                )
                .unwrap(),
                size_of::<usize>(),
                5,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.model_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                6,
            );
        }

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let padded_rows = args.state_stride + args.suffix_len;
        let threadgroups = MTLSize {
            width: args.model_dim,
            height: padded_rows,
            depth: 1,
        };

        compute_encoder.dispatch_threads(threadgroups, threads_per_threadgroup);

        Ok(())
    }

    pub fn encode_prefill(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: ShortConvPrefillArguments,
    ) -> Result<(), ShortConvKernelError> {
        if args.model_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        let tap_count = (args.kernel_size - 1).max(0);
        let work_len = args.suffix_len + tap_count as usize;

        let has_bias = args.b.is_some();
        let pipeline = if has_bias {
            &self.prefill_pipeline_with_bias
        } else {
            &self.prefill_pipeline_no_bias
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(Some(args.padded), 0, 0);
        compute_encoder.set_buffer(Some(args.in_proj), 0, 1);
        compute_encoder.set_buffer(Some(args.w), 0, 2);
        if has_bias {
            compute_encoder.set_buffer(args.b, 0, 3);
        }
        compute_encoder.set_buffer(Some(args.out), 0, 4);
        compute_encoder.set_buffer(Some(args.state_out), 0, 5);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                6,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.kernel_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                7,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &args.in_proj_stride as *const usize as *mut c_void,
                )
                .unwrap(),
                size_of::<usize>(),
                8,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                9,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.model_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                10,
            );
        }

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: work_len,
            height: args.model_dim,
            depth: 1,
        };

        compute_encoder.dispatch_threads(threadgroups, threads_per_threadgroup);

        Ok(())
    }

    pub fn encode_decode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: ShortConvDecodeArguments,
    ) -> Result<(), ShortConvKernelError> {
        if args.model_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        let has_bias = args.b.is_some();
        let pipeline = if has_bias {
            &self.decode_pipeline_with_bias
        } else {
            &self.decode_pipeline_no_bias
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(Some(args.in_proj), 0, 0);
        compute_encoder.set_buffer(Some(args.w), 0, 1);
        if has_bias {
            compute_encoder.set_buffer(args.b, 0, 2);
        }
        compute_encoder.set_buffer(Some(args.state), 0, 3);
        compute_encoder.set_buffer(Some(args.out), 0, 4);
        compute_encoder.set_buffer(Some(args.next_state), 0, 5);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                6,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.kernel_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                7,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &args.in_proj_stride as *const usize as *mut c_void,
                )
                .unwrap(),
                size_of::<usize>(),
                8,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_stride as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                9,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(
                    &(args.model_dim as u32) as *const u32
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                size_of::<u32>(),
                10,
            );
        }

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: args.suffix_len,
            height: args.model_dim,
            depth: 1,
        };

        compute_encoder.dispatch_threads(threadgroups, threads_per_threadgroup);

        Ok(())
    }
}
