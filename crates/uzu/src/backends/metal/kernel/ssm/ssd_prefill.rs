use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{
    KernelDataType, MTLBuffer,
    MTLContext, MTLSize, MTLComputePipelineState, ProtocolObject, Retained,
};

const SSD_PREFILL_SINGLE_THREADS: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
}

pub struct SSDPrefillKernel {
    sequential: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    single_pass: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    single_pass_64: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    pub dt: &'a ProtocolObject<dyn MTLBuffer>, // raw dt values
    pub b: &'a ProtocolObject<dyn MTLBuffer>,
    pub c: &'a ProtocolObject<dyn MTLBuffer>,
    pub d: &'a ProtocolObject<dyn MTLBuffer>,
    pub z: &'a ProtocolObject<dyn MTLBuffer>,
    pub state: &'a ProtocolObject<dyn MTLBuffer>,
    pub y: &'a ProtocolObject<dyn MTLBuffer>,
    pub suffix_len: usize,
    pub group_size: i32,
    pub state_size: i32,
    pub x_strides: [usize; 3],
    pub dt_strides: [usize; 2],
    pub cb_strides: [usize; 3],
    pub state_strides: [usize; 3],
    pub channels: usize,
    pub head_dim: usize,
}

impl SSDPrefillKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let sequential_name =
            format!("ssd_prefill_kernel_sequential_{}", fn_suffix(data_type));
        let single_name =
            format!("ssd_prefill_kernel_{}", fn_suffix(data_type));
        let single_64_name =
            format!("ssd_prefill_kernel_64_{}", fn_suffix(data_type));
        let sequential = context
            .compute_pipeline_state(&sequential_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let single_pass = context
            .compute_pipeline_state(&single_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let single_pass_64 = context
            .compute_pipeline_state(&single_64_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            sequential,
            single_pass,
            single_pass_64,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: SSDPrefillArguments,
        mode: SSDPrefillMode,
    ) -> Result<(), SSMKernelError> {
        match mode {
            SSDPrefillMode::Sequential => {
                self.encode_sequential(compute_encoder, &args)
            },
            SSDPrefillMode::SinglePass => {
                self.encode_single(compute_encoder, &args)
            },
        }
    }

    fn encode_sequential(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &SSDPrefillArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.head_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }
        compute_encoder.set_compute_pipeline_state(&self.sequential);
        self.bind_common_buffers(compute_encoder, args);
        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 32,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.channels,
            height: args.head_dim,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    fn encode_single(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &SSDPrefillArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.head_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }
        if args.state_size == 64 {
            compute_encoder.set_compute_pipeline_state(&self.single_pass_64);
        } else {
            compute_encoder.set_compute_pipeline_state(&self.single_pass);
        }
        self.bind_common_buffers(compute_encoder, args);
        let channels = args.channels as u32;
        let head_dim = args.head_dim as u32;
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&channels as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                15,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&head_dim as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                16,
            );
        }
        let threads_per_threadgroup = MTLSize {
            width: SSD_PREFILL_SINGLE_THREADS,
            height: 1,
            depth: 1,
        };
        let pair_count = args.channels * args.head_dim;
        let total_threads = MTLSize {
            width: pair_count * SSD_PREFILL_SINGLE_THREADS,
            height: 1,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    fn bind_common_buffers(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &SSDPrefillArguments,
    ) {
        compute_encoder.set_buffer(Some(args.x), 0, 0);
        compute_encoder.set_buffer(Some(args.dt), 0, 1);
        compute_encoder.set_buffer(Some(args.b), 0, 2);
        compute_encoder.set_buffer(Some(args.c), 0, 3);
        compute_encoder.set_buffer(Some(args.d), 0, 4);
        compute_encoder.set_buffer(Some(args.z), 0, 5);
        compute_encoder.set_buffer(Some(args.state), 0, 6);
        compute_encoder.set_buffer(Some(args.y), 0, 7);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.suffix_len as *const usize as *mut c_void)
                    .unwrap(),
                size_of::<usize>(),
                8,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.group_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                9,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.state_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                10,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(args.x_strides.as_ptr() as *mut std::ffi::c_void)
                    .unwrap(),
                3 * size_of::<usize>(),
                11,
            );
            compute_encoder.set_bytes(
                NonNull::new(args.dt_strides.as_ptr() as *mut std::ffi::c_void)
                    .unwrap(),
                2 * size_of::<usize>(),
                12,
            );
            compute_encoder.set_bytes(
                NonNull::new(args.cb_strides.as_ptr() as *mut std::ffi::c_void)
                    .unwrap(),
                3 * size_of::<usize>(),
                13,
            );
            compute_encoder.set_bytes(
                NonNull::new(
                    args.state_strides.as_ptr() as *mut std::ffi::c_void
                )
                .unwrap(),
                3 * size_of::<usize>(),
                14,
            );
        }
    }
}
