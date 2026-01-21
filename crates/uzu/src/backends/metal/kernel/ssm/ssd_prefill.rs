use std::mem::size_of;

use crate::backends::metal::{
    BufferRef, ComputeCommandEncoderRef, ComputeEncoderLegacy,
    ComputePipelineState, KernelDataType, MTLContext, MTLSize,
};

use super::{SSMKernelError, fn_suffix};

const SSD_PREFILL_SINGLE_THREADS: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
}

pub struct SSDPrefillKernel {
    sequential: ComputePipelineState,
    single_pass: ComputePipelineState,
    single_pass_64: ComputePipelineState,
}

pub struct SSDPrefillArguments<'a> {
    pub x: BufferRef<'a>,
    pub dt: BufferRef<'a>, // raw dt values
    pub b: BufferRef<'a>,
    pub c: BufferRef<'a>,
    pub d: BufferRef<'a>,
    pub z: BufferRef<'a>,
    pub state: BufferRef<'a>,
    pub y: BufferRef<'a>,
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
        compute_encoder: ComputeCommandEncoderRef<'_>,
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
        compute_encoder: ComputeCommandEncoderRef<'_>,
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
        compute_encoder: ComputeCommandEncoderRef<'_>,
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
        compute_encoder.set_bytes(
            15,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            16,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
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
        compute_encoder: ComputeCommandEncoderRef<'_>,
        args: &SSDPrefillArguments,
    ) {
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.b), 0);
        compute_encoder.set_buffer(3, Some(args.c), 0);
        compute_encoder.set_buffer(4, Some(args.d), 0);
        compute_encoder.set_buffer(5, Some(args.z), 0);
        compute_encoder.set_buffer(6, Some(args.state), 0);
        compute_encoder.set_buffer(7, Some(args.y), 0);
        compute_encoder.set_bytes(
            8,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &args.group_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &args.state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            12,
            (2 * size_of::<usize>()) as u64,
            args.dt_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            13,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            14,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
    }
}
