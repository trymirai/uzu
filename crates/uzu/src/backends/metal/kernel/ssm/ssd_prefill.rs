use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

pub const SSD_PREFILL_CHUNK: usize = 64;
const SSD_PREFILL_THREADGROUP_WIDTH: u64 = SSD_PREFILL_CHUNK as u64;
const SSD_PREFILL_STATE_THREADS: u64 = 64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    SinglePass,
    MultiPass,
}

pub struct SSDPrefillKernel {
    single_pass: MTLComputePipelineState,
    chunk_transform: MTLComputePipelineState,
    chunk_scan: MTLComputePipelineState,
    chunk_apply: MTLComputePipelineState,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a MTLBuffer,
    pub dt: &'a MTLBuffer,
    pub decay: &'a MTLBuffer,
    pub b: &'a MTLBuffer,
    pub c: &'a MTLBuffer,
    pub d: &'a MTLBuffer,
    pub z: &'a MTLBuffer,
    pub state: &'a MTLBuffer,
    pub y: &'a MTLBuffer,
    pub chunk_a: &'a MTLBuffer,
    pub chunk_b: &'a MTLBuffer,
    pub chunk_prefix: &'a MTLBuffer,
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
        let single_name =
            format!("ssd_prefill_kernel_{}", fn_suffix(data_type));
        let transform_name = format!(
            "ssd_prefill_chunk_transform_kernel_{}",
            fn_suffix(data_type)
        );
        let scan_name =
            format!("ssd_prefill_chunk_scan_kernel_{}", fn_suffix(data_type));
        let apply_name =
            format!("ssd_prefill_chunk_apply_kernel_{}", fn_suffix(data_type));
        let single_pass = context
            .compute_pipeline_state(&single_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let chunk_transform = context
            .compute_pipeline_state(&transform_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let chunk_scan = context
            .compute_pipeline_state(&scan_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let chunk_apply = context
            .compute_pipeline_state(&apply_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            single_pass,
            chunk_transform,
            chunk_scan,
            chunk_apply,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: SSDPrefillArguments,
        mode: SSDPrefillMode,
    ) -> Result<(), SSMKernelError> {
        match mode {
            SSDPrefillMode::SinglePass => {
                self.encode_single(compute_encoder, &args)
            },
            SSDPrefillMode::MultiPass => {
                self.encode_multi(compute_encoder, &args)
            },
        }
    }

    fn encode_single(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: &SSDPrefillArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.head_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }
        compute_encoder.set_compute_pipeline_state(&self.single_pass);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.decay), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.c), 0);
        compute_encoder.set_buffer(5, Some(args.d), 0);
        compute_encoder.set_buffer(6, Some(args.z), 0);
        compute_encoder.set_buffer(7, Some(args.state), 0);
        compute_encoder.set_buffer(8, Some(args.y), 0);
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &args.group_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<i32>() as u64,
            &args.state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            13,
            (2 * size_of::<usize>()) as u64,
            args.dt_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            14,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            15,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        let channels = args.channels as u32;
        let head_dim = args.head_dim as u32;
        compute_encoder.set_bytes(
            16,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            17,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        let threads_per_threadgroup = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        let pair_count = args.channels as u64 * args.head_dim as u64;
        let total_threads = MTLSize {
            width: pair_count * SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    fn encode_multi(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: &SSDPrefillArguments,
    ) -> Result<(), SSMKernelError> {
        let pair_count = args.channels * args.head_dim;
        if pair_count == 0 || args.suffix_len == 0 {
            return Ok(());
        }
        let chunk_count =
            (args.suffix_len + SSD_PREFILL_CHUNK - 1) / SSD_PREFILL_CHUNK;
        if chunk_count == 0 {
            return Ok(());
        }
        let state_dim = args.state_size as usize;

        let chunk_a_strides =
            [args.channels * args.head_dim, args.head_dim, 1usize];
        let chunk_state_strides = [
            args.channels * args.head_dim * state_dim,
            args.head_dim * state_dim,
            state_dim,
            1usize,
        ];

        let channels = args.channels as u32;
        let head_dim = args.head_dim as u32;
        let suffix_len = args.suffix_len;
        let group_size = args.group_size;
        let state_size = args.state_size;

        // Stage 1: per-chunk transform
        compute_encoder.set_compute_pipeline_state(&self.chunk_transform);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.decay), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.chunk_a), 0);
        compute_encoder.set_buffer(5, Some(args.chunk_b), 0);
        compute_encoder.set_bytes(
            6,
            size_of::<usize>() as u64,
            &suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &group_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            10,
            (2 * size_of::<usize>()) as u64,
            args.dt_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            11,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            12,
            (3 * size_of::<usize>()) as u64,
            chunk_a_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            13,
            (4 * size_of::<usize>()) as u64,
            chunk_state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            14,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            15,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        let tg_stage = MTLSize {
            width: pair_count as u64,
            height: chunk_count as u64,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder
            .dispatch_thread_groups(tg_stage, threads_per_threadgroup);

        // Stage 2: chunk prefix scan
        compute_encoder.set_compute_pipeline_state(&self.chunk_scan);
        compute_encoder.set_buffer(0, Some(args.chunk_a), 0);
        compute_encoder.set_buffer(1, Some(args.chunk_b), 0);
        compute_encoder.set_buffer(2, Some(args.state), 0);
        compute_encoder.set_buffer(3, Some(args.chunk_prefix), 0);
        compute_encoder.set_bytes(
            4,
            size_of::<usize>() as u64,
            &suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            (3 * size_of::<usize>()) as u64,
            chunk_a_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            7,
            (4 * size_of::<usize>()) as u64,
            chunk_state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            8,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            9,
            (4 * size_of::<usize>()) as u64,
            chunk_state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        let tg_state = MTLSize {
            width: pair_count as u64,
            height: 1,
            depth: 1,
        };
        let state_threads = MTLSize {
            width: SSD_PREFILL_STATE_THREADS,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(tg_state, state_threads);

        // Stage 3: replay chunks with prefixes
        compute_encoder.set_compute_pipeline_state(&self.chunk_apply);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.decay), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.c), 0);
        compute_encoder.set_buffer(5, Some(args.d), 0);
        compute_encoder.set_buffer(6, Some(args.z), 0);
        compute_encoder.set_buffer(7, Some(args.state), 0);
        compute_encoder.set_buffer(8, Some(args.y), 0);
        compute_encoder.set_buffer(9, Some(args.chunk_prefix), 0);
        compute_encoder.set_bytes(
            10,
            size_of::<usize>() as u64,
            &suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<i32>() as u64,
            &group_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            size_of::<i32>() as u64,
            &state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            13,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            14,
            (2 * size_of::<usize>()) as u64,
            args.dt_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            15,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            16,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            17,
            (4 * size_of::<usize>()) as u64,
            chunk_state_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            18,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            19,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        compute_encoder
            .dispatch_thread_groups(tg_stage, threads_per_threadgroup);
        Ok(())
    }
}
