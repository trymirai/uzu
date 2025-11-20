use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

pub const SSD_PREFILL_CHUNK: usize = 64;
const SSD_PREFILL_THREADGROUP_WIDTH: u64 = SSD_PREFILL_CHUNK as u64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
    Matrix,
}

pub struct SSDPrefillKernel {
    sequential: MTLComputePipelineState,
    single_pass: MTLComputePipelineState,
    matrix_dt_prefix_chunk: MTLComputePipelineState,
    matrix_dt_chunk_scan: MTLComputePipelineState,
    matrix_dt_prefix_apply: MTLComputePipelineState,
    matrix_decay_last: MTLComputePipelineState,
    matrix_pack_bc: MTLComputePipelineState,
    matrix_attn: MTLComputePipelineState,
    matrix_gemm: MTLComputePipelineState,
    matrix_dtx: MTLComputePipelineState,
    matrix_dtxdecay: MTLComputePipelineState,
    matrix_pack_b_head: MTLComputePipelineState,
    matrix_residual: MTLComputePipelineState,
    matrix_pack_c_head: MTLComputePipelineState,
    matrix_accumulate_state: MTLComputePipelineState,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a MTLBuffer,
    pub dt: &'a MTLBuffer, // raw dt values
    pub b: &'a MTLBuffer,
    pub c: &'a MTLBuffer,
    pub d: &'a MTLBuffer,
    pub z: &'a MTLBuffer,
    pub state: &'a MTLBuffer,
    pub y: &'a MTLBuffer,
    pub suffix_len: usize,
    pub group_size: i32,
    pub state_size: i32,
    pub x_strides: [usize; 3],
    pub dt_strides: [usize; 2],
    pub cb_strides: [usize; 3],
    pub state_strides: [usize; 3],
    pub channels: usize,
    pub head_dim: usize,
    pub matrix: Option<SSDPrefillMatrixArguments>,
}

pub struct SSDPrefillMatrixArguments {
    pub prefix: MTLBuffer,
    pub chunk_sums: MTLBuffer,
    pub chunk_offsets: MTLBuffer,
    pub decay_last: MTLBuffer,
    pub c_packed: MTLBuffer,
    pub b_packed: MTLBuffer,
    pub cb_groups: MTLBuffer,
    pub c_head_transposed: MTLBuffer,
    pub attn: MTLBuffer,
    pub dtx: MTLBuffer,
    pub y_tmp: MTLBuffer,
    pub dtxdecay: MTLBuffer,
    pub b_head: MTLBuffer,
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
        let sequential = context
            .compute_pipeline_state(&sequential_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let single_pass = context
            .compute_pipeline_state(&single_name, None)
            .map_err(SSMKernelError::MetalError)?;
        let matrix_dt_prefix_chunk = context
            .compute_pipeline_state(
                &format!(
                    "ssd_m2_dtA_prefix_chunk_kernel_{}",
                    fn_suffix(data_type)
                ),
                None,
            )
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel ssd_m2_dtA_prefix_chunk_kernel"
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_dt_chunk_scan = context
            .compute_pipeline_state("ssd_m2_dtA_chunk_scan_kernel", None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel ssd_m2_dtA_chunk_scan_kernel"
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_dt_prefix_apply = context
            .compute_pipeline_state("ssd_m2_dtA_prefix_apply_kernel", None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel ssd_m2_dtA_prefix_apply_kernel"
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_decay_last_name =
            format!("ssd_m2_decay_last_{}", fn_suffix(data_type));
        let matrix_decay_last = context
            .compute_pipeline_state(&matrix_decay_last_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_decay_last_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_attn_name = format!("ssd_m2_attn_{}", fn_suffix(data_type));
        let matrix_attn = context
            .compute_pipeline_state(&matrix_attn_name, None)
            .map_err(|err| {
                eprintln!("Failed to create Metal kernel {}", matrix_attn_name);
                SSMKernelError::MetalError(err)
            })?;
        let matrix_gemm_name =
            format!("ssd_m2_gemm_batched_{}", fn_suffix(data_type));
        let matrix_gemm = context
            .compute_pipeline_state(&matrix_gemm_name, None)
            .map_err(|err| {
                eprintln!("Failed to create Metal kernel {}", matrix_gemm_name);
                SSMKernelError::MetalError(err)
            })?;
        let matrix_dtx_name = format!("ssd_m2_dtx_{}", fn_suffix(data_type));
        let matrix_dtx = context
            .compute_pipeline_state(&matrix_dtx_name, None)
            .map_err(|err| {
                eprintln!("Failed to create Metal kernel {}", matrix_dtx_name);
                SSMKernelError::MetalError(err)
            })?;
        let matrix_dtxdecay_name =
            format!("ssd_m2_dtxdecay_{}", fn_suffix(data_type));
        let matrix_dtxdecay = context
            .compute_pipeline_state(&matrix_dtxdecay_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_dtxdecay_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_pack_bc_name =
            format!("ssd_m2_pack_bc_{}", fn_suffix(data_type));
        let matrix_pack_bc = context
            .compute_pipeline_state(&matrix_pack_bc_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_pack_bc_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_pack_b_head_name =
            format!("ssd_m2_pack_b_heads_{}", fn_suffix(data_type));
        let matrix_pack_b_head = context
            .compute_pipeline_state(&matrix_pack_b_head_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_pack_b_head_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_residual_name =
            format!("ssd_m2_residual_y_{}", fn_suffix(data_type));
        let matrix_residual = context
            .compute_pipeline_state(&matrix_residual_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_residual_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_pack_c_head_name =
            format!("ssd_m2_pack_c_head_kernel_{}", fn_suffix(data_type));
        let matrix_pack_c_head = context
            .compute_pipeline_state(&matrix_pack_c_head_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_pack_c_head_name
                );
                SSMKernelError::MetalError(err)
            })?;
        let matrix_accumulate_state_name =
            format!("ssd_m2_accumulate_state_{}", fn_suffix(data_type));
        let matrix_accumulate_state = context
            .compute_pipeline_state(&matrix_accumulate_state_name, None)
            .map_err(|err| {
                eprintln!(
                    "Failed to create Metal kernel {}",
                    matrix_accumulate_state_name
                );
                SSMKernelError::MetalError(err)
            })?;
        Ok(Self {
            sequential,
            single_pass,
            matrix_dt_prefix_chunk,
            matrix_dt_chunk_scan,
            matrix_dt_prefix_apply,
            matrix_decay_last,
            matrix_pack_bc,
            matrix_attn,
            matrix_gemm,
            matrix_dtx,
            matrix_dtxdecay,
            matrix_pack_b_head,
            matrix_residual,
            matrix_pack_c_head,
            matrix_accumulate_state,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
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
            SSDPrefillMode::Matrix => {
                let matrix_args = args
                    .matrix
                    .as_ref()
                    .expect("Matrix mode requested but matrix buffers missing");
                self.encode_matrix(compute_encoder, &args, matrix_args)
            },
        }
    }

    fn encode_sequential(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
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
            width: args.channels as u64,
            height: args.head_dim as u64,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
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
            width: args.state_size as u64,
            height: 1,
            depth: 1,
        };
        let pair_count = args.channels as u64 * args.head_dim as u64;
        let total_threads = MTLSize {
            width: pair_count * threads_per_threadgroup.width,
            height: 1,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }

    fn encode_matrix(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: &SSDPrefillArguments,
        matrix: &SSDPrefillMatrixArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0
            || args.head_dim == 0
            || args.suffix_len == 0
            || args.state_size <= 0
        {
            return Ok(());
        }
        let suffix_len = args.suffix_len;
        let channels = args.channels;
        let head_dim = args.head_dim;
        let state_size = args.state_size as usize;
        if state_size == 0 {
            return Ok(());
        }
        let group_size = args.group_size.max(1) as usize;
        if group_size == 0 {
            return Ok(());
        }
        let num_groups = channels / group_size;
        if num_groups == 0 {
            return Ok(());
        }
        let chunk_count =
            (suffix_len + SSD_PREFILL_CHUNK - 1) / SSD_PREFILL_CHUNK;
        let suffix_u32 = suffix_len as u32;
        let channels_u32 = channels as u32;
        let chunk_count_u32 = chunk_count as u32;
        let head_dim_u32 = head_dim as u32;
        let state_u32 = state_size as u32;
        let groups_u32 = num_groups as u32;

        if chunk_count == 0 {
            return Ok(());
        }

        // chunk prefix
        compute_encoder
            .set_compute_pipeline_state(&self.matrix_dt_prefix_chunk);
        compute_encoder.set_buffer(0, Some(args.dt), 0);
        compute_encoder.set_buffer(1, Some(&matrix.prefix), 0);
        compute_encoder.set_buffer(2, Some(&matrix.chunk_sums), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &chunk_count_u32 as *const u32 as *const _,
        );
        let tg_grid = MTLSize {
            width: channels as u64,
            height: chunk_count as u64,
            depth: 1,
        };
        let chunk_threads = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(tg_grid, chunk_threads);

        // chunk scan
        compute_encoder.set_compute_pipeline_state(&self.matrix_dt_chunk_scan);
        compute_encoder.set_buffer(0, Some(&matrix.chunk_sums), 0);
        compute_encoder.set_buffer(1, Some(&matrix.chunk_offsets), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &chunk_count_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        let tg_grid = MTLSize {
            width: channels as u64,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(tg_grid, threads_per_tg);

        // apply chunk offsets
        compute_encoder
            .set_compute_pipeline_state(&self.matrix_dt_prefix_apply);
        compute_encoder.set_buffer(0, Some(&matrix.prefix), 0);
        compute_encoder.set_buffer(1, Some(&matrix.chunk_offsets), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &chunk_count_u32 as *const u32 as *const _,
        );
        let tg_grid = MTLSize {
            width: channels as u64,
            height: chunk_count as u64,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(tg_grid, chunk_threads);

        // decay last row
        compute_encoder.set_compute_pipeline_state(&self.matrix_decay_last);
        compute_encoder.set_buffer(0, Some(&matrix.prefix), 0);
        compute_encoder.set_buffer(1, Some(&matrix.decay_last), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: suffix_len as u64,
            height: channels as u64,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // pack B/C per group
        compute_encoder.set_compute_pipeline_state(&self.matrix_pack_bc);
        compute_encoder.set_buffer(0, Some(args.b), 0);
        compute_encoder.set_buffer(1, Some(args.c), 0);
        compute_encoder.set_buffer(2, Some(&matrix.c_packed), 0);
        compute_encoder.set_buffer(3, Some(&matrix.b_packed), 0);
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &groups_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &state_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: state_size as u64,
            height: suffix_len as u64,
            depth: num_groups as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // GEMM for CB groups
        compute_encoder.set_compute_pipeline_state(&self.matrix_gemm);
        compute_encoder.set_buffer(0, Some(&matrix.c_packed), 0);
        compute_encoder.set_buffer(1, Some(&matrix.b_packed), 0);
        compute_encoder.set_buffer(2, Some(&matrix.cb_groups), 0);
        self.dispatch_gemm(
            compute_encoder,
            suffix_u32,
            suffix_u32,
            state_u32,
            state_u32,
            suffix_u32,
            suffix_u32,
            groups_u32,
            (suffix_len * state_size) as u32,
            (state_size * suffix_len) as u32,
            (suffix_len * suffix_len) as u32,
        );

        // combine CB with decay (compute decay on the fly)
        compute_encoder.set_compute_pipeline_state(&self.matrix_attn);
        compute_encoder.set_buffer(0, Some(&matrix.cb_groups), 0);
        compute_encoder.set_buffer(1, Some(&matrix.prefix), 0);
        compute_encoder.set_buffer(2, Some(&matrix.attn), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &groups_u32 as *const u32 as *const _,
        );
        let tg_grid = MTLSize {
            width: channels as u64,
            height: suffix_len as u64,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(tg_grid, threads_per_tg);

        // pack C per head
        compute_encoder.set_compute_pipeline_state(&self.matrix_pack_c_head);
        compute_encoder.set_buffer(0, Some(&matrix.c_packed), 0);
        compute_encoder.set_buffer(1, Some(&matrix.prefix), 0);
        compute_encoder.set_buffer(2, Some(&matrix.c_head_transposed), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &groups_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &state_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: suffix_len as u64,
            height: state_size as u64,
            depth: channels as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // preload state contribution from initial state (writes into dtxdecay scratch)
        compute_encoder.set_compute_pipeline_state(&self.matrix_gemm);
        compute_encoder.set_buffer(0, Some(args.state), 0);
        compute_encoder.set_buffer(1, Some(&matrix.c_head_transposed), 0);
        compute_encoder.set_buffer(2, Some(&matrix.dtxdecay), 0);
        self.dispatch_gemm(
            compute_encoder,
            head_dim_u32,
            suffix_u32,
            state_u32,
            state_u32,
            suffix_u32,
            suffix_u32,
            channels_u32,
            (head_dim * state_size) as u32,
            (state_size * suffix_len) as u32,
            (head_dim * suffix_len) as u32,
        );

        // pack B per head
        compute_encoder.set_compute_pipeline_state(&self.matrix_pack_b_head);
        compute_encoder.set_buffer(0, Some(args.b), 0);
        compute_encoder.set_buffer(1, Some(&matrix.b_head), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &groups_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &state_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: state_size as u64,
            height: suffix_len as u64,
            depth: channels as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // dtx
        compute_encoder.set_compute_pipeline_state(&self.matrix_dtx);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(&matrix.dtx), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: head_dim as u64,
            height: suffix_len as u64,
            depth: channels as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // attention GEMM
        compute_encoder.set_compute_pipeline_state(&self.matrix_gemm);
        compute_encoder.set_buffer(0, Some(&matrix.attn), 0);
        compute_encoder.set_buffer(1, Some(&matrix.dtx), 0);
        compute_encoder.set_buffer(2, Some(&matrix.y_tmp), 0);
        self.dispatch_gemm(
            compute_encoder,
            suffix_u32,
            head_dim_u32,
            suffix_u32,
            suffix_u32,
            head_dim_u32,
            head_dim_u32,
            channels_u32,
            (suffix_len * suffix_len) as u32,
            (suffix_len * head_dim) as u32,
            (suffix_len * head_dim) as u32,
        );

        // add initial state contribution into y_tmp
        compute_encoder
            .set_compute_pipeline_state(&self.matrix_accumulate_state);
        compute_encoder.set_buffer(0, Some(&matrix.dtxdecay), 0);
        compute_encoder.set_buffer(1, Some(&matrix.y_tmp), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: head_dim as u64,
            height: suffix_len as u64,
            depth: channels as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // residual + gate
        compute_encoder.set_compute_pipeline_state(&self.matrix_residual);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.d), 0);
        compute_encoder.set_buffer(2, Some(args.z), 0);
        compute_encoder.set_buffer(3, Some(&matrix.y_tmp), 0);
        compute_encoder.set_buffer(4, Some(args.y), 0);
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: head_dim as u64,
            height: channels as u64,
            depth: suffix_len as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 4,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // dtxdecay
        compute_encoder.set_compute_pipeline_state(&self.matrix_dtxdecay);
        compute_encoder.set_buffer(0, Some(&matrix.dtx), 0);
        compute_encoder.set_buffer(1, Some(&matrix.decay_last), 0);
        compute_encoder.set_buffer(2, Some(&matrix.dtxdecay), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &suffix_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &channels_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        let threads = MTLSize {
            width: suffix_len as u64,
            height: head_dim as u64,
            depth: channels as u64,
        };
        let threads_per_tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        compute_encoder.dispatch_threads(threads, threads_per_tg);

        // next state GEMM
        compute_encoder.set_compute_pipeline_state(&self.matrix_gemm);
        compute_encoder.set_buffer(0, Some(&matrix.dtxdecay), 0);
        compute_encoder.set_buffer(1, Some(&matrix.b_head), 0);
        compute_encoder.set_buffer(2, Some(args.state), 0);
        self.dispatch_gemm(
            compute_encoder,
            head_dim_u32,
            state_u32,
            suffix_u32,
            suffix_u32,
            state_u32,
            state_u32,
            channels_u32,
            (head_dim * suffix_len) as u32,
            (suffix_len * state_size) as u32,
            (head_dim * state_size) as u32,
        );

        Ok(())
    }

    fn dispatch_gemm(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
        batch: u32,
        stride_a: u32,
        stride_b: u32,
        stride_c: u32,
    ) {
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &n as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &lda as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &ldb as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &ldc as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &batch as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &stride_a as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &stride_b as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            size_of::<u32>() as u64,
            &stride_c as *const u32 as *const _,
        );
        let grid = MTLSize {
            width: ((n + 15) / 16) as u64,
            height: ((m + 7) / 8) as u64,
            depth: batch as u64,
        };
        let threads = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_thread_groups(grid, threads);
    }

    fn bind_common_buffers(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
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
