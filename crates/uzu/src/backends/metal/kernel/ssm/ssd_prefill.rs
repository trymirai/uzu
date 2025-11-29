use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

const SSD_PREFILL_SINGLE_THREADS: u64 = 32;
const SSD_PREFILL_PASS1_THREADS: u64 = 256;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
    Flash,
}

pub struct SSDPrefillKernel {
    sequential: MTLComputePipelineState,
    single_pass: MTLComputePipelineState,
    single_pass_64: MTLComputePipelineState,
    // Multi-pass Flash kernels
    flash_pass1_decay: MTLComputePipelineState,
    flash_gemm_cb_fused: MTLComputePipelineState,
    flash_gemm_y: MTLComputePipelineState,
    flash_gemm_state_c: MTLComputePipelineState,
    flash_fused_add_skip_gate: MTLComputePipelineState,
    // State update kernels
    flash_gemm_state_update: MTLComputePipelineState,
    flash_finalize_state: MTLComputePipelineState,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a MTLBuffer,
    pub dt: &'a MTLBuffer,
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
    // Flash mode buffers
    pub flash_scratch: Option<&'a MTLBuffer>, // cum_log_decay: [H, L+1]
    pub flash_cb: Option<&'a MTLBuffer>,      // CB matrix: [H, L, L]
    pub flash_y_float: Option<&'a MTLBuffer>, // y intermediate: [L, H, dh]
    pub flash_state_c: Option<&'a MTLBuffer>, // state @ C^T: [H, dh, L]
}

impl SSDPrefillKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let suffix = fn_suffix(data_type);

        let sequential = context
            .compute_pipeline_state(
                &format!("ssd_prefill_kernel_sequential_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let single_pass = context
            .compute_pipeline_state(
                &format!("ssd_prefill_kernel_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let single_pass_64 = context
            .compute_pipeline_state(
                &format!("ssd_prefill_kernel_64_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;

        // Flash multi-pass kernels
        let flash_pass1_decay = context
            .compute_pipeline_state(
                &format!("ssd_prefill_flash_pass1_decay_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let flash_gemm_cb_fused = context
            .compute_pipeline_state(
                &format!("ssd_gemm_cb_fused_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let flash_gemm_y = context
            .compute_pipeline_state(&format!("ssd_gemm_y_{}", suffix), None)
            .map_err(SSMKernelError::MetalError)?;
        let flash_gemm_state_c = context
            .compute_pipeline_state(
                &format!("ssd_gemm_state_c_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let flash_fused_add_skip_gate = context
            .compute_pipeline_state(
                &format!("ssd_fused_add_skip_gate_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let flash_gemm_state_update = context
            .compute_pipeline_state(
                &format!("ssd_gemm_state_update_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;
        let flash_finalize_state = context
            .compute_pipeline_state(
                &format!("ssd_finalize_state_{}", suffix),
                None,
            )
            .map_err(SSMKernelError::MetalError)?;

        Ok(Self {
            sequential,
            single_pass,
            single_pass_64,
            flash_pass1_decay,
            flash_gemm_cb_fused,
            flash_gemm_y,
            flash_gemm_state_c,
            flash_fused_add_skip_gate,
            flash_gemm_state_update,
            flash_finalize_state,
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
            SSDPrefillMode::Flash => self.encode_flash(compute_encoder, &args),
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
        compute_encoder.dispatch_threads(
            MTLSize {
                width: args.channels as u64,
                height: args.head_dim as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 32,
                depth: 1,
            },
        );
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

        let pair_count = args.channels as u64 * args.head_dim as u64;
        compute_encoder.dispatch_threads(
            MTLSize {
                width: pair_count * SSD_PREFILL_SINGLE_THREADS,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SSD_PREFILL_SINGLE_THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_flash(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: &SSDPrefillArguments,
    ) -> Result<(), SSMKernelError> {
        if args.channels == 0 || args.head_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        let decay_buf = args.flash_scratch.ok_or_else(|| {
            SSMKernelError::InvalidArguments(
                "Flash mode requires flash_scratch buffer".into(),
            )
        })?;
        let cb_buf = args.flash_cb.ok_or_else(|| {
            SSMKernelError::InvalidArguments(
                "Flash mode requires flash_cb buffer".into(),
            )
        })?;
        let y_float_buf = args.flash_y_float.ok_or_else(|| {
            SSMKernelError::InvalidArguments(
                "Flash mode requires flash_y_float buffer".into(),
            )
        })?;
        let state_c_buf = args.flash_state_c.ok_or_else(|| {
            SSMKernelError::InvalidArguments(
                "Flash mode requires flash_state_c buffer".into(),
            )
        })?;

        let seq_len = args.suffix_len as u32;
        let num_heads = args.channels as u32;
        let head_dim = args.head_dim as u32;
        let state_dim = args.state_size as u32;
        let num_groups = (num_heads as i32 / args.group_size.max(1)) as u32;

        // ========== Pass 1: Decay prefix sums ==========
        compute_encoder.set_compute_pipeline_state(&self.flash_pass1_decay);
        compute_encoder.set_buffer(0, Some(args.dt), 0);
        compute_encoder.set_buffer(1, Some(decay_buf), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<usize>() as u64,
            &args.suffix_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<usize>() as u64,
            &args.dt_strides[0] as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<usize>() as u64,
            &args.dt_strides[1] as *const _ as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: num_heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SSD_PREFILL_PASS1_THREADS,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.memory_barrier_with_resources(&[decay_buf]);

        // ========== Pass 2: FUSED CB = tril((C @ B^T) * decay) (BM=32, BN=32) ==========
        let tiles_m = (seq_len as u64 + 31) / 32; // BM=32
        let tiles_n = (seq_len as u64 + 31) / 32; // BN=32

        compute_encoder.set_compute_pipeline_state(&self.flash_gemm_cb_fused);
        compute_encoder.set_buffer(0, Some(args.c), 0);
        compute_encoder.set_buffer(1, Some(args.b), 0);
        compute_encoder.set_buffer(2, Some(decay_buf), 0);
        compute_encoder.set_buffer(3, Some(cb_buf), 0);
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &state_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &num_groups as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<usize>() as u64,
            &args.cb_strides[0] as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.cb_strides[1] as *const _ as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: tiles_n,
                height: tiles_m,
                depth: num_heads as u64,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.memory_barrier_with_resources(&[cb_buf]);

        // ========== Pass 3: GEMM y = CB @ x (BM=32, BN=32) ==========
        let tiles_m_y = (seq_len as u64 + 31) / 32; // BM=32
        let tiles_n_y = (head_dim as u64 + 31) / 32; // BN=32

        compute_encoder.set_compute_pipeline_state(&self.flash_gemm_y);
        compute_encoder.set_buffer(0, Some(cb_buf), 0);
        compute_encoder.set_buffer(1, Some(args.x), 0);
        compute_encoder.set_buffer(2, Some(y_float_buf), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &head_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            6,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: tiles_n_y,
                height: tiles_m_y,
                depth: num_heads as u64,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.memory_barrier_with_resources(&[y_float_buf]);

        // ========== Pass 4.5a: GEMM state_C = state @ C^T ==========
        // Tile sizes match SSDSC_BM=16, SSDSC_BN=32
        let tiles_m_sc = (head_dim as u64 + 15) / 16;
        let tiles_n_sc = (seq_len as u64 + 31) / 32;

        compute_encoder.set_compute_pipeline_state(&self.flash_gemm_state_c);
        compute_encoder.set_buffer(0, Some(args.state), 0);
        compute_encoder.set_buffer(1, Some(args.c), 0);
        compute_encoder.set_buffer(2, Some(state_c_buf), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &head_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &state_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &num_groups as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            8,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            9,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: tiles_n_sc,
                height: tiles_m_sc,
                depth: num_heads as u64,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.memory_barrier_with_resources(&[state_c_buf]);

        // ========== Pass 4: FUSED add_state + skip_gate ==========
        // y_out = (y_in + decay * state_C + D * x) * silu(z)
        let total_elements =
            (seq_len as u64) * (num_heads as u64) * (head_dim as u64);
        let fused_tgs = (total_elements + 255) / 256;

        compute_encoder
            .set_compute_pipeline_state(&self.flash_fused_add_skip_gate);
        compute_encoder.set_buffer(0, Some(y_float_buf), 0);
        compute_encoder.set_buffer(1, Some(state_c_buf), 0);
        compute_encoder.set_buffer(2, Some(decay_buf), 0);
        compute_encoder.set_buffer(3, Some(args.x), 0);
        compute_encoder.set_buffer(4, Some(args.d), 0);
        compute_encoder.set_buffer(5, Some(args.z), 0);
        compute_encoder.set_buffer(6, Some(args.y), 0);
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &head_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            10,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: fused_tgs,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        // ========== Pass 6: State update GEMM ==========
        // contribution = (scaled_x)^T @ B where scaled_x = x * decay_t_to_T
        // Reuse state_c_buf for contribution (H*dh*N fits in H*dh*L since N <= L)
        // Tile sizes: BM=16, BN=32
        let tiles_m_su = (head_dim as u64 + 15) / 16;
        let tiles_n_su = (state_dim as u64 + 31) / 32;

        compute_encoder
            .set_compute_pipeline_state(&self.flash_gemm_state_update);
        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.b), 0);
        compute_encoder.set_buffer(2, Some(decay_buf), 0);
        compute_encoder.set_buffer(3, Some(state_c_buf), 0); // reuse for contribution
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &head_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &state_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &num_groups as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            9,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            10,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: tiles_n_su,
                height: tiles_m_su,
                depth: num_heads as u64,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.memory_barrier_with_resources(&[state_c_buf]);

        // ========== Pass 7: Finalize state ==========
        // state_new = decay_total * state_old + contribution
        let total_state_elems =
            (num_heads as u64) * (head_dim as u64) * (state_dim as u64);
        let state_tgs = (total_state_elems + 255) / 256;

        compute_encoder.set_compute_pipeline_state(&self.flash_finalize_state);
        compute_encoder.set_buffer(0, Some(args.state), 0);
        compute_encoder.set_buffer(1, Some(state_c_buf), 0); // contribution
        compute_encoder.set_buffer(2, Some(decay_buf), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &seq_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &num_heads as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &head_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &state_dim as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            7,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: state_tgs,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        Ok(())
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
            &args.suffix_len as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &args.group_size as *const _ as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &args.state_size as *const _ as *const _,
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
