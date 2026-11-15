//! LFM3-VL vision tower on Metal.
//!
//! Mirrors `lfm3_tower.rs` (CPU f32 reference) with f16 kernels: patchifier
//! GEMM -> embedding RMSNorm -> 14 conv/attention hybrid layers -> LayerNorm ->
//! 4-pack merge -> linear_1 -> erf GELU -> linear_2.
//!
//! Weight upload reuses the CPU tower's f32 tensors (`MetalLinearWeight::from_cpu`),
//! so the load/parse path stays single-sourced.

use std::{ffi::c_void, path::Path, sync::Mutex};

use anyhow::{Result, ensure};
use half::f16;

use super::{
    common::f32_to_f16_parallel,
    lfm3_tower::{Lfm3Mixer, Lfm3VisionConfig, Lfm3VisionTower, rope_tables_for},
    metal_tower::{MetalGpuBuffer, MetalLayerNormWeights, MetalLinearWeight, MetalVisionCtx, VisionCtxHandle},
    tower::LayerNormWeights,
};

unsafe extern "C" {
    fn metal_vision_create_context() -> *mut MetalVisionCtx;
    fn metal_vision_destroy_context(ctx: *mut MetalVisionCtx);
    fn metal_vision_begin_pass(ctx: *mut MetalVisionCtx);
    fn metal_vision_end_pass(ctx: *mut MetalVisionCtx);

    fn metal_vision_write_buffer(
        buf: *mut c_void,
        src: *const c_void,
        bytes: usize,
    );

    fn metal_vision_gelu_erf_fp16(
        ctx: *mut MetalVisionCtx,
        x_buf: *mut c_void,
        num_elements: i32,
    );
    fn metal_vision_add_residual_fp16(
        ctx: *mut MetalVisionCtx,
        x_buf: *mut c_void,
        res_buf: *mut c_void,
        count: i32,
    );

    fn metal_vision_rms_norm_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        weight_buf: *mut c_void,
        out_buf: *mut c_void,
        num_tokens: i32,
        dim: i32,
        eps: f32,
    );

    fn metal_vision_lfm3_qk_norm_rope_fp16(
        ctx: *mut MetalVisionCtx,
        q_buf: *mut c_void,
        k_buf: *mut c_void,
        q_norm_buf: *mut c_void,
        k_norm_buf: *mut c_void,
        cos_buf: *mut c_void,
        sin_buf: *mut c_void,
        num_tokens: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    );

    fn metal_vision_lfm3_conv_gate_scatter_fp16(
        ctx: *mut MetalVisionCtx,
        proj_buf: *mut c_void,
        grid_buf: *mut c_void,
        num_tokens: i32,
        grid_h: i32,
        grid_w: i32,
        merge: i32,
        dim: i32,
    );

    fn metal_vision_lfm3_depthwise_conv3x3_fp16(
        ctx: *mut MetalVisionCtx,
        grid_buf: *mut c_void,
        kernel_buf: *mut c_void,
        out_buf: *mut c_void,
        grid_h: i32,
        grid_w: i32,
        dim: i32,
    );

    fn metal_vision_lfm3_conv_gather_gate_fp16(
        ctx: *mut MetalVisionCtx,
        conv_buf: *mut c_void,
        proj_buf: *mut c_void,
        gated_buf: *mut c_void,
        num_tokens: i32,
        grid_h: i32,
        grid_w: i32,
        merge: i32,
        dim: i32,
    );

    fn metal_vision_lfm3_swiglu_fp16(
        ctx: *mut MetalVisionCtx,
        fused_buf: *mut c_void,
        gated_buf: *mut c_void,
        num_tokens: i32,
        hidden: i32,
    );

    fn metal_vision_lfm3_merge_pack_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        out_buf: *mut c_void,
        num_tokens: i32,
        dim: i32,
        merge: i32,
    );

    fn metal_vision_lfm3_heads_to_major_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        out_buf: *mut c_void,
        num_tokens: i32,
        heads: i32,
        head_dim: i32,
    );

    fn metal_vision_lfm3_flash_attention_fp16(
        ctx: *mut MetalVisionCtx,
        q_buf: *mut c_void,
        k_buf: *mut c_void,
        v_buf: *mut c_void,
        out_buf: *mut c_void,
        num_tokens: i32,
        num_heads: i32,
        num_kv_heads: i32,
    );
}

enum Lfm3MetalMixer {
    Conv {
        in_projection: MetalLinearWeight,
        spatial_conv: MetalGpuBuffer, // f32 [dim, 3, 3]
        out_projection: MetalLinearWeight,
    },
    Attention {
        q_proj: MetalLinearWeight,
        k_proj: MetalLinearWeight,
        v_proj: MetalLinearWeight,
        out_proj: MetalLinearWeight,
        query_norm: MetalGpuBuffer, // f32 [head_dim]
        key_norm: MetalGpuBuffer,   // f32 [head_dim]
    },
}

struct Lfm3MetalLayer {
    mixer: Lfm3MetalMixer,
    pre_mixer_norm: MetalGpuBuffer, // f32 [dim]
    pre_mlp_norm: MetalGpuBuffer,   // f32 [dim]
    mlp_up: MetalLinearWeight,
    mlp_down: MetalLinearWeight,
}

/// Scratch buffers sized for `capacity_patches` per frame.
struct Lfm3Arena {
    capacity_patches: usize,
    buf_in: MetalGpuBuffer,        // n * patch_dim
    buf_x: MetalGpuBuffer,         // n * dim (residual stream)
    buf_normed: MetalGpuBuffer,    // n * dim
    buf_tmp: MetalGpuBuffer,       // n * dim (mixer / mlp output)
    buf_proj: MetalGpuBuffer,      // n * 3 * dim
    buf_grid: MetalGpuBuffer,      // n * dim
    buf_conv_out: MetalGpuBuffer,  // n * dim
    buf_q: MetalGpuBuffer,         // n * dim (token-major)
    buf_k: MetalGpuBuffer,         // n * kv_dim (token-major)
    buf_v: MetalGpuBuffer,         // n * kv_dim (token-major)
    buf_q_hm: MetalGpuBuffer,      // dim * n (head-major)
    buf_k_hm: MetalGpuBuffer,      // kv_dim * n
    buf_v_hm: MetalGpuBuffer,      // kv_dim * n
    buf_attn: MetalGpuBuffer,      // n * dim
    buf_mlp_fused: MetalGpuBuffer, // n * 2 * hidden
    buf_mlp_gated: MetalGpuBuffer, // n * hidden
    buf_packed: MetalGpuBuffer,    // rows * 4 * dim
    buf_h1: MetalGpuBuffer,        // rows * linear_1_out
    buf_out: MetalGpuBuffer,       // rows * linear_2_out
    buf_cos: MetalGpuBuffer,       // n * head_dim f32
    buf_sin: MetalGpuBuffer,       // n * head_dim f32
}

pub struct Lfm3MetalVisionTower {
    ctx: VisionCtxHandle,
    config: Lfm3VisionConfig,
    dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    merge: usize,
    eps: f32,
    projector_eps: f32,
    patch_dim: usize,
    patchifier: MetalLinearWeight,
    embedding_norm: MetalGpuBuffer,
    layers: Vec<Lfm3MetalLayer>,
    projector_ln: MetalLayerNormWeights,
    projector_linear_1: MetalLinearWeight,
    projector_linear_2: MetalLinearWeight,
    arena: Mutex<Option<Lfm3Arena>>,
    forward_lock: Mutex<()>,
}

unsafe impl Send for Lfm3MetalVisionTower {}
unsafe impl Sync for Lfm3MetalVisionTower {}

impl Drop for Lfm3MetalVisionTower {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { metal_vision_destroy_context(self.ctx.raw()) };
        }
    }
}

fn f32_buffer(
    ctx: VisionCtxHandle,
    values: &[f32],
) -> MetalGpuBuffer {
    MetalGpuBuffer::new_f32(ctx, values.len() * 4, Some(values))
}

fn f32_bytes(
    ctx: VisionCtxHandle,
    len: usize,
) -> MetalGpuBuffer {
    MetalGpuBuffer::new_f32(ctx, len * 4, None)
}

impl Lfm3MetalVisionTower {
    /// Whether a Metal context can be created on this machine (cached).
    pub fn is_available() -> bool {
        static AVAILABLE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            let ctx = unsafe { VisionCtxHandle::from_raw(metal_vision_create_context()) };
            if ctx.is_null() {
                return false;
            }
            unsafe { metal_vision_destroy_context(ctx.raw()) };
            true
        });
        *AVAILABLE
    }

    pub fn from_cpu(tower: &Lfm3VisionTower) -> Result<Self> {
        let ctx = unsafe { VisionCtxHandle::from_raw(metal_vision_create_context()) };
        ensure!(!ctx.is_null(), "Failed to create Metal context for the LFM3-VL vision tower");

        let cfg = &tower.config;
        let dim = cfg.model_dim;
        let hidden_dim = cfg.hidden_dim;
        let head_dim = dim / cfg.num_heads;
        ensure!(head_dim == 64, "LF3-M3 metal kernels expect head_dim = 64, got {head_dim}");

        let patchifier = MetalLinearWeight::from_cpu(ctx, &tower.patchifier);
        let embedding_norm = f32_buffer(ctx, &tower.embedding_norm);

        let mut layers = Vec::with_capacity(tower.layers.len());
        for layer in &tower.layers {
            let mixer = match &layer.mixer {
                Lfm3Mixer::Conv {
                    in_projection,
                    spatial_conv,
                    out_projection,
                } => Lfm3MetalMixer::Conv {
                    in_projection: MetalLinearWeight::from_cpu(ctx, in_projection),
                    spatial_conv: f32_buffer(ctx, spatial_conv),
                    out_projection: MetalLinearWeight::from_cpu(ctx, out_projection),
                },
                Lfm3Mixer::Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    out_proj,
                    query_norm,
                    key_norm,
                    ..
                } => Lfm3MetalMixer::Attention {
                    q_proj: MetalLinearWeight::from_cpu(ctx, q_proj),
                    k_proj: MetalLinearWeight::from_cpu(ctx, k_proj),
                    v_proj: MetalLinearWeight::from_cpu(ctx, v_proj),
                    out_proj: MetalLinearWeight::from_cpu(ctx, out_proj),
                    query_norm: f32_buffer(ctx, query_norm),
                    key_norm: f32_buffer(ctx, key_norm),
                },
            };
            layers.push(Lfm3MetalLayer {
                mixer,
                pre_mixer_norm: f32_buffer(ctx, &layer.pre_mixer_norm),
                pre_mlp_norm: f32_buffer(ctx, &layer.pre_mlp_norm),
                mlp_up: MetalLinearWeight::from_cpu(ctx, &layer.mlp_up),
                mlp_down: MetalLinearWeight::from_cpu(ctx, &layer.mlp_down),
            });
        }

        let ln = LayerNormWeights {
            gamma: tower.projector_ln.0.clone(),
            beta: tower.projector_ln.1.clone(),
        };
        Ok(Self {
            ctx,
            config: cfg.clone(),
            dim,
            hidden_dim,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim,
            merge: cfg.spatial_merge_size,
            eps: cfg.norm_epsilon,
            projector_eps: cfg.projector.epsilon,
            patch_dim: cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
            patchifier,
            embedding_norm,
            layers,
            projector_ln: MetalLayerNormWeights::from_cpu(ctx, &ln),
            projector_linear_1: MetalLinearWeight::from_cpu(ctx, &tower.projector_linear_1),
            projector_linear_2: MetalLinearWeight::from_cpu(ctx, &tower.projector_linear_2),
            arena: Mutex::new(None),
            forward_lock: Mutex::new(()),
        })
    }

    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let tower = Lfm3VisionTower::load_from_model_dir(model_dir)?;
        Self::from_cpu(&tower)
    }

    fn ensure_arena(
        &self,
        num_patches: usize,
    ) -> std::sync::MutexGuard<'_, Option<Lfm3Arena>> {
        let mut guard = self.arena.lock().unwrap();
        let need_alloc = match &*guard {
            Some(a) => a.capacity_patches < num_patches,
            None => true,
        };
        if need_alloc {
            // Round up to 64 for allocator granularity; flash-attention itself
            // dispatches 32-row query blocks, so tail rows (>= n) still land
            // inside the arena buffers.
            let cap = num_patches.max(1024).div_ceil(64) * 64;
            let ctx = self.ctx;
            let dim = self.dim;
            let hidden = self.hidden_dim;
            let kv_dim = self.num_kv_heads * self.head_dim;
            let rows = cap / (self.merge * self.merge) + 1;
            let lin1 = self.projector_linear_1.out_features;
            let lin2 = self.projector_linear_2.out_features;
            *guard = Some(Lfm3Arena {
                capacity_patches: cap,
                buf_in: MetalGpuBuffer::new_f16(ctx, cap * self.patch_dim * 2, None),
                buf_x: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_normed: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_tmp: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_proj: MetalGpuBuffer::new_f16(ctx, cap * 3 * dim * 2, None),
                buf_grid: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_conv_out: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_q: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_k: MetalGpuBuffer::new_f16(ctx, cap * kv_dim * 2, None),
                buf_v: MetalGpuBuffer::new_f16(ctx, cap * kv_dim * 2, None),
                buf_q_hm: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_k_hm: MetalGpuBuffer::new_f16(ctx, cap * kv_dim * 2, None),
                buf_v_hm: MetalGpuBuffer::new_f16(ctx, cap * kv_dim * 2, None),
                buf_attn: MetalGpuBuffer::new_f16(ctx, cap * dim * 2, None),
                buf_mlp_fused: MetalGpuBuffer::new_f16(ctx, cap * 2 * hidden * 2, None),
                buf_mlp_gated: MetalGpuBuffer::new_f16(ctx, cap * hidden * 2, None),
                buf_packed: MetalGpuBuffer::new_f16(ctx, rows * self.merge * self.merge * dim * 2, None),
                buf_h1: MetalGpuBuffer::new_f16(ctx, rows * lin1 * 2, None),
                buf_out: MetalGpuBuffer::new_f16(ctx, rows * lin2 * 2, None),
                buf_cos: f32_bytes(ctx, cap * self.head_dim),
                buf_sin: f32_bytes(ctx, cap * self.head_dim),
            });
        }
        guard
    }

    fn dump_f16(
        &self,
        dir: Option<&str>,
        name: &str,
        buf: &MetalGpuBuffer,
        len: usize,
    ) {
        let Some(dir) = dir else {
            return;
        };
        unsafe { metal_vision_end_pass(self.ctx.raw()) };
        let mut raw = vec![f16::ZERO; len];
        buf.read_f16_slice(&mut raw);
        let bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_f32().to_le_bytes()).collect();
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(format!("{dir}/{name}"), bytes);
        unsafe { metal_vision_begin_pass(self.ctx.raw()) };
    }

    fn rms_norm(
        &self,
        input: &MetalGpuBuffer,
        weight: &MetalGpuBuffer,
        output: &MetalGpuBuffer,
        rows: usize,
    ) {
        unsafe {
            metal_vision_rms_norm_fp16(
                self.ctx.raw(),
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                rows as i32,
                self.dim as i32,
                self.eps,
            );
        }
    }

    /// Runs just the patchifier + embedding RMSNorm for one frame and returns
    /// f32 rows. Used by the first parity milestone.
    pub fn forward_embed(
        &self,
        pixels: &[f32],
        grid: [usize; 3],
    ) -> Result<Vec<f32>> {
        let [grid_t, grid_h, grid_w] = grid;
        ensure!(grid_t == 1, "forward_embed supports a single frame");
        let n = grid_h * grid_w;
        ensure!(
            pixels.len() == n * self.patch_dim,
            "pixel_values size mismatch: {} != {n} * {}",
            pixels.len(),
            self.patch_dim
        );

        let _lock = self.forward_lock.lock().unwrap();
        let mut guard = self.ensure_arena(n);
        let arena = guard.as_mut().unwrap();
        let ctx = self.ctx;

        let pixels_f16 = f32_to_f16_parallel(pixels);
        unsafe {
            metal_vision_begin_pass(ctx.raw());
            metal_vision_write_buffer(arena.buf_in.ptr(), pixels_f16.as_ptr() as *const c_void, pixels_f16.len() * 2);
        }

        self.patchifier.forward(ctx, &arena.buf_in, n, &arena.buf_x);
        self.rms_norm(&arena.buf_x, &self.embedding_norm, &arena.buf_x, n);
        let debug_dir = std::env::var("LFM3_VL_METAL_DEBUG_DIR").ok();
        self.dump_f16(debug_dir.as_deref(), "embed.f32", &arena.buf_x, n * self.dim);

        unsafe { metal_vision_end_pass(ctx.raw()) };

        let mut out = vec![0f32; n * self.dim];
        let mut raw = vec![f16::ZERO; n * self.dim];
        arena.buf_x.read_f16_slice(&mut raw);
        for (o, v) in out.iter_mut().zip(raw) {
            *o = v.to_f32();
        }
        Ok(out)
    }

    /// Full tower: patchifier -> mixed layers -> projector. Frames are
    /// independent through the tower, so `grid_t` frames loop over the shared
    /// scratch arena and accumulate projector rows.
    pub fn forward(
        &self,
        pixel_values: &[f32],
        grid: [usize; 3],
    ) -> Result<Vec<f32>> {
        let [grid_t, grid_h, grid_w] = grid;
        ensure!(grid_t >= 1, "LFM3-VL Metal tower needs at least one frame");
        let n = grid_h * grid_w;
        ensure!(
            pixel_values.len() == grid_t * n * self.patch_dim,
            "pixel_values size mismatch: {} != {grid_t} * {n} * {}",
            pixel_values.len(),
            self.patch_dim
        );
        ensure!(
            grid_h % self.merge == 0 && grid_w % self.merge == 0,
            "grid {grid_h}x{grid_w} is not divisible by merge size {}",
            self.merge
        );

        let _lock = self.forward_lock.lock().unwrap();
        let mut guard = self.ensure_arena(n);
        let arena = guard.as_mut().unwrap();
        let ctx = self.ctx;
        let dim = self.dim;
        let debug_dir = std::env::var("LFM3_VL_METAL_DEBUG_DIR").ok();
        let profile = std::env::var("LFM3_VL_METAL_PROFILE").is_ok();
        let mut lap_start = std::time::Instant::now();
        let lap = |label: &str, lap_start: &mut std::time::Instant| {
            if profile {
                unsafe { metal_vision_end_pass(ctx.raw()) };
                eprintln!("vision {label}: {:.2}ms", lap_start.elapsed().as_secs_f64() * 1e3);
                unsafe { metal_vision_begin_pass(ctx.raw()) };
                *lap_start = std::time::Instant::now();
            }
        };

        let (cos, sin) = rope_tables_for(&self.config, grid_h, grid_w);
        arena.buf_cos.write_f32(&cos);
        arena.buf_sin.write_f32(&sin);

        // Frames are independent through the projector (attention and the 2x2
        // merge never cross a frame boundary), so video is a per-frame loop
        // over the same scratch arena; only the projector rows accumulate.
        let rows = n / (self.merge * self.merge);
        let lin2 = self.projector_linear_2.out_features;
        let mut all = Vec::with_capacity(grid_t * rows * lin2);
        for frame in 0..grid_t {
            let frame_pixels = &pixel_values[frame * n * self.patch_dim..(frame + 1) * n * self.patch_dim];
            let pixels_f16 = f32_to_f16_parallel(frame_pixels);
            unsafe {
                metal_vision_begin_pass(ctx.raw());
                metal_vision_write_buffer(
                    arena.buf_in.ptr(),
                    pixels_f16.as_ptr() as *const c_void,
                    pixels_f16.len() * 2,
                );
            }

            self.patchifier.forward(ctx, &arena.buf_in, n, &arena.buf_x);
            self.rms_norm(&arena.buf_x, &self.embedding_norm, &arena.buf_x, n);
            self.dump_f16(debug_dir.as_deref(), "embed.f32", &arena.buf_x, n * dim);
            lap("patchifier", &mut lap_start);

            for (li, layer) in self.layers.iter().enumerate() {
                self.rms_norm(&arena.buf_x, &layer.pre_mixer_norm, &arena.buf_normed, n);
                match &layer.mixer {
                    Lfm3MetalMixer::Conv {
                        in_projection,
                        spatial_conv,
                        out_projection,
                    } => {
                        in_projection.forward(ctx, &arena.buf_normed, n, &arena.buf_proj);
                        unsafe {
                            metal_vision_lfm3_conv_gate_scatter_fp16(
                                ctx.raw(),
                                arena.buf_proj.ptr(),
                                arena.buf_grid.ptr(),
                                n as i32,
                                grid_h as i32,
                                grid_w as i32,
                                self.merge as i32,
                                dim as i32,
                            );
                            metal_vision_lfm3_depthwise_conv3x3_fp16(
                                ctx.raw(),
                                arena.buf_grid.ptr(),
                                spatial_conv.ptr(),
                                arena.buf_conv_out.ptr(),
                                grid_h as i32,
                                grid_w as i32,
                                dim as i32,
                            );
                            // `buf_normed` is free after the in-projection GEMM.
                            metal_vision_lfm3_conv_gather_gate_fp16(
                                ctx.raw(),
                                arena.buf_conv_out.ptr(),
                                arena.buf_proj.ptr(),
                                arena.buf_normed.ptr(),
                                n as i32,
                                grid_h as i32,
                                grid_w as i32,
                                self.merge as i32,
                                dim as i32,
                            );
                        }
                        out_projection.forward(ctx, &arena.buf_normed, n, &arena.buf_tmp);
                    },
                    Lfm3MetalMixer::Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        out_proj,
                        query_norm,
                        key_norm,
                    } => {
                        q_proj.forward(ctx, &arena.buf_normed, n, &arena.buf_q);
                        k_proj.forward(ctx, &arena.buf_normed, n, &arena.buf_k);
                        v_proj.forward(ctx, &arena.buf_normed, n, &arena.buf_v);
                        unsafe {
                            metal_vision_lfm3_qk_norm_rope_fp16(
                                ctx.raw(),
                                arena.buf_q.ptr(),
                                arena.buf_k.ptr(),
                                query_norm.ptr(),
                                key_norm.ptr(),
                                arena.buf_cos.ptr(),
                                arena.buf_sin.ptr(),
                                n as i32,
                                self.num_heads as i32,
                                self.num_kv_heads as i32,
                                self.head_dim as i32,
                                self.eps,
                            );
                            // MPS reads dense head-major matrices; token-major rows
                            // would stride over the other heads' columns.
                            metal_vision_lfm3_heads_to_major_fp16(
                                ctx.raw(),
                                arena.buf_q.ptr(),
                                arena.buf_q_hm.ptr(),
                                n as i32,
                                self.num_heads as i32,
                                self.head_dim as i32,
                            );
                            metal_vision_lfm3_heads_to_major_fp16(
                                ctx.raw(),
                                arena.buf_k.ptr(),
                                arena.buf_k_hm.ptr(),
                                n as i32,
                                self.num_kv_heads as i32,
                                self.head_dim as i32,
                            );
                            metal_vision_lfm3_heads_to_major_fp16(
                                ctx.raw(),
                                arena.buf_v.ptr(),
                                arena.buf_v_hm.ptr(),
                                n as i32,
                                self.num_kv_heads as i32,
                                self.head_dim as i32,
                            );
                            // `buf_attn` is written after `buf_normed` is consumed.
                            metal_vision_lfm3_flash_attention_fp16(
                                ctx.raw(),
                                arena.buf_q_hm.ptr(),
                                arena.buf_k_hm.ptr(),
                                arena.buf_v_hm.ptr(),
                                arena.buf_attn.ptr(),
                                n as i32,
                                self.num_heads as i32,
                                self.num_kv_heads as i32,
                            );
                        }
                        out_proj.forward(ctx, &arena.buf_attn, n, &arena.buf_tmp);
                    },
                }
                if let Some(dir) = debug_dir.as_deref() {
                    self.dump_f16(Some(dir), &format!("mixer{li:02}.f32"), &arena.buf_tmp, n * dim);
                }
                unsafe {
                    metal_vision_add_residual_fp16(ctx.raw(), arena.buf_x.ptr(), arena.buf_tmp.ptr(), (n * dim) as i32);
                }
                if profile {
                    lap(&format!("mixer{li:02}"), &mut lap_start);
                }

                self.rms_norm(&arena.buf_x, &layer.pre_mlp_norm, &arena.buf_normed, n);
                layer.mlp_up.forward(ctx, &arena.buf_normed, n, &arena.buf_mlp_fused);
                unsafe {
                    metal_vision_lfm3_swiglu_fp16(
                        ctx.raw(),
                        arena.buf_mlp_fused.ptr(),
                        arena.buf_mlp_gated.ptr(),
                        n as i32,
                        self.hidden_dim as i32,
                    );
                }
                layer.mlp_down.forward(ctx, &arena.buf_mlp_gated, n, &arena.buf_tmp);
                if let Some(dir) = debug_dir.as_deref() {
                    self.dump_f16(Some(dir), &format!("ffn{li:02}.f32"), &arena.buf_tmp, n * dim);
                }
                unsafe {
                    metal_vision_add_residual_fp16(ctx.raw(), arena.buf_x.ptr(), arena.buf_tmp.ptr(), (n * dim) as i32);
                }
                if profile {
                    lap(&format!("ffn{li:02}"), &mut lap_start);
                }
            }

            lap("layers", &mut lap_start);
            self.projector_ln.forward(ctx, &arena.buf_x, &arena.buf_x, n, dim, self.projector_eps);
            unsafe {
                metal_vision_lfm3_merge_pack_fp16(
                    ctx.raw(),
                    arena.buf_x.ptr(),
                    arena.buf_packed.ptr(),
                    n as i32,
                    dim as i32,
                    self.merge as i32,
                );
            }
            self.projector_linear_1.forward(ctx, &arena.buf_packed, rows, &arena.buf_h1);
            let lin1 = self.projector_linear_1.out_features;
            unsafe { metal_vision_gelu_erf_fp16(ctx.raw(), arena.buf_h1.ptr(), (rows * lin1) as i32) };
            self.projector_linear_2.forward(ctx, &arena.buf_h1, rows, &arena.buf_out);

            unsafe { metal_vision_end_pass(ctx.raw()) };

            let mut raw = vec![f16::ZERO; rows * lin2];
            arena.buf_out.read_f16_slice(&mut raw);
            let out: Vec<f32> = raw.into_iter().map(|v| v.to_f32()).collect();
            if let Some(dir) = debug_dir.as_deref() {
                let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(format!("{dir}/projector.f32"), bytes);
            }
            all.extend(out);
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;
    use crate::vision::lfm3_image_processing::Lfm3ImageProcessor;

    fn load_pixels(
        bundle: &Path,
        image_path: &str,
    ) -> (Vec<f32>, [usize; 3]) {
        let config = Lfm3VisionConfig::load(bundle).unwrap();
        let processor =
            Lfm3ImageProcessor::new(config.patch_size, config.temporal_patch_size, config.spatial_merge_size);
        let image = image::open(image_path).unwrap().to_rgb8();
        processor.process(image.as_raw(), image.width() as usize, image.height() as usize)
    }

    /// M1 parity: patchifier + embedding RMSNorm against the CPU f32 tower.
    /// Env: LFM3_VL_MODEL (bundle), LFM3_VL_IMAGE.
    #[uzu_test]
    fn lfm3_metal_embed_matches_cpu() {
        let Ok(bundle) = std::env::var("LFM3_VL_MODEL") else {
            println!("{{\"skipped\": \"LFM3_VL_MODEL not set\"}}");
            return;
        };
        let image_path = std::env::var("LFM3_VL_IMAGE").expect("LFM3_VL_IMAGE is required");
        let bundle = Path::new(&bundle);
        let (pixels, grid) = load_pixels(bundle, &image_path);

        let cpu = Lfm3VisionTower::load_from_model_dir(bundle).unwrap();
        let n = grid[1] * grid[2];
        let cpu_embed = cpu.embed(&pixels, n);

        let metal = Lfm3MetalVisionTower::from_cpu(&cpu).unwrap();
        let metal_embed = metal.forward_embed(&pixels, grid).unwrap();

        let max_abs = cpu_embed.iter().zip(&metal_embed).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu_embed.iter().map(|v| v.abs()).fold(1e-9f32, f32::max);
        println!(
            "{}",
            serde_json::json!({
                "embed_max_abs_diff": max_abs,
                "embed_max_rel": max_abs / max_ref,
                "patches": n,
            })
        );
        assert!(max_abs < 5e-2, "embed mismatch: max_abs_diff = {max_abs}");
    }

    /// Full-tower parity (projector rows) against the CPU f32 reference.
    /// Env: LFM3_VL_MODEL (bundle), LFM3_VL_IMAGE.
    #[uzu_test]
    fn lfm3_metal_tower_matches_cpu() {
        let Ok(bundle) = std::env::var("LFM3_VL_MODEL") else {
            println!("{{\"skipped\": \"LFM3_VL_MODEL not set\"}}");
            return;
        };
        let image_path = std::env::var("LFM3_VL_IMAGE").expect("LFM3_VL_IMAGE is required");
        let bundle = Path::new(&bundle);
        let (pixels, grid) = load_pixels(bundle, &image_path);

        let cpu = Lfm3VisionTower::load_from_model_dir(bundle).unwrap();
        let cpu_out = cpu.forward(&pixels, grid).unwrap();
        if let Ok(dir) = std::env::var("LFM3_VL_DEBUG_DIR") {
            let bytes: Vec<u8> = cpu_out.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(format!("{dir}/projector.f32"), bytes);
        }

        let metal = Lfm3MetalVisionTower::from_cpu(&cpu).unwrap();
        let t0 = std::time::Instant::now();
        let metal_out = metal.forward(&pixels, grid).unwrap();
        let tower_s = t0.elapsed().as_secs_f64();

        assert_eq!(cpu_out.len(), metal_out.len(), "output size mismatch");
        let max_abs = cpu_out.iter().zip(&metal_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_ref = cpu_out.iter().map(|v| v.abs()).fold(1e-9f32, f32::max);
        println!(
            "{}",
            serde_json::json!({
                "max_abs_diff": max_abs,
                "max_rel": max_abs / max_ref,
                "rows": cpu_out.len() / 2048,
                "tower_s": tower_s,
            })
        );
        assert!(max_abs < 5e-2, "tower mismatch: max_abs_diff = {max_abs}");
    }
}
