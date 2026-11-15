use std::{ffi::c_void, path::Path};

use anyhow::{Result, ensure};
use half::f16;

use super::{
    common::f32_to_f16_parallel,
    config::{CanvasConfig, VisionConfig},
    image_processing::{PreprocessedVision, nearest_position_ids, window_index},
    tower::{LinearWeight, VisionTower},
};

#[repr(C)]
pub struct MetalVisionCtx {
    _private: [u8; 0],
}

/// Copyable handle to the ObjC vision context.
///
/// Rust-side signatures carry this handle instead of a raw pointer; FFI calls
/// unwrap it via [`VisionCtxHandle::raw`].
#[derive(Clone, Copy)]
pub struct VisionCtxHandle(*mut MetalVisionCtx);

impl VisionCtxHandle {
    /// # Safety
    ///
    /// `ptr` must be a context returned by `metal_vision_create_context` and
    /// must not be used after `metal_vision_destroy_context`.
    pub unsafe fn from_raw(ptr: *mut MetalVisionCtx) -> Self {
        Self(ptr)
    }

    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    pub(crate) fn raw(self) -> *mut MetalVisionCtx {
        self.0
    }
}

#[allow(dead_code)]
unsafe extern "C" {
    fn metal_vision_create_context() -> *mut MetalVisionCtx;
    fn metal_vision_destroy_context(ctx: *mut MetalVisionCtx);
    fn metal_vision_begin_pass(ctx: *mut MetalVisionCtx);
    fn metal_vision_end_pass(ctx: *mut MetalVisionCtx);

    fn metal_vision_alloc_buffer(
        ctx: *mut MetalVisionCtx,
        bytes: usize,
        initial_data: *const c_void,
    ) -> *mut c_void;
    fn metal_vision_free_buffer(buf: *mut c_void);
    fn metal_vision_read_buffer(
        buf: *mut c_void,
        dst: *mut c_void,
        bytes: usize,
    );
    fn metal_vision_write_buffer(
        buf: *mut c_void,
        src: *const c_void,
        bytes: usize,
    );

    fn metal_vision_gemm_fp16(
        ctx: *mut MetalVisionCtx,
        a_buf: *mut c_void,
        m: i32,
        k: i32,
        b_buf: *mut c_void,
        n: i32,
        bias_buf: *mut c_void,
        c_buf: *mut c_void,
    );

    fn metal_vision_layer_norm_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        gamma_buf: *mut c_void,
        beta_buf: *mut c_void,
        out_buf: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    );

    fn metal_vision_gelu_tanh_fp16(
        ctx: *mut MetalVisionCtx,
        x_buf: *mut c_void,
        num_elements: i32,
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
    fn metal_vision_add_pos_embed_indexed_fp16(
        ctx: *mut MetalVisionCtx,
        h_buf: *mut c_void,
        pos_table_buf: *mut c_void,
        pos_indices_buf: *mut c_void,
        num_patches: i32,
        hidden_size: i32,
    );

    fn metal_vision_attention_qkv_fp16(
        ctx: *mut MetalVisionCtx,
        qkv_buf: *mut c_void,
        seg_lens: *const i32,
        num_segs: i32,
        total_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        out_buf: *mut c_void,
    );

    fn metal_vision_window_gather_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        indices: *const u32,
        out_buf: *mut c_void,
        total_tokens: i32,
        hidden_size: i32,
    );

    fn metal_vision_window_pool_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        out_buf: *mut c_void,
        num_windows: i32,
        hidden_size: i32,
        window_size: i32,
    );

    fn metal_vision_top_merger_regroup_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        out_buf: *mut c_void,
        in_tok_offset: i32,
        out_tok_offset: i32,
        h: i32,
        w: i32,
        hidden_size: i32,
    );

    fn metal_vision_window_attention_qkv_fp16(
        ctx: *mut MetalVisionCtx,
        qkv_buf: *mut c_void,
        out_buf: *mut c_void,
        num_windows: i32,
        num_heads: i32,
        head_dim: i32,
    );
    fn metal_vision_fc1_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        m: i32,
        k: i32,
        weight_buf: *mut c_void,
        n: i32,
        bias_buf: *mut c_void,
        out_buf: *mut c_void,
    );
    fn metal_vision_gemm_add_residual_fp16(
        ctx: *mut MetalVisionCtx,
        in_buf: *mut c_void,
        m: i32,
        k: i32,
        weight_buf: *mut c_void,
        n: i32,
        bias_buf: *mut c_void,
        c_buf: *mut c_void,
        res_buf: *mut c_void,
    );
}

pub struct MetalGpuBuffer {
    ptr: *mut c_void,
    pub bytes: usize,
}

unsafe impl Send for MetalGpuBuffer {}
unsafe impl Sync for MetalGpuBuffer {}

impl Drop for MetalGpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { metal_vision_free_buffer(self.ptr) };
        }
    }
}

impl MetalGpuBuffer {
    pub fn new_u32(
        ctx: VisionCtxHandle,
        bytes: usize,
        initial_data: Option<&[u32]>,
    ) -> Self {
        let ptr = unsafe {
            let data_ptr = initial_data.map(|d| d.as_ptr() as *const c_void).unwrap_or(std::ptr::null());
            metal_vision_alloc_buffer(ctx.raw(), bytes, data_ptr)
        };
        assert!(!ptr.is_null(), "Failed to allocate Metal GPU buffer of {} bytes", bytes);
        Self {
            ptr,
            bytes,
        }
    }

    pub fn new_f16(
        ctx: VisionCtxHandle,
        bytes: usize,
        initial_data: Option<&[f16]>,
    ) -> Self {
        let ptr = unsafe {
            let data_ptr = initial_data.map(|d| d.as_ptr() as *const c_void).unwrap_or(std::ptr::null());
            metal_vision_alloc_buffer(ctx.raw(), bytes, data_ptr)
        };
        assert!(!ptr.is_null(), "Failed to allocate Metal GPU buffer of {} bytes", bytes);
        Self {
            ptr,
            bytes,
        }
    }

    pub fn write_f32(
        &self,
        src: &[f32],
    ) {
        assert!(src.len() * 4 <= self.bytes);
        unsafe {
            metal_vision_write_buffer(self.ptr, src.as_ptr() as *const c_void, src.len() * 4);
        }
    }

    /// Raw buffer handle, for kernels declared outside this module.
    pub(crate) fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn new_f32(
        ctx: VisionCtxHandle,
        bytes: usize,
        initial_data: Option<&[f32]>,
    ) -> Self {
        let ptr = unsafe {
            let data_ptr = initial_data.map(|d| d.as_ptr() as *const c_void).unwrap_or(std::ptr::null());
            metal_vision_alloc_buffer(ctx.raw(), bytes, data_ptr)
        };
        assert!(!ptr.is_null(), "Failed to allocate Metal GPU buffer of {} bytes", bytes);
        Self {
            ptr,
            bytes,
        }
    }

    /// Reads a prefix of the buffer (buffers are arena-sized, reads are exact).
    pub fn read_f16_slice(
        &self,
        dst: &mut [f16],
    ) {
        assert!(dst.len() * 2 <= self.bytes);
        unsafe {
            metal_vision_read_buffer(self.ptr, dst.as_mut_ptr() as *mut c_void, dst.len() * 2);
        }
    }

    pub fn write_f16(
        &self,
        src: &[f16],
    ) {
        assert_eq!(self.bytes, src.len() * 2);
        unsafe {
            metal_vision_write_buffer(self.ptr, src.as_ptr() as *const c_void, self.bytes);
        }
    }
}

pub struct MetalLinearWeight {
    pub weight: MetalGpuBuffer,
    pub bias: Option<MetalGpuBuffer>,
    pub in_features: usize,
    pub out_features: usize,
}

impl MetalLinearWeight {
    pub fn from_cpu(
        ctx: VisionCtxHandle,
        lin: &LinearWeight,
    ) -> Self {
        let weight_f16: Vec<f16> = lin.weight.iter().map(|&x| f16::from_f32(x)).collect();
        let weight = MetalGpuBuffer::new_f16(ctx, weight_f16.len() * 2, Some(&weight_f16));
        let bias = lin.bias.as_ref().map(|b| {
            let b_f16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            MetalGpuBuffer::new_f16(ctx, b_f16.len() * 2, Some(&b_f16))
        });
        Self {
            weight,
            bias,
            in_features: lin.in_features,
            out_features: lin.out_features,
        }
    }

    pub fn forward(
        &self,
        ctx: VisionCtxHandle,
        a: &MetalGpuBuffer,
        m: usize,
        c: &MetalGpuBuffer,
    ) {
        let bias_ptr = self.bias.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
        unsafe {
            metal_vision_gemm_fp16(
                ctx.raw(),
                a.ptr,
                m as i32,
                self.in_features as i32,
                self.weight.ptr,
                self.out_features as i32,
                bias_ptr,
                c.ptr,
            );
        }
    }
}

pub struct MetalLayerNormWeights {
    pub gamma: MetalGpuBuffer,
    pub beta: Option<MetalGpuBuffer>,
}

impl MetalLayerNormWeights {
    pub fn from_cpu(
        ctx: VisionCtxHandle,
        ln: &super::tower::LayerNormWeights,
    ) -> Self {
        let gamma_f16: Vec<f16> = ln.gamma.iter().map(|&x| f16::from_f32(x)).collect();
        let gamma = MetalGpuBuffer::new_f16(ctx, gamma_f16.len() * 2, Some(&gamma_f16));
        let beta = ln.beta.as_ref().map(|b| {
            let b_f16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            MetalGpuBuffer::new_f16(ctx, b_f16.len() * 2, Some(&b_f16))
        });
        Self {
            gamma,
            beta,
        }
    }

    pub fn forward(
        &self,
        ctx: VisionCtxHandle,
        in_buf: &MetalGpuBuffer,
        out_buf: &MetalGpuBuffer,
        num_tokens: usize,
        hidden_size: usize,
        eps: f32,
    ) {
        let beta_ptr = self.beta.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
        unsafe {
            metal_vision_layer_norm_fp16(
                ctx.raw(),
                in_buf.ptr,
                self.gamma.ptr,
                beta_ptr,
                out_buf.ptr,
                num_tokens as i32,
                hidden_size as i32,
                eps,
            );
        }
    }
}

pub struct MetalSiglipAttentionWeights {
    pub qkv_proj: MetalLinearWeight,
    pub out_proj: MetalLinearWeight,
}

impl MetalSiglipAttentionWeights {
    pub fn from_cpu(
        ctx: VisionCtxHandle,
        attn: &super::tower::SiglipAttentionWeights,
    ) -> Self {
        Self {
            qkv_proj: MetalLinearWeight::from_cpu(ctx, &attn.qkv_proj),
            out_proj: MetalLinearWeight::from_cpu(ctx, &attn.out_proj),
        }
    }
}

pub struct MetalSiglipMLPWeights {
    pub fc1: MetalLinearWeight,
    pub fc2: MetalLinearWeight,
}

impl MetalSiglipMLPWeights {
    pub fn from_cpu(
        ctx: VisionCtxHandle,
        mlp: &super::tower::SiglipMLPWeights,
    ) -> Self {
        Self {
            fc1: MetalLinearWeight::from_cpu(ctx, &mlp.fc1),
            fc2: MetalLinearWeight::from_cpu(ctx, &mlp.fc2),
        }
    }
}

pub struct MetalEncoderLayerWeights {
    pub layer_norm1: MetalLayerNormWeights,
    pub self_attn: MetalSiglipAttentionWeights,
    pub layer_norm2: MetalLayerNormWeights,
    pub mlp: MetalSiglipMLPWeights,
    pub eps: f32,
}

impl MetalEncoderLayerWeights {
    pub fn from_cpu(
        ctx: VisionCtxHandle,
        layer: &super::tower::EncoderLayerWeights,
    ) -> Self {
        Self {
            layer_norm1: MetalLayerNormWeights::from_cpu(ctx, &layer.layer_norm1),
            self_attn: MetalSiglipAttentionWeights::from_cpu(ctx, &layer.self_attn),
            layer_norm2: MetalLayerNormWeights::from_cpu(ctx, &layer.layer_norm2),
            mlp: MetalSiglipMLPWeights::from_cpu(ctx, &layer.mlp),
            eps: layer.eps,
        }
    }
}

pub struct MetalWindowAttentionMergerWeights {
    pub layer_norm1: MetalLayerNormWeights,
    pub self_attn: MetalSiglipAttentionWeights,
    pub pre_norm: MetalLayerNormWeights,
    pub linear_1: MetalLinearWeight,
    pub linear_2: MetalLinearWeight,
    pub window: usize,
    pub eps: f32,
}

pub struct MetalTopMergerWeights {
    pub pre_norm: MetalLayerNormWeights,
    pub linear_1: MetalLinearWeight,
    pub linear_2: MetalLinearWeight,
    pub eps: f32,
}

pub struct TowerScratchArena {
    pub capacity_patches: usize,
    pub buf_in: MetalGpuBuffer,
    pub buf_h: MetalGpuBuffer,
    pub buf_pos_indices: MetalGpuBuffer,
    pub buf_normed: MetalGpuBuffer,
    pub buf_qkv: MetalGpuBuffer,
    pub buf_attn: MetalGpuBuffer,
    pub buf_mlp_hidden: MetalGpuBuffer,
    pub buf_mlp_out: MetalGpuBuffer,
    pub buf_gathered: MetalGpuBuffer,
    pub buf_normed_gathered: MetalGpuBuffer,
    pub buf_patch_res: MetalGpuBuffer,
    pub buf_flat_normed: MetalGpuBuffer,
    pub buf_fc1: MetalGpuBuffer,
    pub buf_vit_out: MetalGpuBuffer,
    pub buf_post_normed: MetalGpuBuffer,
    pub buf_top_flat: MetalGpuBuffer,
    pub buf_top_flat_normed: MetalGpuBuffer,
    pub buf_top_fc1: MetalGpuBuffer,
    pub buf_final: MetalGpuBuffer,
}

pub struct MetalVisionTower {
    ctx: VisionCtxHandle,
    pub vision_config: VisionConfig,
    pub canvas_config: CanvasConfig,
    pub patch_proj: MetalLinearWeight,
    pub buf_position_embedding: MetalGpuBuffer,
    pub layers: Vec<MetalEncoderLayerWeights>,
    pub vit_merger: MetalWindowAttentionMergerWeights,
    pub post_layernorm: MetalLayerNormWeights,
    pub merger: MetalTopMergerWeights,
    arena: std::sync::Mutex<Option<TowerScratchArena>>,
    forward_lock: std::sync::Mutex<()>,
}

unsafe impl Send for MetalVisionTower {}
unsafe impl Sync for MetalVisionTower {}

impl Drop for MetalVisionTower {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { metal_vision_destroy_context(self.ctx.raw()) };
        }
    }
}

impl MetalVisionTower {
    pub fn from_cpu_tower(tower: VisionTower) -> Result<Self> {
        let ctx = unsafe { VisionCtxHandle::from_raw(metal_vision_create_context()) };
        ensure!(!ctx.is_null(), "Failed to create MetalVisionContext");

        let patch_proj = MetalLinearWeight::from_cpu(ctx, &tower.patch_proj);
        let layers = tower.layers.iter().map(|l| MetalEncoderLayerWeights::from_cpu(ctx, l)).collect();

        let vit_merger = MetalWindowAttentionMergerWeights {
            layer_norm1: MetalLayerNormWeights::from_cpu(ctx, &tower.vit_merger.layer_norm1),
            self_attn: MetalSiglipAttentionWeights::from_cpu(ctx, &tower.vit_merger.self_attn),
            pre_norm: MetalLayerNormWeights::from_cpu(ctx, &tower.vit_merger.pre_norm),
            linear_1: MetalLinearWeight::from_cpu(ctx, &tower.vit_merger.linear_1),
            linear_2: MetalLinearWeight::from_cpu(ctx, &tower.vit_merger.linear_2),
            window: tower.vit_merger.window,
            eps: tower.vit_merger.eps,
        };

        let post_layernorm = MetalLayerNormWeights::from_cpu(ctx, &tower.post_layernorm);

        let merger = MetalTopMergerWeights {
            pre_norm: MetalLayerNormWeights::from_cpu(ctx, &tower.merger.mlp.pre_norm),
            linear_1: MetalLinearWeight::from_cpu(ctx, &tower.merger.mlp.linear_1),
            linear_2: MetalLinearWeight::from_cpu(ctx, &tower.merger.mlp.linear_2),
            eps: tower.merger.mlp.eps,
        };

        let pos_f16: Vec<f16> = tower.position_embedding.iter().map(|&x| f16::from_f32(x)).collect();
        let pos_grid = tower.vision_config.image_size / tower.vision_config.patch_size;
        ensure!(
            tower.position_embedding.len() == pos_grid * pos_grid * tower.vision_config.hidden_size,
            "Position embedding holds {} values, expected {pos_grid}x{pos_grid}x{}",
            tower.position_embedding.len(),
            tower.vision_config.hidden_size
        );
        let buf_position_embedding = MetalGpuBuffer::new_f16(ctx, pos_f16.len() * 2, Some(&pos_f16));

        Ok(Self {
            ctx,
            vision_config: tower.vision_config,
            canvas_config: tower.canvas_config,
            patch_proj,
            buf_position_embedding,
            layers,
            vit_merger,
            post_layernorm,
            merger,
            arena: std::sync::Mutex::new(None),
            forward_lock: std::sync::Mutex::new(()),
        })
    }

    fn ensure_arena(
        &self,
        num_patches: usize,
    ) -> std::sync::MutexGuard<'_, Option<TowerScratchArena>> {
        let mut guard = self.arena.lock().unwrap();
        let need_realloc = match &*guard {
            Some(a) => a.capacity_patches < num_patches,
            None => true,
        };
        if need_realloc {
            let cap = num_patches.max(10240);
            let ctx = self.ctx;
            let hidden_size = self.vision_config.hidden_size;
            let intermediate_size = self.vision_config.intermediate_size;
            let in_patch_features = 14 * 14 * 3;
            let window_size = self.vit_merger.window * self.vit_merger.window;
            let flat_dim = window_size * hidden_size;
            let vit_linear1_out = self.vit_merger.linear_1.out_features;
            let vit_linear2_out = self.vit_merger.linear_2.out_features;
            let top_linear1_out = self.merger.linear_1.out_features;
            let out_features = self.merger.linear_2.out_features;

            *guard = Some(TowerScratchArena {
                capacity_patches: cap,
                buf_in: MetalGpuBuffer::new_f16(ctx, cap * in_patch_features * 2, None),
                buf_h: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_pos_indices: MetalGpuBuffer::new_u32(ctx, cap * 4, None),
                buf_normed: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_qkv: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 3 * 2, None),
                buf_attn: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_mlp_hidden: MetalGpuBuffer::new_f16(ctx, cap * intermediate_size * 2, None),
                buf_mlp_out: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_gathered: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_normed_gathered: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_patch_res: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_flat_normed: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * flat_dim * 2, None),
                buf_fc1: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * vit_linear1_out * 2, None),
                buf_vit_out: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * vit_linear2_out * 2, None),
                buf_post_normed: MetalGpuBuffer::new_f16(ctx, cap * hidden_size * 2, None),
                buf_top_flat: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * flat_dim * 2, None),
                buf_top_flat_normed: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * flat_dim * 2, None),
                buf_top_fc1: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * top_linear1_out * 2, None),
                buf_final: MetalGpuBuffer::new_f16(ctx, (cap / 4 + 1) * out_features * 2, None),
            });
        }
        guard
    }

    pub fn load_from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let cpu_tower = VisionTower::load_from_model_dir(model_dir)?;
        Self::from_cpu_tower(cpu_tower)
    }

    pub fn forward_f16(
        &self,
        preprocessed: &PreprocessedVision,
    ) -> Result<Vec<f16>> {
        let _lock = self.forward_lock.lock().unwrap();
        let ctx = self.ctx;
        let in_patch_features = 14 * 14 * 3;
        let num_patches = preprocessed.patches.len() / in_patch_features;
        ensure!(num_patches > 0, "No visual patches to process");

        let hidden_size = self.vision_config.hidden_size; // 1152
        let intermediate_size = self.vision_config.intermediate_size; // 4304
        let num_heads = self.vision_config.num_attention_heads; // 16
        let head_dim = hidden_size / num_heads; // 72

        let mut arena_guard = self.ensure_arena(num_patches);
        let arena = arena_guard.as_mut().unwrap();

        // Begin batched GPU command encoding
        unsafe {
            metal_vision_begin_pass(ctx.raw());
        }

        // 1. Initial patch projection on GPU (FP16)
        let patches_f16: Vec<f16> = f32_to_f16_parallel(&preprocessed.patches);
        unsafe {
            metal_vision_write_buffer(arena.buf_in.ptr, patches_f16.as_ptr() as *const c_void, patches_f16.len() * 2);
        }
        self.patch_proj.forward(ctx, &arena.buf_in, num_patches, &arena.buf_h);

        // 2. Add position embeddings via fused GPU kernel (FP16)
        let mut pos_indices: Vec<u32> = Vec::with_capacity(num_patches);
        for &(_, h, w) in &preprocessed.grids {
            for idx in nearest_position_ids(h, w, 70) {
                pos_indices.push(idx as u32);
            }
        }
        ensure!(pos_indices.len() == num_patches, "Position indices length mismatch");

        unsafe {
            metal_vision_write_buffer(
                arena.buf_pos_indices.ptr,
                pos_indices.as_ptr() as *const c_void,
                pos_indices.len() * 4,
            );
            metal_vision_add_pos_embed_indexed_fp16(
                ctx.raw(),
                arena.buf_h.ptr,
                self.buf_position_embedding.ptr,
                arena.buf_pos_indices.ptr,
                num_patches as i32,
                hidden_size as i32,
            );
        }

        // 3. Encoder layers with window merger spliced in at insert_layer_id
        let mut current_grids = preprocessed.grids.clone();
        let mut segments: Vec<i32> = current_grids.iter().map(|gr| (gr.1 * gr.2) as i32).collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let (active_h, current_num_tokens) = if i <= self.canvas_config.insert_layer_id {
                (&arena.buf_h, num_patches)
            } else {
                (&arena.buf_vit_out, num_patches / 4)
            };

            // LayerNorm 1
            layer.layer_norm1.forward(ctx, active_h, &arena.buf_normed, current_num_tokens, hidden_size, layer.eps);

            // Fused QKV Projection (Single FP16 MPS GEMM: [N, 1152] @ [3456, 1152]^T -> [N, 3456])
            layer.self_attn.qkv_proj.forward(ctx, &arena.buf_normed, current_num_tokens, &arena.buf_qkv);

            // Scaled Dot-Product Attention:
            // Hardware AMX pipeline (MPSMatrixMultiplication + batch_softmax_fp16_kernel):
            // Tested on Apple M3: AMX path takes 2.07s vs 3.65s for pure GPU shader FlashAttention (AMX is 43% faster!).
            // Do NOT replace this call with a pure compute shader loop on Apple Silicon.
            unsafe {
                metal_vision_attention_qkv_fp16(
                    ctx.raw(),
                    arena.buf_qkv.ptr,
                    segments.as_ptr(),
                    segments.len() as i32,
                    current_num_tokens as i32,
                    num_heads as i32,
                    head_dim as i32,
                    arena.buf_attn.ptr,
                );
            }

            // Out proj + residual add (Fused single-pass bias + residual into active_h)
            let out_bias_ptr = layer.self_attn.out_proj.bias.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
            unsafe {
                metal_vision_gemm_add_residual_fp16(
                    ctx.raw(),
                    arena.buf_attn.ptr,
                    current_num_tokens as i32,
                    hidden_size as i32,
                    layer.self_attn.out_proj.weight.ptr,
                    hidden_size as i32,
                    out_bias_ptr,
                    arena.buf_mlp_out.ptr,
                    active_h.ptr,
                );
            }

            // LayerNorm 2
            layer.layer_norm2.forward(ctx, active_h, &arena.buf_normed, current_num_tokens, hidden_size, layer.eps);

            // Fused MLP fc1 + Bias + GELU tanh: buf_mlp_hidden = gelu(fc1(buf_normed) + bias)
            let fc1_bias_ptr = layer.mlp.fc1.bias.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
            unsafe {
                metal_vision_fc1_fp16(
                    ctx.raw(),
                    arena.buf_normed.ptr,
                    current_num_tokens as i32,
                    hidden_size as i32,
                    layer.mlp.fc1.weight.ptr,
                    intermediate_size as i32,
                    fc1_bias_ptr,
                    arena.buf_mlp_hidden.ptr,
                );
            }

            // Fused MLP fc2 + Bias + Residual: active_h += fc2(buf_mlp_hidden) + bias
            let fc2_bias_ptr = layer.mlp.fc2.bias.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
            unsafe {
                metal_vision_gemm_add_residual_fp16(
                    ctx.raw(),
                    arena.buf_mlp_hidden.ptr,
                    current_num_tokens as i32,
                    intermediate_size as i32,
                    layer.mlp.fc2.weight.ptr,
                    hidden_size as i32,
                    fc2_bias_ptr,
                    arena.buf_mlp_out.ptr,
                    active_h.ptr,
                );
            }

            // Check if ViT window merger should be applied
            if i == self.canvas_config.insert_layer_id {
                let window = self.vit_merger.window; // 2
                let window_size = window * window; // 4
                let mut indices: Vec<u32> = Vec::with_capacity(current_num_tokens);
                let mut offset = 0;
                for &(_, gr_h, gr_w) in &current_grids {
                    indices.extend(window_index(gr_h, gr_w, window, offset).iter().map(|&idx| idx as u32));
                    offset += gr_h * gr_w;
                }
                ensure!(indices.len() == current_num_tokens, "Window indices length mismatch");

                unsafe {
                    metal_vision_window_gather_fp16(
                        ctx.raw(),
                        arena.buf_h.ptr,
                        indices.as_ptr(),
                        arena.buf_gathered.ptr,
                        current_num_tokens as i32,
                        hidden_size as i32,
                    );
                }

                // LayerNorm 1 on buf_h
                self.vit_merger.layer_norm1.forward(
                    ctx,
                    &arena.buf_h,
                    &arena.buf_normed,
                    current_num_tokens,
                    hidden_size,
                    self.vit_merger.eps,
                );
                unsafe {
                    metal_vision_window_gather_fp16(
                        ctx.raw(),
                        arena.buf_normed.ptr,
                        indices.as_ptr(),
                        arena.buf_normed_gathered.ptr,
                        current_num_tokens as i32,
                        hidden_size as i32,
                    );
                }

                // Window self attention: fast custom kernel for 2x2 windows (FP16)
                let num_windows = current_num_tokens / window_size;
                self.vit_merger.self_attn.qkv_proj.forward(
                    ctx,
                    &arena.buf_normed_gathered,
                    current_num_tokens,
                    &arena.buf_qkv,
                );
                unsafe {
                    metal_vision_window_attention_qkv_fp16(
                        ctx.raw(),
                        arena.buf_qkv.ptr,
                        arena.buf_attn.ptr,
                        num_windows as i32,
                        num_heads as i32,
                        head_dim as i32,
                    );
                }
                // Out proj + residual add into buf_gathered
                let vm_out_bias_ptr =
                    self.vit_merger.self_attn.out_proj.bias.as_ref().map(|b| b.ptr).unwrap_or(std::ptr::null_mut());
                unsafe {
                    metal_vision_gemm_add_residual_fp16(
                        ctx.raw(),
                        arena.buf_attn.ptr,
                        current_num_tokens as i32,
                        hidden_size as i32,
                        self.vit_merger.self_attn.out_proj.weight.ptr,
                        hidden_size as i32,
                        vm_out_bias_ptr,
                        arena.buf_mlp_out.ptr,
                        arena.buf_gathered.ptr,
                    );
                }

                // Patch residual: mean pool over window
                unsafe {
                    metal_vision_window_pool_fp16(
                        ctx.raw(),
                        arena.buf_gathered.ptr,
                        arena.buf_patch_res.ptr,
                        num_windows as i32,
                        hidden_size as i32,
                        window_size as i32,
                    );
                }

                // pre_norm(flat): flat has shape [num_windows, window_size * hidden_size] = [num_windows, 4608]
                let flat_dim = window_size * hidden_size;
                self.vit_merger.pre_norm.forward(
                    ctx,
                    &arena.buf_gathered,
                    &arena.buf_flat_normed,
                    num_windows,
                    flat_dim,
                    self.vit_merger.eps,
                );

                self.vit_merger.linear_1.forward(ctx, &arena.buf_flat_normed, num_windows, &arena.buf_fc1);
                unsafe {
                    metal_vision_gelu_tanh_fp16(
                        ctx.raw(),
                        arena.buf_fc1.ptr,
                        (num_windows * self.vit_merger.linear_1.out_features) as i32,
                    );
                }

                self.vit_merger.linear_2.forward(ctx, &arena.buf_fc1, num_windows, &arena.buf_vit_out);
                unsafe {
                    metal_vision_add_residual_fp16(
                        ctx.raw(),
                        arena.buf_vit_out.ptr,
                        arena.buf_patch_res.ptr,
                        (num_windows * hidden_size) as i32,
                    );
                }

                let k = self.vit_merger.window;
                current_grids = current_grids.iter().map(|&(t, gr_h, gr_w)| (t, gr_h / k, gr_w / k)).collect();
                segments = current_grids.iter().map(|gr| (gr.1 * gr.2) as i32).collect();
            }
        }

        // 4. Post layer norm
        let num_vit_tokens = num_patches / 4;
        self.post_layernorm.forward(
            ctx,
            &arena.buf_vit_out,
            &arena.buf_post_normed,
            num_vit_tokens,
            hidden_size,
            self.vision_config.layer_norm_eps,
        );

        // 5. Top merger: 2x2 spatial downsampling
        let mut total_out_tokens = 0;
        for &(_, h, w) in &current_grids {
            total_out_tokens += (h / 2) * (w / 2);
        }
        let flat_dim = 4 * hidden_size;

        let mut in_tok_offset = 0;
        let mut out_tok_offset = 0;
        for &(_, h, w) in &current_grids {
            unsafe {
                metal_vision_top_merger_regroup_fp16(
                    ctx.raw(),
                    arena.buf_post_normed.ptr,
                    arena.buf_top_flat.ptr,
                    in_tok_offset,
                    out_tok_offset,
                    h as i32,
                    w as i32,
                    hidden_size as i32,
                );
            }
            in_tok_offset += (h * w) as i32;
            out_tok_offset += ((h / 2) * (w / 2)) as i32;
        }

        // pre_norm
        self.merger.pre_norm.forward(
            ctx,
            &arena.buf_top_flat,
            &arena.buf_top_flat_normed,
            total_out_tokens,
            flat_dim,
            self.merger.eps,
        );

        self.merger.linear_1.forward(ctx, &arena.buf_top_flat_normed, total_out_tokens, &arena.buf_top_fc1);
        unsafe {
            metal_vision_gelu_erf_fp16(
                ctx.raw(),
                arena.buf_top_fc1.ptr,
                (total_out_tokens * self.merger.linear_1.out_features) as i32,
            );
        }

        let out_features = self.merger.linear_2.out_features; // 3584
        self.merger.linear_2.forward(ctx, &arena.buf_top_fc1, total_out_tokens, &arena.buf_final);

        // Commit and wait for the single GPU command buffer
        unsafe {
            metal_vision_end_pass(ctx.raw());
        }

        // Read final visual tokens back from GPU in FP16
        let mut visual_tokens = vec![f16::ZERO; total_out_tokens * out_features];
        unsafe {
            metal_vision_read_buffer(
                arena.buf_final.ptr,
                visual_tokens.as_mut_ptr() as *mut c_void,
                total_out_tokens * out_features * 2,
            );
        }

        Ok(visual_tokens)
    }

    pub fn forward(
        &self,
        preprocessed: &PreprocessedVision,
    ) -> Result<Vec<f32>> {
        let f16_tokens = self.forward_f16(preprocessed)?;
        Ok(f16_tokens.into_iter().map(|x| x.to_f32()).collect())
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn test_attention_qkv_fp16_parity() {
        let ctx = unsafe { VisionCtxHandle::from_raw(metal_vision_create_context()) };
        assert!(!ctx.is_null());

        let num_heads = 16;
        let head_dim = 72;
        let hidden_size = num_heads * head_dim;
        let qkv_stride = 3 * hidden_size;

        let seg_lens = vec![48, 32];
        let num_segs = seg_lens.len();
        let total_tokens: usize = seg_lens.iter().sum();

        let mut qkv = vec![f16::ZERO; total_tokens * qkv_stride];
        for t in 0..total_tokens {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let q_val = ((t * 7 + h * 3 + d) % 17) as f32 / 17.0 - 0.5;
                    let k_val = ((t * 11 + h * 5 + d) % 19) as f32 / 19.0 - 0.5;
                    let v_val = ((t * 13 + h * 2 + d) % 23) as f32 / 23.0 - 0.5;

                    let base = t * qkv_stride + h * head_dim + d;
                    qkv[base] = f16::from_f32(q_val);
                    qkv[base + hidden_size] = f16::from_f32(k_val);
                    qkv[base + 2 * hidden_size] = f16::from_f32(v_val);
                }
            }
        }

        let mut cpu_out = vec![f16::ZERO; total_tokens * hidden_size];
        let alpha = 1.0f32 / (head_dim as f32).sqrt();
        let mut seg_start = 0;
        for &seg_len in &seg_lens {
            for h in 0..num_heads {
                for i in 0..seg_len {
                    let q_tok = seg_start + i;
                    let mut scores = vec![0.0f32; seg_len];
                    let mut max_score = -1e30f32;

                    for j in 0..seg_len {
                        let k_tok = seg_start + j;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            let q_d = qkv[q_tok * qkv_stride + h * head_dim + d].to_f32();
                            let k_d = qkv[k_tok * qkv_stride + hidden_size + h * head_dim + d].to_f32();
                            dot += q_d * k_d;
                        }
                        let s = dot * alpha;
                        scores[j] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    let mut probs = vec![0.0f32; seg_len];
                    for j in 0..seg_len {
                        let p = (scores[j] - max_score).exp();
                        probs[j] = p;
                        sum_exp += p;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for j in 0..seg_len {
                        probs[j] *= inv_sum;
                    }

                    for d in 0..head_dim {
                        let mut out_val = 0.0f32;
                        for j in 0..seg_len {
                            let k_tok = seg_start + j;
                            let v_d = qkv[k_tok * qkv_stride + 2 * hidden_size + h * head_dim + d].to_f32();
                            out_val += probs[j] * v_d;
                        }
                        cpu_out[q_tok * hidden_size + h * head_dim + d] = f16::from_f32(out_val);
                    }
                }
            }
            seg_start += seg_len;
        }

        let qkv_bytes = qkv.len() * 2;
        let out_bytes = cpu_out.len() * 2;
        let gpu_qkv = unsafe { metal_vision_alloc_buffer(ctx.raw(), qkv_bytes, qkv.as_ptr() as *const c_void) };
        let gpu_out = unsafe { metal_vision_alloc_buffer(ctx.raw(), out_bytes, std::ptr::null()) };

        let seg_lens_i32: Vec<i32> = seg_lens.iter().map(|&x| x as i32).collect();
        unsafe {
            metal_vision_attention_qkv_fp16(
                ctx.raw(),
                gpu_qkv,
                seg_lens_i32.as_ptr(),
                num_segs as i32,
                total_tokens as i32,
                num_heads as i32,
                head_dim as i32,
                gpu_out,
            );
        }

        let mut gpu_res = vec![f16::ZERO; total_tokens * hidden_size];
        unsafe {
            metal_vision_read_buffer(gpu_out, gpu_res.as_mut_ptr() as *mut c_void, out_bytes);
            metal_vision_free_buffer(gpu_qkv);
            metal_vision_free_buffer(gpu_out);
            metal_vision_destroy_context(ctx.raw());
        }

        let mut max_diff = 0.0f32;
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i].to_f32() - gpu_res[i].to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(max_diff < 1e-2, "Max difference too large: {}", max_diff);
    }
}
