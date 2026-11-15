//! Metal GPU implementation of the LFM3-Audio MaskGIT depth decoder.
//!
//! Uploads `DepthWeights` once as BF16 on Metal and pre-allocates shared
//! CPU/GPU I/O buffers so each MaskGIT pass runs in a single command buffer.

use anyhow::{Context as _, Result, ensure};
use futures::executor::block_on;
use half::bf16;

use super::DepthConfig;
use super::depth::DepthWeights;
use crate::{
    backends::{
        Backend,
        common::{
            BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, CpuBuffer, GlobalBufferCpuGpu, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{
                CodecRmsNormMetalKernel, DepthAttentionMetalKernel, DepthEmbedSlotsMetalKernel,
                DepthHeadNormSlotMetalKernel, DepthQkvNormRopeMetalKernel, DepthResidualAddMetalKernel,
                DepthSiluMulFusedMetalKernel,
            },
        },
    },
    data_type::{DataType, DataTypeRepr},
    vision::tower::LinearWeight,
};

type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;
type Scratch = <Metal as Backend>::ScratchBuffer;
type Encoding = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;
type MatmulK = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

struct MetalLinear {
    weights: GpuBuffer,
    bias: Option<GpuBuffer>,
    out_features: u32,
    in_features: u32,
}

struct MetalDepthLayer {
    qkvg_proj: MetalLinear,
    o_proj: MetalLinear,
    w13_proj: MetalLinear,
    w2: MetalLinear,
    query_norm: GpuBuffer,
    key_norm: GpuBuffer,
    pre_mixer_norm: GpuBuffer,
    pre_mlp_norm: GpuBuffer,
}

struct DepthKernels {
    embed_slots: DepthEmbedSlotsMetalKernel,
    qkv_norm_rope: DepthQkvNormRopeMetalKernel,
    attention: DepthAttentionMetalKernel,
    residual_add: DepthResidualAddMetalKernel,
    head_norm_slot: DepthHeadNormSlotMetalKernel,
    rms_norm: CodecRmsNormMetalKernel,
    silu_mul_fused: DepthSiluMulFusedMetalKernel,
}

pub struct MetalDepth {
    config: DepthConfig,
    slots: usize,
    vocab: usize,
    matmul: MatmulK,
    kernels: DepthKernels,
    linear: MetalLinear,
    mask: GpuBuffer,
    layers: Vec<MetalDepthLayer>,
    embeddings_packed: GpuBuffer,
    embedding_weights: Vec<MetalLinear>,
    embedding_norms: Vec<GpuBuffer>,
    embedding_norm: GpuBuffer,
    rope: GpuBuffer,
    hidden_in: GpuBuffer,
    projected: GpuBuffer,
    codes_in: GpuBuffer,
    masked_in: GpuBuffer,
    logits_out: Vec<GpuBuffer>,
    logits_scratch: Vec<f32>,
}

fn upload<T: DataTypeRepr>(context: &MetalContext, data: &[T]) -> Result<GpuBuffer> {
    let mut buffer =
        context.create_buffer_cpu_gpu(std::mem::size_of_val(data)).context("allocating GPU buffer")?;
    buffer.copyin(data);
    buffer.handoff().map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn upload_bf16(context: &MetalContext, data: &[f32]) -> Result<GpuBuffer> {
    let bf16_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    upload(context, &bf16_data)
}

fn upload_linear(context: &MetalContext, w: &LinearWeight) -> Result<MetalLinear> {
    let bias = w.bias.as_ref().map(|b| upload_bf16(context, b)).transpose()?;
    Ok(MetalLinear {
        weights: upload_bf16(context, &w.weight)?,
        bias,
        out_features: w.out_features as u32,
        in_features: w.in_features as u32,
    })
}

fn upload_fused_linear(context: &MetalContext, parts: &[&LinearWeight]) -> Result<MetalLinear> {
    let in_features = parts[0].in_features;
    let out_features: usize = parts.iter().map(|p| p.out_features).sum();
    let mut weights = Vec::with_capacity(out_features * in_features);
    let has_bias = parts.iter().any(|p| p.bias.is_some());
    let mut bias = has_bias.then(|| Vec::with_capacity(out_features));
    for p in parts {
        ensure!(p.in_features == in_features, "fused linear in_features mismatch");
        weights.extend_from_slice(&p.weight);
        if let Some(b) = bias.as_mut() {
            if let Some(pb) = &p.bias {
                b.extend_from_slice(pb);
            } else {
                b.resize(b.len() + p.out_features, 0.0);
            }
        }
    }
    Ok(MetalLinear {
        weights: upload_bf16(context, &weights)?,
        bias: bias.as_deref().map(|b| upload_bf16(context, b)).transpose()?,
        out_features: out_features as u32,
        in_features: in_features as u32,
    })
}

fn matmul_into(
    matmul: &mut MatmulK,
    linear: &MetalLinear,
    input: impl BufferRef<Backend = Metal>,
    m: u32,
    output: impl BufferMut<Backend = Metal>,
    cb: &mut Encoding,
) -> Result<()> {
    matmul.encode(
        MatmulArguments {
            a: MatmulA::FullPrecision { values: input, offset: 0 },
            b: MatmulB::FullPrecision { b: &linear.weights },
            b_leading_dimension: None,
            b_transpose: true,
            output: MatmulOutput::new(
                output,
                MatmulDOps { bias: linear.bias.as_ref(), ..MatmulDOps::none() },
            ),
            gather_indices: None::<&GpuBuffer>,
            m,
            n: linear.out_features,
            k: linear.in_features,
        },
        cb,
    )?;
    Ok(())
}

fn matmul_scratch(
    matmul: &mut MatmulK,
    linear: &MetalLinear,
    input: impl BufferRef<Backend = Metal>,
    m: u32,
    cb: &mut Encoding,
) -> Result<Scratch> {
    let mut out = cb.allocate_scratch_for_shape(&[m, linear.out_features], DataType::BF16)?;
    matmul_into(matmul, linear, input, m, &mut out, cb)?;
    Ok(out)
}

impl MetalDepth {
    pub fn new(context: &MetalContext, weights: &DepthWeights, config: &DepthConfig) -> Result<Self> {
        ensure!(config.head_dim == 32, "MetalDepth requires head_dim=32, got {}", config.head_dim);
        let slots = weights.embeddings.len();
        let hidden = config.hidden;
        let vocab = weights.embeddings[0].embedding.len() / hidden;

        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(MetalDepthLayer {
                qkvg_proj: upload_fused_linear(context, &[&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.gate_proj])?,
                o_proj: upload_linear(context, &layer.o_proj)?,
                w13_proj: upload_fused_linear(context, &[&layer.w1, &layer.w3])?,
                w2: upload_linear(context, &layer.w2)?,
                query_norm: upload_bf16(context, &layer.query_norm)?,
                key_norm: upload_bf16(context, &layer.key_norm)?,
                pre_mixer_norm: upload_bf16(context, &layer.pre_mixer_norm)?,
                pre_mlp_norm: upload_bf16(context, &layer.pre_mlp_norm)?,
            });
        }

        let mut packed_emb = Vec::with_capacity(slots * vocab * hidden);
        let mut embedding_weights = Vec::with_capacity(slots);
        let mut embedding_norms = Vec::with_capacity(slots);
        for cb in &weights.embeddings {
            packed_emb.extend_from_slice(&cb.embedding);
            embedding_weights.push(MetalLinear {
                weights: upload_bf16(context, &cb.embedding)?,
                bias: None,
                out_features: vocab as u32,
                in_features: hidden as u32,
            });
            embedding_norms.push(upload_bf16(context, &cb.norm)?);
        }

        let half_dim = config.head_dim / 2;
        let mut rope_data = Vec::with_capacity(slots * half_dim * 2);
        for pos in 0..slots {
            for k in 0..half_dim {
                let exponent = (2 * k) as f64 / config.head_dim as f64;
                let angle = pos as f64 * config.rope_theta.powf(-exponent);
                rope_data.push(angle.cos() as f32);
                rope_data.push(angle.sin() as f32);
            }
        }

        let matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        )?;
        let kernels = DepthKernels {
            embed_slots: DepthEmbedSlotsMetalKernel::new(context)?,
            qkv_norm_rope: DepthQkvNormRopeMetalKernel::new(context)?,
            attention: DepthAttentionMetalKernel::new(context)?,
            residual_add: DepthResidualAddMetalKernel::new(context)?,
            head_norm_slot: DepthHeadNormSlotMetalKernel::new(context)?,
            rms_norm: CodecRmsNormMetalKernel::new(context)?,
            silu_mul_fused: DepthSiluMulFusedMetalKernel::new(context)?,
        };

        let hidden_in = context.create_buffer_cpu_gpu(weights.linear.in_features * 2)?;
        let projected = context.create_buffer_cpu_gpu(slots * hidden * 2)?;
        let codes_in = context.create_buffer_cpu_gpu(slots * std::mem::size_of::<i32>())?;
        let masked_in = context.create_buffer_cpu_gpu(slots * std::mem::size_of::<u32>())?;
        let mut logits_out = Vec::with_capacity(slots);
        for _ in 0..slots {
            logits_out.push(context.create_buffer_cpu_gpu(vocab * 2)?);
        }

        Ok(Self {
            linear: upload_linear(context, &weights.linear)?,
            mask: upload_bf16(context, &weights.mask)?,
            layers,
            embeddings_packed: upload_bf16(context, &packed_emb)?,
            embedding_weights,
            embedding_norms,
            embedding_norm: upload_bf16(context, &weights.embedding_norm)?,
            rope: upload(context, &rope_data)?,
            hidden_in,
            projected,
            codes_in,
            masked_in,
            logits_out,
            logits_scratch: vec![0.0f32; slots * vocab],
            config: config.clone(),
            slots,
            vocab,
            matmul,
            kernels,
        })
    }

    /// Runs one MaskGIT depth trunk pass on Metal, returning a slice into the
    /// pre-allocated logits scratch buffer (`target_slots.len() * vocab`).
    /// When `hidden_gpu` is `Some`, projects the GPU hidden buffer first (pass 0);
    /// subsequent passes reuse the projected state on GPU.
    pub fn forward_pass_gpu(
        &mut self,
        context: &MetalContext,
        hidden_gpu: Option<&GpuBuffer>,
        codes: &[i64],
        masked: &[bool],
        target_slots: &[usize],
    ) -> Result<&[f32]> {
        let slots = self.slots as u32;
        let hidden_dim = self.config.hidden as u32;
        let vocab = self.vocab as u32;
        let heads = self.config.heads as u32;
        let kv_heads = self.config.kv_heads as u32;
        let eps = self.config.eps as f32;
        let scale = (self.config.head_dim as f32).sqrt().recip();

        {
            let codes_dst = self.codes_in.as_slice_mut::<i32>();
            for (dst, &src) in codes_dst.iter_mut().zip(codes) {
                *dst = src as i32;
            }
            let masked_dst = self.masked_in.as_slice_mut::<u32>();
            for (dst, &src) in masked_dst.iter_mut().zip(masked) {
                *dst = u32::from(src);
            }
        }

        let mut cb = context.create_command_buffer(Some("lfm3_depth"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        if let Some(h_buf) = hidden_gpu {
            matmul_into(&mut self.matmul, &self.linear, h_buf, 1, &mut self.projected, &mut cb)?;
        }

        let mut x = cb.allocate_scratch_for_shape(&[slots, hidden_dim], DataType::BF16)?;
        self.kernels.embed_slots.encode(
            &self.projected,
            &self.mask,
            &self.embeddings_packed,
            &self.codes_in,
            &self.masked_in,
            &mut x,
            slots,
            hidden_dim,
            vocab,
            &mut cb,
        );

        let count = slots * hidden_dim;
        let inter = self.config.intermediate as u32;
        let qkvg_stride = (heads + 2 * kv_heads) * 32 + heads;
        for layer in &self.layers {
            let mut normed = cb.allocate_scratch_for_shape(&[slots, hidden_dim], DataType::BF16)?;
            self.kernels.rms_norm.encode(&x, &layer.pre_mixer_norm, &mut normed, slots, hidden_dim, eps, &mut cb);

            let mut qkvg = matmul_scratch(&mut self.matmul, &layer.qkvg_proj, &normed, slots, &mut cb)?;
            self.kernels.qkv_norm_rope.encode(
                &mut qkvg,
                &layer.query_norm,
                &layer.key_norm,
                &self.rope,
                slots,
                heads,
                kv_heads,
                qkvg_stride,
                eps,
                &mut cb,
            );

            let mut attended = cb.allocate_scratch_for_shape(&[slots, heads * 32], DataType::BF16)?;
            self.kernels.attention.encode(&qkvg, &mut attended, slots, heads, kv_heads, qkvg_stride, scale, &mut cb);

            let o_out = matmul_scratch(&mut self.matmul, &layer.o_proj, &attended, slots, &mut cb)?;
            self.kernels.residual_add.encode(&mut x, &o_out, count, &mut cb);

            self.kernels.rms_norm.encode(&x, &layer.pre_mlp_norm, &mut normed, slots, hidden_dim, eps, &mut cb);
            let w13_out = matmul_scratch(&mut self.matmul, &layer.w13_proj, &normed, slots, &mut cb)?;
            let mut activated = cb.allocate_scratch_for_shape(&[slots, inter], DataType::BF16)?;
            self.kernels.silu_mul_fused.encode(&w13_out, &mut activated, slots, inter, &mut cb);
            let mlp_out = matmul_scratch(&mut self.matmul, &layer.w2, &activated, slots, &mut cb)?;
            self.kernels.residual_add.encode(&mut x, &mlp_out, count, &mut cb);
        }

        let mut final_x = cb.allocate_scratch_for_shape(&[slots, hidden_dim], DataType::BF16)?;
        self.kernels.rms_norm.encode(&x, &self.embedding_norm, &mut final_x, slots, hidden_dim, eps, &mut cb);

        let mut normed_row = cb.allocate_scratch_for_shape(&[1, hidden_dim], DataType::BF16)?;
        for &slot in target_slots {
            self.kernels.head_norm_slot.encode(
                &final_x,
                &self.embedding_norms[slot],
                &mut normed_row,
                slot as u32,
                hidden_dim,
                eps,
                &mut cb,
            );
            matmul_into(
                &mut self.matmul,
                &self.embedding_weights[slot],
                &normed_row,
                1,
                &mut self.logits_out[slot],
                &mut cb,
            )?;
        }

        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let total = target_slots.len() * self.vocab;
        for (idx, &slot) in target_slots.iter().enumerate() {
            let row_slice = self.logits_out[slot].as_slice::<bf16>();
            let dst = &mut self.logits_scratch[idx * self.vocab..(idx + 1) * self.vocab];
            for (d, &v) in dst.iter_mut().zip(row_slice) {
                *d = v.to_f32();
            }
        }
        Ok(&self.logits_scratch[..total])
    }

    pub fn forward_pass(
        &mut self,
        context: &MetalContext,
        hidden: Option<&[f32]>,
        codes: &[i64],
        masked: &[bool],
        target_slots: &[usize],
    ) -> Result<Vec<f32>> {
        if let Some(h) = hidden {
            let dst = self.hidden_in.as_slice_mut::<bf16>();
            for (d, &v) in dst.iter_mut().zip(h) {
                *d = bf16::from_f32(v);
            }
            // Borrow `self.hidden_in` via a temporary `forward_pass_gpu` call is split
            // to satisfy the borrow checker without copying.
            return self.forward_pass_host_hidden(context, codes, masked, target_slots);
        }
        Ok(self.forward_pass_gpu(context, None, codes, masked, target_slots)?.to_vec())
    }

    fn forward_pass_host_hidden(
        &mut self,
        context: &MetalContext,
        codes: &[i64],
        masked: &[bool],
        target_slots: &[usize],
    ) -> Result<Vec<f32>> {
        let mut cb = context.create_command_buffer(Some("lfm3_depth_proj"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        matmul_into(&mut self.matmul, &self.linear, &self.hidden_in, 1, &mut self.projected, &mut cb)?;
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(self.forward_pass_gpu(context, None, codes, masked, target_slots)?.to_vec())
    }
}
