use anyhow::{Result, anyhow};
use futures::executor::block_on;
use half::bf16;

use super::{
    config::HaruCodecConfig,
    linear::{BiasKind, HaruLinear, MetalGpuBuffer, alloc_zero_bf16, upload_bf16_from_f32, upload_f32},
};
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, kernel::matmul::MatmulKernel,
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            kernel::{
                HaruAddBiasMetalKernel, HaruLatentAffineMetalKernel, VibeVoiceAddBiasGeluMetalKernel,
                VibeVoiceCausalConvPackMetalKernel, VibeVoiceConvTransposeOverlapMetalKernel,
                VibeVoiceDepthwiseMixerMetalKernel, VibeVoiceFfnResidualBiasMetalKernel, VibeVoiceRmsNormMetalKernel,
                matmul::MatmulMetalKernel,
            },
        },
    },
    data_type::DataType,
    safetensors::ParameterTree,
};

pub struct PingPongBuffer {
    pub a: MetalGpuBuffer,
    pub b: MetalGpuBuffer,
}

impl PingPongBuffer {
    fn new_zero(context: &MetalContext, rows: u32, cols: u32) -> Result<Self> {
        let elems = (rows as usize) * (cols as usize);
        Ok(Self {
            a: alloc_zero_bf16(context, elems)?,
            b: alloc_zero_bf16(context, elems)?,
        })
    }

    fn pair(&mut self, idx: usize) -> (&MetalGpuBuffer, &mut MetalGpuBuffer) {
        if idx & 1 == 0 {
            (&self.a, &mut self.b)
        } else {
            (&self.b, &mut self.a)
        }
    }
}

pub struct CodecEncoderState {
    idx: usize,
    stem: PingPongBuffer,
    stage_dw: Vec<Vec<PingPongBuffer>>,
    down: Vec<PingPongBuffer>,
    head: PingPongBuffer,
    mean_out_bf16_gpu: MetalGpuBuffer,
}

pub struct CodecDecoderState {
    idx: usize,
    stem: PingPongBuffer,
    stage_dw: Vec<Vec<PingPongBuffer>>,
    up: Vec<PingPongBuffer>,
    head: PingPongBuffer,
}

struct ConvNeXtBlock {
    channels: u32,
    norm_w: MetalGpuBuffer,
    dw_w: MetalGpuBuffer,
    dw_b: MetalGpuBuffer,
    gamma: MetalGpuBuffer,
    ffn_norm_w: MetalGpuBuffer,
    fc1: HaruLinear,
    fc2: HaruLinear,
    ffn_gamma: MetalGpuBuffer,
}

impl ConvNeXtBlock {
    async fn load(
        tree: &ParameterTree<'_, Metal>,
        matmul: &MatmulMetalKernel,
        context: &MetalContext,
        channels: u32,
    ) -> Result<Self> {
        let ffn_channels = channels * 4;
        let norm_w =
            tree.subtree("norm").read_buffer("weight", &[channels], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let conv_tree = tree.subtree("conv");
        let dw_w = conv_tree.read_buffer("weight", &[channels, 7], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let dw_b = conv_tree.read_buffer("bias", &[channels], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let gamma = tree.read_buffer("gamma", &[channels], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let ffn_norm_w = tree
            .subtree("ffn_norm")
            .read_buffer("weight", &[channels], DataType::F32)
            .await
            .map_err(|e| anyhow!("{e:?}"))?;
        let fc1 =
            HaruLinear::load_from_tree(&tree.subtree("fc1"), matmul, context, ffn_channels, channels, BiasKind::F32)
                .await?;
        let fc2 =
            HaruLinear::load_from_tree(&tree.subtree("fc2"), matmul, context, channels, ffn_channels, BiasKind::F32)
                .await?;
        let ffn_gamma =
            tree.read_buffer("ffn_gamma", &[channels], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        Ok(Self {
            channels,
            norm_w,
            dw_w,
            dw_b,
            gamma,
            ffn_norm_w,
            fc1,
            fc2,
            ffn_gamma,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        kernels: &CodecKernels,
        matmul: &mut MatmulMetalKernel,
        x_in: impl BufferRef<Backend = Metal>,
        steps: u32,
        norm_eps: f32,
        dw_state: &mut PingPongBuffer,
        state_idx: usize,
        mut x_out: &mut <Metal as Backend>::ScratchBuffer,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<()> {
        let c = self.channels;
        let ffn_c = c * 4;
        let state_steps_max = steps.max(6);

        let mut n_in = cb.allocate_scratch_for_shape(&[steps, c], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        kernels.rms_norm.encode(x_in, &self.norm_w, &mut n_in, steps, c, norm_eps, cb);

        {
            let (st_in, st_out) = dw_state.pair(state_idx);
            kernels.dw_mixer.encode(
                x_in,
                &n_in,
                st_in,
                &self.dw_w,
                &self.dw_b,
                &self.gamma,
                x_out.reborrow(),
                st_out,
                steps,
                c,
                state_steps_max,
                cb,
            );
        }

        let mut n_ffn = cb.allocate_scratch_for_shape(&[steps, c], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        kernels.rms_norm.encode(&*x_out, &self.ffn_norm_w, &mut n_ffn, steps, c, norm_eps, cb);

        let mut h = cb.allocate_scratch_for_shape(&[steps, ffn_c], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.fc1.encode_into(matmul, None, &n_ffn, steps, &mut h, cb)?;
        let fc1_bias = self.fc1.bias_f32().ok_or_else(|| anyhow!("missing fc1 bias"))?;
        kernels.add_bias_gelu.encode(&mut h, fc1_bias, ffn_c, steps, cb);

        let mut fc2_out = cb.allocate_scratch_for_shape(&[steps, c], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.fc2.encode_into(matmul, None, &h, steps, &mut fc2_out, cb)?;
        let fc2_bias = self.fc2.bias_f32().ok_or_else(|| anyhow!("missing fc2 bias"))?;
        kernels.ffn_residual_bias.encode(x_out.reborrow(), &fc2_out, fc2_bias, &self.ffn_gamma, c, steps, cb);
        Ok(())
    }
}

struct CausalConvLayer {
    proj: HaruLinear,
    c_in: u32,
    c_out: u32,
    k_len: u32,
    stride: u32,
}

impl CausalConvLayer {
    async fn load(
        tree: &ParameterTree<'_, Metal>,
        matmul: &MatmulMetalKernel,
        context: &MetalContext,
        c_in: u32,
        c_out: u32,
        k_len: u32,
        stride: u32,
    ) -> Result<Self> {
        let proj = HaruLinear::load_from_tree(tree, matmul, context, c_out, k_len * c_in, BiasKind::F32).await?;
        Ok(Self {
            proj,
            c_in,
            c_out,
            k_len,
            stride,
        })
    }

    fn encode(
        &self,
        kernels: &CodecKernels,
        matmul: &mut MatmulMetalKernel,
        x_in: impl BufferRef<Backend = Metal>,
        in_steps: u32,
        conv_state: &mut PingPongBuffer,
        state_idx: usize,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, u32)> {
        let ctx = self.k_len - self.stride;
        let out_steps = in_steps / self.stride;
        let y_rows = out_steps.max(ctx);
        let mut windows = cb
            .allocate_scratch_for_shape(&[y_rows, self.k_len * self.c_in], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        {
            let (st_in, st_out) = conv_state.pair(state_idx);
            kernels.conv_pack.encode(
                x_in,
                st_in,
                &mut windows,
                st_out,
                in_steps,
                out_steps,
                self.c_in,
                self.k_len,
                self.stride,
                y_rows,
                cb,
            );
        }
        let mut out =
            cb.allocate_scratch_for_shape(&[out_steps, self.c_out], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.proj.encode_into(matmul, Some(&kernels.add_bias), &windows, out_steps, &mut out, cb)?;
        Ok((out, out_steps))
    }
}

struct ConvTransposeUpLayer {
    proj: HaruLinear,
    bias_f32: MetalGpuBuffer,
    c_out: u32,
    stride: u32,
}

impl ConvTransposeUpLayer {
    async fn load(
        tree: &ParameterTree<'_, Metal>,
        matmul: &MatmulMetalKernel,
        context: &MetalContext,
        c_in: u32,
        c_out: u32,
        stride: u32,
    ) -> Result<Self> {
        let proj =
            HaruLinear::load_from_tree(tree, matmul, context, c_out * 2 * stride, c_in, BiasKind::None).await?;
        let bias_f32 = tree.read_buffer("bias", &[c_out], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        Ok(Self {
            proj,
            bias_f32,
            c_out,
            stride,
        })
    }

    fn encode(
        &self,
        kernels: &CodecKernels,
        matmul: &mut MatmulMetalKernel,
        x_in: impl BufferRef<Backend = Metal>,
        t_steps: u32,
        up_state: &mut PingPongBuffer,
        state_idx: usize,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, u32)> {
        let mut matmul_y = cb
            .allocate_scratch_for_shape(&[t_steps, self.c_out * 2 * self.stride], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        self.proj.encode_into(matmul, None, x_in, t_steps, &mut matmul_y, cb)?;

        let out_steps = t_steps * self.stride;
        let mut out =
            cb.allocate_scratch_for_shape(&[out_steps, self.c_out], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        let (st_in, st_out) = up_state.pair(state_idx);
        kernels.conv_transpose.encode(
            &matmul_y,
            &self.bias_f32,
            st_in,
            &mut out,
            st_out,
            t_steps,
            self.c_out,
            self.stride,
            cb,
        );
        Ok((out, out_steps))
    }
}

struct CodecKernels {
    latent_affine: HaruLatentAffineMetalKernel,
    add_bias: HaruAddBiasMetalKernel,
    rms_norm: VibeVoiceRmsNormMetalKernel,
    dw_mixer: VibeVoiceDepthwiseMixerMetalKernel,
    conv_pack: VibeVoiceCausalConvPackMetalKernel,
    conv_transpose: VibeVoiceConvTransposeOverlapMetalKernel,
    add_bias_gelu: VibeVoiceAddBiasGeluMetalKernel,
    ffn_residual_bias: VibeVoiceFfnResidualBiasMetalKernel,
}

impl CodecKernels {
    fn new(context: &MetalContext) -> Result<Self> {
        Ok(Self {
            latent_affine: HaruLatentAffineMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            add_bias: HaruAddBiasMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            rms_norm: VibeVoiceRmsNormMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            dw_mixer: VibeVoiceDepthwiseMixerMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            conv_pack: VibeVoiceCausalConvPackMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            conv_transpose: VibeVoiceConvTransposeOverlapMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            add_bias_gelu: VibeVoiceAddBiasGeluMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            ffn_residual_bias: VibeVoiceFfnResidualBiasMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
        })
    }
}

async fn load_stages(
    stages_tree: &ParameterTree<'_, Metal>,
    matmul: &MatmulMetalKernel,
    context: &MetalContext,
    depths: &[u32],
    channels_fn: impl Fn(usize) -> u32,
) -> Result<Vec<Vec<ConvNeXtBlock>>> {
    let mut stages = Vec::with_capacity(depths.len());
    for (s_idx, &depth) in depths.iter().enumerate() {
        let ch = channels_fn(s_idx);
        let st = stages_tree.subtree(&s_idx.to_string());
        let mut blocks = Vec::with_capacity(depth as usize);
        for b_idx in 0..depth {
            blocks.push(ConvNeXtBlock::load(&st.subtree(&b_idx.to_string()), matmul, context, ch).await?);
        }
        stages.push(blocks);
    }
    Ok(stages)
}

#[allow(clippy::too_many_arguments)]
fn encode_convnext_stage(
    blocks: &[ConvNeXtBlock],
    stage_states: &mut [PingPongBuffer],
    kernels: &CodecKernels,
    matmul: &mut MatmulMetalKernel,
    x: &mut <Metal as Backend>::ScratchBuffer,
    steps: u32,
    norm_eps: f32,
    state_idx: usize,
    cb: &mut MetalCommandBufferEncoding,
) -> Result<()> {
    for (block, dw_state) in blocks.iter().zip(stage_states.iter_mut()) {
        let mut next =
            cb.allocate_scratch_for_shape(&[steps, block.channels], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        block.encode(kernels, matmul, &*x, steps, norm_eps, dw_state, state_idx, &mut next, cb)?;
        *x = next;
    }
    Ok(())
}

pub struct VibeVoiceCodec {
    pub config: HaruCodecConfig,
    pub frame_samples: u32,
    enc_stem: CausalConvLayer,
    enc_stages: Vec<Vec<ConvNeXtBlock>>,
    enc_down: Vec<CausalConvLayer>,
    enc_head: CausalConvLayer,
    dec_stem: CausalConvLayer,
    dec_stages: Vec<Vec<ConvNeXtBlock>>,
    dec_up: Vec<ConvTransposeUpLayer>,
    dec_head: CausalConvLayer,
    scale_one: MetalGpuBuffer,
    bias_zero: MetalGpuBuffer,
    matmul: MatmulMetalKernel,
    kernels: CodecKernels,
    silence_latent: Vec<f32>,
}

impl VibeVoiceCodec {
    pub async fn load(
        tree: &ParameterTree<'_, Metal>,
        config: &HaruCodecConfig,
        frame_samples: u32,
        context: &MetalContext,
    ) -> Result<Self> {
        let matmul = MatmulMetalKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        let kernels = CodecKernels::new(context)?;
        let num_up = config.ratios.len();
        let top_ch = config.base_channels << num_up;

        let enc_t = tree.subtree("encoder");
        let enc_stem = CausalConvLayer::load(&enc_t.subtree("stem"), &matmul, context, 1, config.base_channels, 7, 1)
            .await?;
        let enc_stages = load_stages(&enc_t.subtree("stages"), &matmul, context, &config.encoder_depths, |i| {
            config.base_channels << i
        })
        .await?;
        let mut enc_down = Vec::with_capacity(num_up);
        for i in 0..num_up {
            let stride = config.ratios[num_up - 1 - i];
            let c_in = config.base_channels << i;
            let c_out = c_in * 2;
            enc_down.push(
                CausalConvLayer::load(&enc_t.subtree("down").subtree(&i.to_string()), &matmul, context, c_in, c_out, 2 * stride, stride)
                    .await?,
            );
        }
        let enc_head =
            CausalConvLayer::load(&enc_t.subtree("head"), &matmul, context, top_ch, config.latent_dim, 7, 1).await?;

        let dec_t = tree.subtree("decoder");
        let dec_stem =
            CausalConvLayer::load(&dec_t.subtree("stem"), &matmul, context, config.latent_dim, top_ch, 7, 1).await?;
        let dec_stages = load_stages(&dec_t.subtree("stages"), &matmul, context, &config.decoder_depths, |i| {
            top_ch >> i
        })
        .await?;
        let mut dec_up = Vec::with_capacity(num_up);
        for (i, &stride) in config.ratios.iter().enumerate() {
            let c_in = top_ch >> i;
            let c_out = c_in / 2;
            dec_up.push(
                ConvTransposeUpLayer::load(
                    &dec_t.subtree("up").subtree(&i.to_string()),
                    &matmul,
                    context,
                    c_in,
                    c_out,
                    stride,
                )
                .await?,
            );
        }
        let dec_head =
            CausalConvLayer::load(&dec_t.subtree("head"), &matmul, context, config.base_channels, 1, 7, 1).await?;

        let mut codec = Self {
            config: config.clone(),
            frame_samples,
            enc_stem,
            enc_stages,
            enc_down,
            enc_head,
            dec_stem,
            dec_stages,
            dec_up,
            dec_head,
            scale_one: upload_f32(context, &[1.0f32])?,
            bias_zero: upload_f32(context, &[0.0f32])?,
            matmul,
            kernels,
            silence_latent: vec![0.0f32; config.latent_dim as usize],
        };

        let silence_frames = 16usize;
        let zeros = vec![0.0f32; silence_frames * frame_samples as usize];
        let mut st = codec.create_encoder_state(context)?;
        let means = codec.encode_f32_async(&zeros, &mut st, context).await?;
        let dim = config.latent_dim as usize;
        codec.silence_latent = means[(silence_frames - 1) * dim..silence_frames * dim].to_vec();

        Ok(codec)
    }

    pub fn silence_latent(&self) -> &[f32] {
        &self.silence_latent
    }

    pub fn create_encoder_state(&self, context: &MetalContext) -> Result<CodecEncoderState> {
        let mut stage_dw = Vec::with_capacity(self.enc_stages.len());
        for (s_idx, blocks) in self.enc_stages.iter().enumerate() {
            let ch = self.config.base_channels << s_idx;
            let mut v = Vec::with_capacity(blocks.len());
            for _ in blocks {
                v.push(PingPongBuffer::new_zero(context, 6, ch)?);
            }
            stage_dw.push(v);
        }
        let mut down = Vec::with_capacity(self.enc_down.len());
        for d in &self.enc_down {
            down.push(PingPongBuffer::new_zero(context, d.k_len - d.stride, d.c_in)?);
        }
        Ok(CodecEncoderState {
            idx: 0,
            stem: PingPongBuffer::new_zero(context, 6, 1)?,
            stage_dw,
            down,
            head: PingPongBuffer::new_zero(context, 6, self.enc_head.c_in)?,
            mean_out_bf16_gpu: alloc_zero_bf16(context, self.config.latent_dim as usize)?,
        })
    }

    pub fn create_decoder_state(&self, context: &MetalContext) -> Result<CodecDecoderState> {
        let top_ch = self.dec_stem.c_out;
        let mut stage_dw = Vec::with_capacity(self.dec_stages.len());
        for (s_idx, blocks) in self.dec_stages.iter().enumerate() {
            let ch = top_ch >> s_idx;
            let mut v = Vec::with_capacity(blocks.len());
            for _ in blocks {
                v.push(PingPongBuffer::new_zero(context, 6, ch)?);
            }
            stage_dw.push(v);
        }
        let mut up = Vec::with_capacity(self.dec_up.len());
        for u in &self.dec_up {
            up.push(PingPongBuffer::new_zero(context, u.stride, u.c_out)?);
        }
        Ok(CodecDecoderState {
            idx: 0,
            stem: PingPongBuffer::new_zero(context, 6, self.config.latent_dim)?,
            stage_dw,
            up,
            head: PingPongBuffer::new_zero(context, 6, self.config.base_channels)?,
        })
    }

    /// Encode `pcm_bf16` (`[num_frames * frame_samples, 1]`) -> `BF16 [num_frames, latent_dim]`.
    pub fn encode_bf16_into_cb(
        &mut self,
        pcm_bf16: impl BufferRef<Backend = Metal>,
        num_frames: u32,
        state: &mut CodecEncoderState,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let idx = state.idx;
        let in_samples = num_frames * self.frame_samples;
        let (mut x, mut steps) =
            self.enc_stem.encode(&self.kernels, &mut self.matmul, pcm_bf16, in_samples, &mut state.stem, idx, cb)?;

        for s_idx in 0..self.enc_down.len() {
            encode_convnext_stage(
                &self.enc_stages[s_idx],
                &mut state.stage_dw[s_idx],
                &self.kernels,
                &mut self.matmul,
                &mut x,
                steps,
                self.config.norm_eps,
                idx,
                cb,
            )?;
            let (next_x, next_steps) = self.enc_down[s_idx].encode(
                &self.kernels,
                &mut self.matmul,
                &x,
                steps,
                &mut state.down[s_idx],
                idx,
                cb,
            )?;
            x = next_x;
            steps = next_steps;
        }

        let last_stage = self.enc_stages.len() - 1;
        encode_convnext_stage(
            &self.enc_stages[last_stage],
            &mut state.stage_dw[last_stage],
            &self.kernels,
            &mut self.matmul,
            &mut x,
            steps,
            self.config.norm_eps,
            idx,
            cb,
        )?;
        let (mean_bf16, _) =
            self.enc_head.encode(&self.kernels, &mut self.matmul, &x, steps, &mut state.head, idx, cb)?;
        state.idx ^= 1;
        Ok(mean_bf16)
    }

    pub fn encode_f32_into_cb(
        &mut self,
        pcm_f32: &[f32],
        state: &mut CodecEncoderState,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let fs = self.frame_samples as usize;
        if pcm_f32.len() != fs {
            return Err(anyhow!(
                "encode_f32_into_cb expects exactly {} samples, got {}",
                fs,
                pcm_f32.len()
            ));
        }
        let pcm_bf16: Vec<bf16> = pcm_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let pcm_c = cb.allocate_constant_from_slice(&pcm_bf16).map_err(|e| anyhow!("{e:?}"))?;
        let mean = self.encode_bf16_into_cb(&pcm_c, 1, state, cb)?;
        drop(pcm_c);
        Ok(mean)
    }

    pub async fn encode_f32_async(
        &mut self,
        pcm_f32: &[f32],
        state: &mut CodecEncoderState,
        context: &MetalContext,
    ) -> Result<Vec<f32>> {
        let fs = self.frame_samples as usize;
        if pcm_f32.is_empty() || !pcm_f32.len().is_multiple_of(fs) {
            return Err(anyhow!(
                "encode_f32 expects a positive multiple of {} samples, got {}",
                fs,
                pcm_f32.len()
            ));
        }
        let num_frames = (pcm_f32.len() / fs) as u32;
        if num_frames == 1 {
            let mut cb = context.create_command_buffer(Some("vibevoice_encode"), None).map_err(|e| anyhow!("{e:?}"))?;
            let mean_scratch = self.encode_f32_into_cb(pcm_f32, state, &mut cb)?;
            cb.encode_copy(&mean_scratch, &mut state.mean_out_bf16_gpu);
            cb.end_encoding().submit().wait_until_completed().await.map_err(|e| anyhow!("{e:?}"))?;
            let out_bf16: Vec<bf16> = context.copyout(&state.mean_out_bf16_gpu).await.map_err(|e| anyhow!("{e:?}"))?;
            return Ok(out_bf16.into_iter().map(bf16::to_f32).collect());
        }
        let pcm_gpu = upload_bf16_from_f32(context, pcm_f32)?;
        let mut cb = context.create_command_buffer(Some("vibevoice_encode"), None).map_err(|e| anyhow!("{e:?}"))?;
        let mean_scratch = self.encode_bf16_into_cb(&pcm_gpu, num_frames, state, &mut cb)?;
        let mut out_gpu = alloc_zero_bf16(context, (num_frames * self.config.latent_dim) as usize)?;
        cb.encode_copy(&mean_scratch, &mut out_gpu);
        cb.end_encoding().submit().wait_until_completed().await.map_err(|e| anyhow!("{e:?}"))?;
        let out_bf16: Vec<bf16> = context.copyout(&out_gpu).await.map_err(|e| anyhow!("{e:?}"))?;
        Ok(out_bf16.into_iter().map(bf16::to_f32).collect())
    }

    pub fn encode_f32(
        &mut self,
        pcm_f32: &[f32],
        state: &mut CodecEncoderState,
        context: &MetalContext,
    ) -> Result<Vec<f32>> {
        block_on(self.encode_f32_async(pcm_f32, state, context))
    }

    /// Decode `latent_f32` (`[num_frames, latent_dim]` F32) -> `BF16 [num_frames * frame_samples, 1]`.
    pub fn decode_f32_into_cb(
        &mut self,
        latent_f32: impl BufferRef<Backend = Metal>,
        num_frames: u32,
        state: &mut CodecDecoderState,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        self.decode_with_affine_into_cb(latent_f32, num_frames, None, state, cb)
    }

    /// Decode `latent_f32` while applying `(scale, bias)` affine transform before casting to BF16.
    pub fn decode_scaled_f32_into_cb(
        &mut self,
        latent_f32: impl BufferRef<Backend = Metal>,
        num_frames: u32,
        scale_buf: &MetalGpuBuffer,
        bias_buf: &MetalGpuBuffer,
        state: &mut CodecDecoderState,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        self.decode_with_affine_into_cb(latent_f32, num_frames, Some((scale_buf, bias_buf)), state, cb)
    }

    fn decode_with_affine_into_cb(
        &mut self,
        latent_f32: impl BufferRef<Backend = Metal>,
        num_frames: u32,
        affine: Option<(&MetalGpuBuffer, &MetalGpuBuffer)>,
        state: &mut CodecDecoderState,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let total = num_frames * self.config.latent_dim;
        let mut latent_bf16 = cb
            .allocate_scratch_for_shape(&[num_frames, self.config.latent_dim], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        let (scale_buf, bias_buf) = affine.unwrap_or((&self.scale_one, &self.bias_zero));
        self.kernels.latent_affine.encode(latent_f32, scale_buf, bias_buf, &mut latent_bf16, total, cb);

        let idx = state.idx;
        let (mut x, mut steps) =
            self.dec_stem.encode(&self.kernels, &mut self.matmul, &latent_bf16, num_frames, &mut state.stem, idx, cb)?;

        for s_idx in 0..self.dec_up.len() {
            encode_convnext_stage(
                &self.dec_stages[s_idx],
                &mut state.stage_dw[s_idx],
                &self.kernels,
                &mut self.matmul,
                &mut x,
                steps,
                self.config.norm_eps,
                idx,
                cb,
            )?;
            let (next_x, next_steps) =
                self.dec_up[s_idx].encode(&self.kernels, &mut self.matmul, &x, steps, &mut state.up[s_idx], idx, cb)?;
            x = next_x;
            steps = next_steps;
        }

        let last_stage = self.dec_stages.len() - 1;
        encode_convnext_stage(
            &self.dec_stages[last_stage],
            &mut state.stage_dw[last_stage],
            &self.kernels,
            &mut self.matmul,
            &mut x,
            steps,
            self.config.norm_eps,
            idx,
            cb,
        )?;
        let (pcm_bf16, _) =
            self.dec_head.encode(&self.kernels, &mut self.matmul, &x, steps, &mut state.head, idx, cb)?;
        state.idx ^= 1;
        Ok(pcm_bf16)
    }

    pub fn decode_f32(
        &mut self,
        latents_f32: &[f32],
        state: &mut CodecDecoderState,
        context: &MetalContext,
    ) -> Result<Vec<f32>> {
        let dim = self.config.latent_dim as usize;
        if latents_f32.is_empty() || !latents_f32.len().is_multiple_of(dim) {
            return Err(anyhow!(
                "decode_f32 expects a positive multiple of {} floats, got {}",
                dim,
                latents_f32.len()
            ));
        }
        let num_frames = (latents_f32.len() / dim) as u32;
        let lat_gpu = upload_f32(context, latents_f32)?;
        let mut cb = context.create_command_buffer(Some("vibevoice_decode"), None).map_err(|e| anyhow!("{e:?}"))?;
        let pcm_scratch = self.decode_f32_into_cb(&lat_gpu, num_frames, state, &mut cb)?;
        let mut out_gpu = alloc_zero_bf16(context, (num_frames * self.frame_samples) as usize)?;
        cb.encode_copy(&pcm_scratch, &mut out_gpu);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
        let out_bf16: Vec<bf16> = block_on(context.copyout(&out_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        Ok(out_bf16.into_iter().map(bf16::to_f32).collect())
    }

    /// Match `VibeVoiceCodec.sample(mean)` (`haru_mlx`):
    /// `z = mean + scale * (latent_std / 0.8) * noise`.
    pub fn sample_latents(&self, mean: &[f32], scale: f32, unit_noise: &[f32]) -> Vec<f32> {
        let s = scale * (self.config.latent_std / 0.8);
        mean.iter().zip(unit_noise.iter()).map(|(&m, &n)| m + s * n).collect()
    }
}
