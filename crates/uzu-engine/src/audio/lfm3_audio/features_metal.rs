//! Metal GPU implementation of the LFM3-Audio speech encoder downsampler and projector.

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    backends::{
        Backend,
        common::{
            BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{
                AudioDownsamplerConvInMetalKernel, AudioDownsamplerDepthwiseConvMetalKernel,
                AudioDownsamplerFlattenMetalKernel, AudioGeluErfMetalKernel, AudioLayerNormMetalKernel,
                AudioSiluMaskMetalKernel,
            },
        },
    },
    data_type::{DataType, DataTypeRepr},
    vision::tower::SafetensorLoader,
};

type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;
type Encoding = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;
type MatmulK = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

struct MetalConv2d {
    weights: GpuBuffer,
    bias: GpuBuffer,
    out_ch: u32,
}

struct MetalLinear {
    weights: GpuBuffer,
    bias: Option<GpuBuffer>,
    out_features: u32,
    in_features: u32,
}

pub struct MetalAudioFeatures {
    matmul: MatmulK,
    conv_in_k: AudioDownsamplerConvInMetalKernel,
    dw_conv_k: AudioDownsamplerDepthwiseConvMetalKernel,
    silu_mask_k: AudioSiluMaskMetalKernel,
    flatten_k: AudioDownsamplerFlattenMetalKernel,
    layer_norm_k: AudioLayerNormMetalKernel,
    gelu_erf_k: AudioGeluErfMetalKernel,
    conv_in: MetalConv2d,
    conv_mid: MetalConv2d,
    lin_mid: MetalLinear,
    conv_end: MetalConv2d,
    lin_end: MetalLinear,
    lin_out: MetalLinear,
    channels: usize,
    proj_gamma: GpuBuffer,
    proj_beta: GpuBuffer,
    proj_linear_1: MetalLinear,
    proj_linear_2: MetalLinear,
    proj_eps: f32,
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

fn load_conv2d(loader: &mut SafetensorLoader, context: &MetalContext, prefix: &str) -> Result<MetalConv2d> {
    let (w, shape) = loader.read_tensor_f32(&format!("{prefix}.weights"))?;
    ensure!(shape.len() == 4, "conv weight {prefix} must be 4D, got {shape:?}");
    let (b, _) = loader.read_tensor_f32(&format!("{prefix}.biases"))?;
    Ok(MetalConv2d {
        weights: upload_bf16(context, &w)?,
        bias: upload_bf16(context, &b)?,
        out_ch: shape[0] as u32,
    })
}

fn load_linear(
    loader: &mut SafetensorLoader,
    context: &MetalContext,
    w_key: &str,
    b_key: &str,
) -> Result<MetalLinear> {
    let (w, shape) = loader.read_tensor_f32(w_key)?;
    ensure!(shape.len() == 2 || shape.len() == 4, "linear weight {w_key} must be 2D/4D, got {shape:?}");
    let out_features = shape[0] as u32;
    let in_features = if shape.len() == 4 {
        ensure!(shape[1] == 1 && shape[2] == 1, "pointwise conv {w_key} must be 1x1, got {shape:?}");
        shape[3] as u32
    } else {
        shape[1] as u32
    };
    let (b, _) = loader.read_tensor_f32(b_key)?;
    Ok(MetalLinear {
        weights: upload_bf16(context, &w)?,
        bias: Some(upload_bf16(context, &b)?),
        out_features,
        in_features,
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

fn alloc_bf16(context: &MetalContext, elems: usize) -> Result<GpuBuffer> {
    context
        .create_buffer_gpu(elems * std::mem::size_of::<bf16>())
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

impl MetalAudioFeatures {
    pub(crate) fn load(
        loader: &mut SafetensorLoader,
        context: &MetalContext,
        num_mel_bins: usize,
        channels: usize,
        hidden: usize,
    ) -> Result<Self> {
        let conv_in = load_conv2d(loader, context, "audio.downsampler.conv_in")?;
        let conv_mid = load_conv2d(loader, context, "audio.downsampler.conv_mid")?;
        let lin_mid = load_linear(loader, context, "audio.downsampler.lin_mid.weights", "audio.downsampler.lin_mid.biases")?;
        let conv_end = load_conv2d(loader, context, "audio.downsampler.conv_end")?;
        let lin_end = load_linear(loader, context, "audio.downsampler.lin_end.weights", "audio.downsampler.lin_end.biases")?;
        let lin_out = load_linear(loader, context, "audio.downsampler.lin_out.weights", "audio.downsampler.lin_out.biases")?;
        ensure!(lin_out.in_features as usize == num_mel_bins * channels / 8);
        ensure!(lin_out.out_features as usize == hidden);

        let (gamma, _) = loader.read_tensor_f32("audio.projector.layer_norm.scales")?;
        let (beta, _) = loader.read_tensor_f32("audio.projector.layer_norm.biases")?;
        let proj_linear_1 = load_linear(loader, context, "audio.projector.linear_1.weights", "audio.projector.linear_1.biases")?;
        let proj_linear_2 = load_linear(loader, context, "audio.projector.linear_2.weights", "audio.projector.linear_2.biases")?;

        Ok(Self {
            matmul: MatmulK::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
            conv_in_k: AudioDownsamplerConvInMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            dw_conv_k: AudioDownsamplerDepthwiseConvMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            silu_mask_k: AudioSiluMaskMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            flatten_k: AudioDownsamplerFlattenMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            layer_norm_k: AudioLayerNormMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            gelu_erf_k: AudioGeluErfMetalKernel::new(context).map_err(|e| anyhow::anyhow!("{e:?}"))?,
            conv_in,
            conv_mid,
            lin_mid,
            conv_end,
            lin_end,
            lin_out,
            channels,
            proj_gamma: upload_bf16(context, &gamma)?,
            proj_beta: upload_bf16(context, &beta)?,
            proj_linear_1,
            proj_linear_2,
            proj_eps: 1e-5,
        })
    }

    /// Encodes the conv downsampler on Metal GPU. Returns `(output_bf16_buf, tokens, scratch_to_keep_alive)`.
    pub(crate) fn encode_downsampler(
        &mut self,
        context: &MetalContext,
        cb: &mut Encoding,
        mel: &[f32],
        num_mel: usize,
        num_frames: usize,
    ) -> Result<(GpuBuffer, usize, Vec<GpuBuffer>)> {
        let ch = self.channels;
        let mel_buf = cb.allocate_constant_from_slice(mel).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let f1 = num_mel.div_ceil(2);
        let t1 = num_frames.div_ceil(2);
        let v1 = num_frames.div_ceil(2);
        let mut y1 = alloc_bf16(context, f1 * t1 * ch)?;
        self.conv_in_k.encode(
            &mel_buf,
            &self.conv_in.weights,
            &self.conv_in.bias,
            &mut y1,
            num_mel as u32,
            num_frames as u32,
            f1 as u32,
            t1 as u32,
            self.conv_in.out_ch,
            v1 as u32,
            cb,
        );

        let f2 = f1.div_ceil(2);
        let t2 = t1.div_ceil(2);
        let v2 = v1.div_ceil(2);
        let mut dw2 = alloc_bf16(context, f2 * t2 * ch)?;
        self.dw_conv_k.encode(
            &y1,
            &self.conv_mid.weights,
            &self.conv_mid.bias,
            &mut dw2,
            f1 as u32,
            t1 as u32,
            f2 as u32,
            t2 as u32,
            ch as u32,
            cb,
        );
        let mut y2 = alloc_bf16(context, f2 * t2 * self.lin_mid.out_features as usize)?;
        matmul_into(&mut self.matmul, &self.lin_mid, &dw2, (f2 * t2) as u32, &mut y2, cb)?;
        let total2 = (f2 * t2 * self.lin_mid.out_features as usize) as u32;
        self.silu_mask_k.encode(&mut y2, total2, t2 as u32, self.lin_mid.out_features, v2 as u32, cb);

        let f3 = f2.div_ceil(2);
        let t3 = t2.div_ceil(2);
        let v3 = v2.div_ceil(2);
        let mut dw3 = alloc_bf16(context, f3 * t3 * ch)?;
        self.dw_conv_k.encode(
            &y2,
            &self.conv_end.weights,
            &self.conv_end.bias,
            &mut dw3,
            f2 as u32,
            t2 as u32,
            f3 as u32,
            t3 as u32,
            ch as u32,
            cb,
        );
        let mut y3 = alloc_bf16(context, f3 * t3 * self.lin_end.out_features as usize)?;
        matmul_into(&mut self.matmul, &self.lin_end, &dw3, (f3 * t3) as u32, &mut y3, cb)?;
        let total3 = (f3 * t3 * self.lin_end.out_features as usize) as u32;
        self.silu_mask_k.encode(&mut y3, total3, t3 as u32, self.lin_end.out_features, v3 as u32, cb);

        let tokens = v3;
        let mut flat = alloc_bf16(context, tokens * ch * f3)?;
        self.flatten_k.encode(&y3, &mut flat, f3 as u32, t3 as u32, ch as u32, tokens as u32, cb);

        let mut out = alloc_bf16(context, tokens * self.lin_out.out_features as usize)?;
        matmul_into(&mut self.matmul, &self.lin_out, &flat, tokens as u32, &mut out, cb)?;

        Ok((out, tokens, vec![y1, dw2, y2, dw3, y3, flat]))
    }

    /// Encodes `LayerNorm -> Linear -> GELU -> Linear` on Metal GPU.
    pub(crate) fn encode_projector(
        &mut self,
        context: &MetalContext,
        cb: &mut Encoding,
        trunk_hidden: impl BufferRef<Backend = Metal>,
        tokens: usize,
    ) -> Result<(GpuBuffer, Vec<GpuBuffer>)> {
        let hidden = self.proj_linear_1.in_features as usize;
        let mut normed = alloc_bf16(context, tokens * hidden)?;
        self.layer_norm_k.encode(
            trunk_hidden,
            &self.proj_gamma,
            &self.proj_beta,
            &mut normed,
            tokens as u32,
            hidden as u32,
            self.proj_eps,
            cb,
        );

        let mid_dim = self.proj_linear_1.out_features as usize;
        let mut mid = alloc_bf16(context, tokens * mid_dim)?;
        matmul_into(&mut self.matmul, &self.proj_linear_1, &normed, tokens as u32, &mut mid, cb)?;
        self.gelu_erf_k.encode(&mut mid, (tokens * mid_dim) as u32, cb);

        let out_dim = self.proj_linear_2.out_features as usize;
        let mut final_out = alloc_bf16(context, tokens * out_dim)?;
        matmul_into(&mut self.matmul, &self.proj_linear_2, &mid, tokens as u32, &mut final_out, cb)?;

        Ok((final_out, vec![normed, mid]))
    }
}
