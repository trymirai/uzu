use std::mem::size_of;

use anyhow::{Context as _, Result, anyhow};
use half::bf16;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, Context, CpuBuffer, GlobalBufferCpuGpu,
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                quantization::QuantizationMode,
            },
            kernel::matmul::{
                ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                MatmulShape, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection, WeightsLayout,
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            kernel::{HaruAddBiasMetalKernel, matmul::MatmulMetalKernel},
        },
    },
    config::weight_matrix::AnyWeightMatrixSpec,
    data_type::{DataType, DataTypeRepr},
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub type MetalGpuBuffer = <Metal as Backend>::GlobalGpuBuffer;

pub fn upload_slice<T: DataTypeRepr>(
    context: &MetalContext,
    data: &[T],
) -> Result<MetalGpuBuffer> {
    let bytes = data.len().max(1) * size_of::<T>();
    let mut buffer = context.create_buffer_cpu_gpu(bytes).map_err(|e| anyhow!("{e:?}"))?;
    if !data.is_empty() {
        buffer.copyin(data);
    }
    buffer.handoff().map_err(|e| anyhow!("{e:?}"))
}

pub fn upload_f32(
    context: &MetalContext,
    data: &[f32],
) -> Result<MetalGpuBuffer> {
    upload_slice(context, data)
}

pub fn upload_u32(
    context: &MetalContext,
    data: &[u32],
) -> Result<MetalGpuBuffer> {
    upload_slice(context, data)
}

pub fn upload_bf16_from_f32(
    context: &MetalContext,
    data: &[f32],
) -> Result<MetalGpuBuffer> {
    let converted: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    upload_slice(context, &converted)
}

pub fn alloc_zero_bf16(
    context: &MetalContext,
    elems: usize,
) -> Result<MetalGpuBuffer> {
    upload_slice(context, &vec![bf16::ZERO; elems.max(1)])
}

#[derive(Debug, Clone, Copy)]
pub enum BiasKind {
    None,
    Bf16,
    F32,
}

enum HaruLinearStorage {
    Bf16 {
        weight: MetalGpuBuffer,
    },
    Quantized {
        weights: MetalGpuBuffer,
        scales: MetalGpuBuffer,
        biases: MetalGpuBuffer,
        mode: QuantizationMode,
        group_size: u32,
        layout: WeightsLayout<Metal>,
        signed_codes: bool,
    },
}

pub struct HaruLinear {
    storage: HaruLinearStorage,
    bias_bf16: Option<MetalGpuBuffer>,
    bias_f32: Option<MetalGpuBuffer>,
    pub in_features: u32,
    pub out_features: u32,
}

impl HaruLinear {
    pub async fn load_from_tree(
        tree: &ParameterTree<'_, Metal>,
        matmul: &MatmulMetalKernel,
        context: &MetalContext,
        out_features: u32,
        in_features: u32,
        bias_kind: BiasKind,
    ) -> Result<Self> {
        let storage = match tree.metadata::<AnyWeightMatrixSpec>("spec") {
            Ok(AnyWeightMatrixSpec::MLXSpec(spec)) => {
                if spec.bits != 4 && spec.bits != 8 {
                    return Err(anyhow!("Unsupported MLXSpec bits {} (expected 4 or 8)", spec.bits));
                }
                let mode = QuantizationMode::from_unsigned_bits(spec.bits)
                    .ok_or_else(|| anyhow!("Unsupported MLXSpec bits {}", spec.bits))?;
                if out_features % 4 != 0 {
                    return Err(anyhow!(
                        "Quantized GroupOutput out_features={} must be divisible by 4",
                        out_features
                    ));
                }
                let packing_divisor = mode.packing_divisor();
                if !in_features.is_multiple_of(packing_divisor) {
                    return Err(anyhow!(
                        "Quantized in_features={} must be divisible by packing divisor {}",
                        in_features,
                        packing_divisor
                    ));
                }
                let groups = in_features.div_ceil(spec.group_size);
                let mut weights = tree
                    .read_buffer("weights", &[out_features, in_features / packing_divisor], mode.storage_type())
                    .await
                    .map_err(|e| anyhow!("{e:?}"))?;
                let scales = tree
                    .read_buffer("scales", &[groups, out_features], DataType::BF16)
                    .await
                    .map_err(|e| anyhow!("{e:?}"))?;
                let biases = tree
                    .read_buffer("biases", &[groups, out_features], DataType::BF16)
                    .await
                    .map_err(|e| anyhow!("{e:?}"))?;

                let shape = MatmulShape {
                    m: 1,
                    n: out_features,
                    k: in_features,
                    b_transpose: true,
                    b_leading_dimension: None,
                    b_prologue: GemmBPrologueKind::ScaleBiasDequant,
                    b_is_trellis: false,
                    b_bits: Some(spec.bits),
                    b_group_size: Some(spec.group_size),
                    signed_codes: false,
                    a: ActivationFormat::Bf16,
                    gathered: false,
                    params_layout: Some(QuantParamsLayout::GroupOutput),
                    d_transform: GemmDTransform::empty(),
                };
                let (layout, signed_codes) = matmul
                    .optimize_weights_layout(&shape, &mut weights, context)
                    .map_err(|e| anyhow!("{e:?}"))?;

                HaruLinearStorage::Quantized {
                    weights,
                    scales,
                    biases,
                    mode,
                    group_size: spec.group_size,
                    layout,
                    signed_codes,
                }
            },
            Ok(other) => {
                return Err(anyhow!("Unsupported sidecar WeightMatrixSpec: {:?}", other));
            },
            Err(ParameterLoaderError::KeyNotFound(_)) => {
                let weight = tree
                    .read_buffer("weight", &[out_features, in_features], DataType::BF16)
                    .await
                    .map_err(|e| anyhow!("{e:?}"))?;
                HaruLinearStorage::Bf16 {
                    weight,
                }
            },
            Err(err) => return Err(anyhow!("{err:?}")),
        };

        let bias_bf16 = match bias_kind {
            BiasKind::Bf16 => Some(
                tree.read_buffer("bias", &[out_features], DataType::BF16)
                    .await
                    .context("reading bf16 bias")?,
            ),
            _ => None,
        };
        let bias_f32 = match bias_kind {
            BiasKind::F32 => Some(
                tree.read_buffer("bias", &[out_features], DataType::F32)
                    .await
                    .context("reading f32 bias")?,
            ),
            _ => None,
        };

        Ok(Self {
            storage,
            bias_bf16,
            bias_f32,
            in_features,
            out_features,
        })
    }

    pub fn bias_f32(&self) -> Option<&MetalGpuBuffer> {
        self.bias_f32.as_ref()
    }

    pub fn encode_into(
        &self,
        matmul: &mut MatmulMetalKernel,
        add_bias_f32: Option<&HaruAddBiasMetalKernel>,
        input: impl BufferRef<Backend = Metal>,
        m: u32,
        mut output: impl BufferMut<Backend = Metal>,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<()> {
        let d_ops = MatmulDOps {
            bias: self.bias_bf16.as_ref(),
            ..MatmulDOps::none()
        };
        match &self.storage {
            HaruLinearStorage::Bf16 {
                weight,
            } => {
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: input,
                                offset: 0,
                            },
                            b: MatmulB::FullPrecision {
                                b: weight,
                            },
                            b_leading_dimension: None,
                            b_transpose: true,
                            output: MatmulOutput::new(output.reborrow(), d_ops),
                            gather_indices: None::<&MetalGpuBuffer>,
                            m,
                            n: self.out_features,
                            k: self.in_features,
                        },
                        cb,
                    )
                    .map_err(|e| anyhow!("{e:?}"))?;
            },
            HaruLinearStorage::Quantized {
                weights,
                scales,
                biases,
                mode,
                group_size,
                layout,
                signed_codes,
            } => {
                let groups = self.in_features.div_ceil(*group_size);
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: input,
                                offset: 0,
                            },
                            b: MatmulB::Quantized(QuantizedB {
                                codes: weights,
                                scales,
                                correction: QuantizedCorrection::Biases(biases),
                                params: QuantParams::new(QuantParamsLayout::GroupOutput, self.out_features, groups),
                                mode: *mode,
                                group_size: *group_size,
                                signed_codes: *signed_codes,
                                weights_layout: *layout,
                            }),
                            b_leading_dimension: None,
                            b_transpose: true,
                            output: MatmulOutput::new(output.reborrow(), d_ops),
                            gather_indices: None::<&MetalGpuBuffer>,
                            m,
                            n: self.out_features,
                            k: self.in_features,
                        },
                        cb,
                    )
                    .map_err(|e| anyhow!("{e:?}"))?;
            },
        }
        if let (Some(kernel), Some(bias)) = (add_bias_f32, &self.bias_f32) {
            kernel.encode(output, bias, self.out_features, m, cb);
        }
        Ok(())
    }
}
