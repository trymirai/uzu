use std::collections::{HashMap, hash_map::Entry};

use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const FP_PATH: &str = "FpGemv";
const QUANT_PATH: &str = "QuantGemv";

const FP_BLOCK: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    num_simdgroups: u32,
}

const QMV_RESULTS_PER_SIMDGROUP: u32 = 4;

fn rows_per_threadgroup(
    k_split: u32,
    num_simdgroups: u32,
) -> u32 {
    (num_simdgroups / k_split) * QMV_RESULTS_PER_SIMDGROUP
}

fn fp_k_split(
    n: u32,
    k: u32,
    input_aligned: bool,
) -> u32 {
    if !input_aligned || n >= 4096 {
        1
    } else if k >= 16 * n || n <= 512 {
        8
    } else if n <= 1024 || k >= 3072 {
        4
    } else {
        2
    }
}

pub(crate) struct GemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvDispatch {
    pub(crate) fn new(
        _context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        })
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                    specialization.b_prologue,
                    specialization.group_size,
                    specialization.bits,
                    specialization.k_split,
                    specialization.input_aligned,
                    specialization.num_simdgroups,
                    specialization.output_transform,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });
        let path = if is_quant {
            QUANT_PATH
        } else {
            FP_PATH
        };
        let bad_leading_dimension = if is_quant {
            arguments.b_leading_dimension.is_some()
        } else {
            arguments.b_leading_dimension.is_some_and(|ld| ld != arguments.k)
        };
        if !arguments.b_transpose || arguments.b_offset != 0 || bad_leading_dimension {
            return Err(MatmulError::UnsupportedLayout {
                path,
            });
        }
        if is_quant && !arguments.n.is_multiple_of(8) {
            return Err(MatmulError::UnsupportedLayout {
                path: QUANT_PATH,
            });
        }

        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let output_transform = arguments.d_transform.mask();
        let b_prologue = arguments.b.b_prologue();
        let bits = arguments.b.bits_per_b().unwrap_or(0);
        let group_size = arguments.b.group_size().unwrap_or(0);

        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        let block_size = if !is_quant {
            FP_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = k % block_size == 0;
        let (num_simdgroups, k_split) = if is_quant {
            // Output RHT needs a full 32-row threadgroup (8 simdgroups x 4 rows)
            // so the simdgroup-wide Hadamard covers one 32-element block.
            (
                if rht_factors.is_some() {
                    8u32
                } else {
                    2u32
                },
                1u32,
            )
        } else if rht_factors.is_some() {
            (8u32, 1u32)
        } else {
            (8u32, fp_k_split(n, k, input_aligned))
        };
        let group_count_x = n.div_ceil(rows_per_threadgroup(k_split, num_simdgroups));

        let specialization = GemvSpecialization {
            b_prologue,
            group_size,
            bits,
            output_transform,
            input_aligned,
            k_split,
            num_simdgroups,
        };
        let context = encoder.context();
        let pipeline = self.get_or_create(context, specialization)?;

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                pipeline.encode(
                    weights,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            }) => {
                let (weights, scales, zero_points, biases) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, scales, None, Some(biases)),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, scales, Some(zero_points), None),
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, scales, None, None),
                    MatmulB::FullPrecision {
                        ..
                    } => unreachable!(),
                };
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    encoder,
                );
            },
        }

        Ok(())
    }
}
