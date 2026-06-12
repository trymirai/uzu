use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use super::policy::{self, DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, FP_BLOCK};
use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, device_tier::DeviceTier, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
}

impl GemvSpecialization {
    pub(crate) fn select<TB: AsBufferRangeRef>(
        args: &MatmulArguments<Metal, TB>,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_tier: DeviceTier,
    ) -> Option<GemvSpecialization> {
        if !args.b_transpose || args.b_offset != 0 {
            return None;
        }
        let is_quant = !matches!(args.b, MatmulB::FullPrecision { .. });
        let bad_leading_dimension = if is_quant {
            args.b_leading_dimension.is_some()
        } else {
            args.b_leading_dimension.is_some_and(|ld| ld != args.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if args.d_transform.accumulate && !args.n.is_multiple_of(32) {
            return None;
        }
        if args.d_transform.rht_factors.is_some() && !args.n.is_multiple_of(32) {
            return None;
        }
        if is_quant {
            if args.n < DEFAULT_RESULTS_PER_SIMDGROUP || args.m >= 5 {
                return None;
            }
        } else {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || args.n < DEFAULT_RESULTS_PER_SIMDGROUP || args.m > max_gemv_batch_threshold() {
                return None;
            }
        }

        let bits = args.b.bits_per_b().unwrap_or(0);
        let block_size = if !is_quant {
            FP_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = args.k.is_multiple_of(block_size);
        let has_rht = args.d_transform.rht_factors.is_some();
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let tile = if is_quant && bf16_io {
            policy::quant_tile(args.m, args.n, args.k, bits, has_rht, device_tier)
        } else if is_quant || has_rht {
            // Non-bf16 quant IO and fp+RHT keep the default tile (the only
            // one instantiated for those modes).
            policy::GemvTile {
                num_simdgroups: DEFAULT_NUM_SIMDGROUPS,
                k_split: 1,
                results_per_simdgroup: DEFAULT_RESULTS_PER_SIMDGROUP,
            }
        } else {
            policy::fp_tile(args.m, args.n, args.k, input_aligned, device_tier)
        };
        Some(Self {
            b_prologue: args.b.b_prologue(),
            group_size: args.b.group_size().unwrap_or(0),
            bits,
            output_transform: args.d_transform.mask(),
            input_aligned,
            k_split: tile.k_split,
            results_per_simdgroup: tile.results_per_simdgroup,
            num_simdgroups: tile.num_simdgroups,
        })
    }
}

fn rows_per_threadgroup(
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
) -> u32 {
    (num_simdgroups / k_split) * results_per_simdgroup
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
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.b_prologue,
                    specialization.group_size,
                    specialization.bits,
                    specialization.k_split,
                    specialization.input_aligned,
                    specialization.results_per_simdgroup,
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
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;

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

        let group_count_x = n.div_ceil(rows_per_threadgroup(
            specialization.k_split,
            specialization.results_per_simdgroup,
            specialization.num_simdgroups,
        ));

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
