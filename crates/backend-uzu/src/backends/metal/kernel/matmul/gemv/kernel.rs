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
        metal::{
            Metal,
            context::{GpuDeviceTier, MetalContext},
            kernel::GemvMetalKernel,
        },
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
pub struct GemvDispatchPath {
    pub k_split: u32,
    pub results_per_simdgroup: u32,
    pub num_simdgroups: u32,
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
        device_tier: GpuDeviceTier,
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
        let k_split = if is_quant || has_rht {
            1
        } else {
            policy::fp_k_split(args.m, args.n, args.k, input_aligned, device_tier)
        };
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let (num_simdgroups, results_per_simdgroup) = if is_quant && bf16_io {
            policy::quant_tile(args.m, args.n, args.k, has_rht, device_tier)
        } else if is_quant || has_rht {
            // Non-bf16 quant IO and fp+RHT keep the default tile (the only
            // one instantiated for those modes).
            (DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP)
        } else {
            (DEFAULT_NUM_SIMDGROUPS, policy::fp_results_per_simdgroup(args.m, args.n, args.k, device_tier))
        };
        Some(GemvSpecialization {
            b_prologue: args.b.b_prologue(),
            group_size: args.b.group_size().unwrap_or(0),
            bits,
            output_transform: args.d_transform.mask(),
            input_aligned,
            k_split,
            results_per_simdgroup,
            num_simdgroups,
        })
    }

    pub(crate) fn with_dispatch_path(
        mut self,
        path: GemvDispatchPath,
    ) -> Self {
        assert!(matches!(path.k_split, 1 | 2 | 4 | 8), "GemvDispatchPath::k_split must be one of 1, 2, 4, 8",);
        assert!(
            matches!(path.results_per_simdgroup, 1 | 2 | 4 | 8),
            "GemvDispatchPath::results_per_simdgroup must be one of 1, 2, 4, 8",
        );
        assert!(matches!(path.num_simdgroups, 2 | 4 | 8), "GemvDispatchPath::num_simdgroups must be one of 2, 4, 8",);
        assert!(
            path.num_simdgroups.is_multiple_of(path.k_split),
            "GemvDispatchPath::k_split must divide num_simdgroups",
        );
        self.k_split = path.k_split;
        self.results_per_simdgroup = path.results_per_simdgroup;
        self.num_simdgroups = path.num_simdgroups;
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_dispatch_path_sets_all_forced_path_fields() {
        let specialization = GemvSpecialization {
            b_prologue: GemmBPrologueKind::FullPrecision,
            group_size: 0,
            bits: 0,
            output_transform: GemmDTransform::empty(),
            input_aligned: true,
            k_split: 1,
            results_per_simdgroup: DEFAULT_RESULTS_PER_SIMDGROUP,
            num_simdgroups: DEFAULT_NUM_SIMDGROUPS,
        }
        .with_dispatch_path(GemvDispatchPath {
            k_split: 8,
            results_per_simdgroup: 1,
            num_simdgroups: 8,
        });

        assert_eq!(specialization.k_split, 8);
        assert_eq!(specialization.results_per_simdgroup, 1);
        assert_eq!(specialization.num_simdgroups, 8);
    }
}
