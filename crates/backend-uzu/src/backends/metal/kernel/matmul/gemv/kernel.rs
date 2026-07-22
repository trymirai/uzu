use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use super::policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK};
use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{
            Metal,
            context::MetalContext,
            device_tier::DeviceTier,
            kernel::{GemvKey, GemvMetalKernel},
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

/// Identifies one GEMV pipeline: the template variant, plus the function constants that
/// are specialized into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemvSpecialization {
    key: GemvKey,
    output_transform: GemmDTransform,
    gathered: bool,
}

impl GemvSpecialization {
    pub(crate) fn select<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        args: &MatmulArguments<'a, 'b, 'd, Metal, TB>,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_tier: DeviceTier,
    ) -> Option<GemvSpecialization> {
        if !args.b_transpose {
            return None;
        }
        let is_quant = !matches!(args.b, MatmulB::FullPrecision { .. });
        let gathered = args.gather_indices.is_some();
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

        let weights = args.b.weights_key()?;
        let bits = weights.bits().unwrap_or(0);
        let block_size = if !is_quant {
            FP_K_BLOCK
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
            policy::DEFAULT_TILE
        } else {
            policy::fp_tile(args.m, args.n, args.k, input_aligned, device_tier)
        };
        let key = GemvKey {
            at: input_data_type,
            bt: weights_data_type,
            dt: output_data_type,
            weights_key: weights,
            k_split: tile.k_split,
            input_aligned,
            results_per_simdgroup: tile.results_per_simdgroup,
            num_simdgroups: tile.num_simdgroups,
        };

        // The tile tables are fleet-tuned data, so a bad row would otherwise surface as a
        // missing pipeline at dispatch. Ask the generated check instead: it is the shader's
        // own CONSTRAINTs, and falling back to GEMM is always correct.
        key.validate().ok()?;

        Some(Self {
            key,
            output_transform: args.d_transform.mask(),
            gathered,
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

/// The data types used to live here as well, but they are part of the key the selector
/// already builds, so the cache no longer needs its own copy.
pub(crate) struct GemvDispatch {
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvDispatch {
    pub(crate) fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let key = specialization.key;
                let (b_prologue, bits, group_size) = key.weights_key.to_template_args();
                let kernel = GemvMetalKernel::new(
                    context,
                    key.at,
                    key.bt,
                    key.dt,
                    b_prologue,
                    group_size,
                    bits,
                    key.k_split,
                    key.input_aligned,
                    key.results_per_simdgroup,
                    key.num_simdgroups,
                    specialization.output_transform,
                    specialization.gathered,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;

        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;

        let group_count_x = n.div_ceil(rows_per_threadgroup(
            specialization.key.k_split,
            specialization.key.results_per_simdgroup,
            specialization.key.num_simdgroups,
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
                    gather_indices,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    soft_cap,
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
                    gather_indices,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    soft_cap,
                    encoder,
                );
            },
        }

        Ok(())
    }
}
