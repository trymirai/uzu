use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::{
                QuantizationMode,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, kernel::GemvMetalKernel},
    },
};

const FP_PATH: &str = "FpGemv";
const QUANT_PATH: &str = "QuantGemv";

const FP_BLOCK: u32 = 128;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GemvKey {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
}

const QMV_SIMDGROUPS: u32 = 8;
const QMV_RESULTS_PER_SIMDGROUP: u32 = 4;

fn rows_per_threadgroup(k_split: u32) -> u32 {
    (QMV_SIMDGROUPS / k_split) * QMV_RESULTS_PER_SIMDGROUP
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
    } else if n <= 1024 {
        4
    } else if k >= 3072 {
        4
    } else {
        2
    }
}

pub(crate) struct GemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvKey, GemvMetalKernel>,
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
        key: GemvKey,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                    key.b_prologue,
                    key.group_size,
                    key.bits,
                    key.k_split,
                    key.input_aligned,
                    key.output_transform,
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
        match arguments.b {
            MatmulB::FullPrecision {
                ..
            } => self.encode_fp(arguments, encoder),
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            } => self.encode_quant(arguments, encoder),
        }
    }

    fn encode_fp<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let output_mask = arguments.d_transform.mask();
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;
        let weights = match b {
            MatmulB::FullPrecision {
                b: w,
            } => w,
            _ => unreachable!(),
        };
        if !b_transpose || b_offset != 0 || b_leading_dimension.is_some_and(|ld| ld != k) {
            return Err(MatmulError::UnsupportedLayout {
                path: FP_PATH,
            });
        }
        let context = encoder.context();

        let input_aligned = k % FP_BLOCK == 0;
        let k_split = if rht_factors.is_some() {
            1
        } else {
            fp_k_split(n, k, input_aligned)
        };
        let group_count_x = n.div_ceil(rows_per_threadgroup(k_split));
        let key = GemvKey {
            b_prologue: GemmBPrologueKind::FullPrecision,
            group_size: 0,
            bits: 0,
            output_transform: output_mask,
            input_aligned,
            k_split,
        };
        self.get_or_create(context, key)?.encode(
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

        Ok(())
    }

    fn encode_quant<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() || arguments.b_offset != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: QUANT_PATH,
            });
        }

        let output_mask = arguments.d_transform.mask();
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let hadamard_factors = arguments.d_transform.rht_factors;

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
        let (weights, scales, zero_points, biases, b_prologue, mode, group_size) = match b {
            MatmulB::ScaleBiasDequant {
                b: w,
                scales,
                biases,
                mode,
                group_size,
            } => (w, scales, None, Some(biases), GemmBPrologueKind::ScaleBiasDequant, mode, group_size),
            MatmulB::ScaleZeroPointDequant {
                b: w,
                scales,
                zero_points,
                mode,
                group_size,
            } => (w, scales, Some(zero_points), None, GemmBPrologueKind::ScaleZeroPointDequant, mode, group_size),
            MatmulB::ScaleSymmetricDequant {
                b: w,
                scales,
                mode,
                group_size,
            } => (w, scales, None, None, GemmBPrologueKind::ScaleSymmetricDequant, mode, group_size),
            MatmulB::FullPrecision {
                ..
            } => unreachable!(),
        };

        let bits = match mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        if n % 8 != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: QUANT_PATH,
            });
        }

        let block_size = if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = k % block_size == 0;

        let group_count_x = n.div_ceil(rows_per_threadgroup(1));
        let key = GemvKey {
            b_prologue,
            group_size,
            bits,
            output_transform: output_mask,
            input_aligned,
            k_split: 1,
        };
        let context = encoder.context();
        self.get_or_create(context, key)?.encode(
            weights,
            Some(scales),
            zero_points,
            biases,
            (a, a_offset),
            &mut *d,
            output_bias,
            hadamard_factors,
            k,
            n,
            m,
            ab_scale,
            group_count_x,
            encoder,
        );

        Ok(())
    }
}
