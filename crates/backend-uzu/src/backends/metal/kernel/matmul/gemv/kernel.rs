use std::collections::{HashMap, hash_map::Entry};

use super::spec::GemvSpecialization;
use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform, matmul::GemvParams},
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, kernel::GemvMetalKernel},
    },
};

const FP_PATH: &str = "FpGemv";
const QUANT_PATH: &str = "QuantGemv";

type UnifiedKernel = GemvMetalKernel;

/// Per-threadgroup tile layout (function-constant specialization). The quant
/// branch hardcodes its own layout and ignores these, so quant uses a single
/// canonical tile; the full-precision branch drives them from the host
/// heuristic in `GemvSpecialization::select`.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GemvTile {
    tg_simd_rows: u32,
    tg_simd_cols: u32,
    sg_thread_rows: u32,
    sg_thread_cols: u32,
    thread_out_rows: u32,
    thread_out_cols: u32,
}

impl GemvTile {
    const QUANT: GemvTile = GemvTile {
        tg_simd_rows: 8,
        tg_simd_cols: 1,
        sg_thread_rows: 1,
        sg_thread_cols: 32,
        thread_out_rows: 4,
        thread_out_cols: 4,
    };

    fn from_spec(specialization: &GemvSpecialization) -> Self {
        Self {
            tg_simd_rows: specialization.threadgroup_rows,
            tg_simd_cols: specialization.threadgroup_cols,
            sg_thread_rows: specialization.threads_per_simdgroup_row,
            sg_thread_cols: specialization.threads_per_simdgroup_col,
            thread_out_rows: specialization.elements_per_thread_row,
            thread_out_cols: specialization.elements_per_thread_col,
        }
    }
}

/// Output transforms gemv branches on. Scale is always applied (via
/// `GemvParams::ab_scale`), so the SCALE bit is intentionally excluded.
fn output_transform(
    accumulate: bool,
    bias: bool,
    rht: bool,
) -> GemmDTransform {
    let mut transform = GemmDTransform::empty();
    transform.set(GemmDTransform::ACCUMULATE, accumulate);
    transform.set(GemmDTransform::BIAS, bias);
    transform.set(GemmDTransform::RHT, rht);
    transform
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GemvKey {
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    tile: GemvTile,
    output_transform: GemmDTransform,
}

pub(crate) struct GemvDispatch {
    data_type: DataType,
    pipelines: HashMap<GemvKey, UnifiedKernel>,
}

impl GemvDispatch {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let mut kernel = Self {
            data_type,
            pipelines: HashMap::new(),
        };
        for &config in GemvSpecialization::precompile_configs(data_type) {
            kernel.get_or_create(
                context,
                GemvKey {
                    group_size: 0,
                    bits: 0,
                    quant_method: QuantizationMethod::ScaleBias,
                    tile: GemvTile::from_spec(&config),
                    output_transform: output_transform(config.is_accumulate, config.is_bias, config.is_hadamard),
                },
            )?;
        }
        Ok(kernel)
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        key: GemvKey,
    ) -> Result<&UnifiedKernel, MatmulError<Metal>> {
        match self.pipelines.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = UnifiedKernel::new(
                    context,
                    self.data_type,
                    key.group_size,
                    key.bits,
                    key.quant_method,
                    key.output_transform,
                    key.tile.tg_simd_rows,
                    key.tile.tg_simd_cols,
                    key.tile.sg_thread_rows,
                    key.tile.sg_thread_cols,
                    key.tile.thread_out_rows,
                    key.tile.thread_out_cols,
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
            } => self.encode_quant(arguments, encoder),
        }
    }

    fn encode_fp<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let is_accumulate = arguments.d_transform.accumulate;
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

        let specialization =
            GemvSpecialization::select(k, n, is_accumulate, output_bias.is_some(), rht_factors.is_some());
        let output_rows_per_threadgroup = specialization.output_rows_per_threadgroup();
        let group_count_x = n.div_ceil(output_rows_per_threadgroup);
        let group_count_y = m;

        let key = GemvKey {
            group_size: 0,
            bits: 0,
            quant_method: QuantizationMethod::ScaleBias,
            tile: GemvTile::from_spec(&specialization),
            output_transform: output_transform(is_accumulate, output_bias.is_some(), rht_factors.is_some()),
        };
        let params = GemvParams {
            in_vec_size: k,
            out_vec_size: n,
            batch_size: m,
            matrix_leading_dimension: k,
            output_rows_per_threadgroup,
            ab_scale,
        };
        let context = encoder.context();
        self.get_or_create(context, key)?.encode(
            weights,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            (a, a_offset),
            &mut *d,
            rht_factors,
            output_bias,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
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

        let ab_scale = arguments.d_transform.ab_scale;
        let is_accumulate = arguments.d_transform.accumulate;
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
        let (weights, scales, zp_or_bias, method, mode, group_size) = match b {
            MatmulB::ScaleBiasDequant {
                b: w,
                scales,
                biases,
                mode,
                group_size,
            } => (w, scales, biases, QuantizationMethod::ScaleBias, mode, group_size),
            MatmulB::ScaleZeroPointDequant {
                b: w,
                scales,
                zero_points,
                mode,
                group_size,
            } => (w, scales, zero_points, QuantizationMethod::ScaleZeroPoint, mode, group_size),
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

        let (zero_points, biases) = match method {
            QuantizationMethod::ScaleZeroPoint => (Some(zp_or_bias), None),
            QuantizationMethod::ScaleBias => (None, Some(zp_or_bias)),
        };

        let group_count_x = m;
        let group_count_y = n.div_ceil(32);

        let key = GemvKey {
            group_size,
            bits,
            quant_method: method,
            tile: GemvTile::QUANT,
            output_transform: output_transform(is_accumulate, output_bias.is_some(), hadamard_factors.is_some()),
        };
        let params = GemvParams {
            in_vec_size: k,
            out_vec_size: n,
            batch_size: m,
            matrix_leading_dimension: k,
            output_rows_per_threadgroup: 0,
            ab_scale,
        };
        let context = encoder.context();
        self.get_or_create(context, key)?.encode(
            weights,
            Some(scales),
            zero_points,
            biases,
            (a, a_offset),
            &mut *d,
            hadamard_factors,
            output_bias,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            encoder,
        );

        Ok(())
    }
}
