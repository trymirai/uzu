use std::collections::{HashMap, hash_map::Entry};

use super::spec::GemvSpecialization;
use crate::{
    DataType,
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            gpu_types::{
                QuantizationMode,
                gemm::{GemmBPrologueKind, GemmDTransform},
                matmul::{GemvParams, GemvTiling},
            },
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, kernel::{GemvMetalKernel, QmvFastMetalKernel}},
    },
};

const FP_PATH: &str = "FpGemv";
const QUANT_PATH: &str = "QuantGemv";

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GemvFpKey {
    tiling: GemvTiling,
    output_transform: GemmDTransform,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct QmvFastKey {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
}

pub(crate) struct GemvDispatch {
    data_type: DataType,
    fp_pipelines: HashMap<GemvFpKey, GemvMetalKernel>,
    quant_pipelines: HashMap<QmvFastKey, QmvFastMetalKernel>,
}

impl GemvDispatch {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let mut kernel = Self {
            data_type,
            fp_pipelines: HashMap::new(),
            quant_pipelines: HashMap::new(),
        };
        for config in GemvSpecialization::precompile_configs(data_type) {
            kernel.get_or_create_fp(
                context,
                GemvFpKey {
                    tiling: config.tiling,
                    output_transform: config.output_transform(),
                },
            )?;
        }
        Ok(kernel)
    }

    fn get_or_create_fp(
        &mut self,
        context: &MetalContext,
        key: GemvFpKey,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.fp_pipelines.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.data_type,
                    key.output_transform,
                    key.tiling,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_quant(
        &mut self,
        context: &MetalContext,
        key: QmvFastKey,
    ) -> Result<&QmvFastMetalKernel, MatmulError<Metal>> {
        match self.quant_pipelines.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = QmvFastMetalKernel::new(
                    context,
                    self.data_type,
                    key.b_prologue,
                    key.group_size,
                    key.bits,
                    key.output_transform,
                    key.input_aligned,
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
        let output_mask = arguments.d_transform.mask();
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
        let output_rows_per_threadgroup = specialization.tiling.output_rows_per_threadgroup();
        let group_count_x = n.div_ceil(output_rows_per_threadgroup);
        let group_count_y = m;

        let key = GemvFpKey {
            tiling: specialization.tiling,
            output_transform: output_mask,
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
        self.get_or_create_fp(context, key)?.encode(
            (a, a_offset),
            weights,
            &mut *d,
            output_bias,
            rht_factors,
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
        let (weights, scales, zp_or_bias, b_prologue, mode, group_size) = match b {
            MatmulB::ScaleBiasDequant {
                b: w,
                scales,
                biases,
                mode,
                group_size,
            } => (w, scales, biases, GemmBPrologueKind::ScaleBiasDequant, mode, group_size),
            MatmulB::ScaleZeroPointDequant {
                b: w,
                scales,
                zero_points,
                mode,
                group_size,
            } => (w, scales, zero_points, GemmBPrologueKind::ScaleZeroPointDequant, mode, group_size),
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

        let (zero_points, biases) = match b_prologue {
            GemmBPrologueKind::ScaleZeroPointDequant => (Some(zp_or_bias), None),
            GemmBPrologueKind::ScaleBiasDequant => (None, Some(zp_or_bias)),
            GemmBPrologueKind::FullPrecision => unreachable!(),
        };

        // block_size = values_per_thread * 32; values_per_thread = 2 packs ×
        // pack_factor (8 at 4-bit, 4 at 8-bit). Aligned K skips the tail block.
        let block_size = if bits == 4 { 512 } else { 256 };
        let input_aligned = k % block_size == 0;

        let key = QmvFastKey {
            b_prologue,
            group_size,
            bits,
            output_transform: output_mask,
            input_aligned,
        };
        let context = encoder.context();
        self.get_or_create_quant(context, key)?.encode(
            weights,
            scales,
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
            encoder,
        );

        Ok(())
    }
}
