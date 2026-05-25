use std::collections::{HashMap, hash_map::Entry};

use super::spec::GemvSpecialization;
use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Backend, Buffer, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
            kernel::{
                Kernels, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{MatmulGemvMetalKernel, TensorAddBiasMetalKernel},
        },
    },
};

const FP_PATH: &str = "FpGemv";
const QUANT_PATH: &str = "QuantGemv";

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct QmvKey {
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    use_hadamard: bool,
}

pub(crate) struct GemvKernel {
    data_type: DataType,
    fp_pipelines: HashMap<GemvSpecialization, MatmulGemvMetalKernel>,
    qmv: HashMap<QmvKey, <<Metal as Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel>,
    qmv_fast: HashMap<QmvKey, <<Metal as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel>,
    bias_add: TensorAddBiasMetalKernel,
}

impl GemvKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let bias_add = TensorAddBiasMetalKernel::new(context, data_type, true).map_err(MatmulError::BackendError)?;
        let mut kernel = Self {
            data_type,
            fp_pipelines: HashMap::new(),
            qmv: HashMap::new(),
            qmv_fast: HashMap::new(),
            bias_add,
        };
        for &config in GemvSpecialization::precompile_configs(data_type) {
            kernel.fp_get_or_create(context, config)?;
        }
        Ok(kernel)
    }

    fn fp_get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&MatmulGemvMetalKernel, MatmulError<Metal>> {
        match self.fp_pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = MatmulGemvMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.threadgroup_rows,
                    specialization.threadgroup_cols,
                    specialization.threads_per_simdgroup_row,
                    specialization.threads_per_simdgroup_col,
                    specialization.elements_per_thread_row,
                    specialization.elements_per_thread_col,
                    specialization.is_accumulate,
                    specialization.is_bias,
                    specialization.is_hadamard,
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
        if !b_transpose {
            return Err(MatmulError::UnsupportedLayout {
                path: FP_PATH,
            });
        }
        if b_offset != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: FP_PATH,
            });
        }
        if b_leading_dimension.is_some_and(|ld| ld != k) {
            return Err(MatmulError::UnsupportedLayout {
                path: FP_PATH,
            });
        }

        let specialization =
            GemvSpecialization::select(k, n, is_accumulate, output_bias.is_some(), rht_factors.is_some());

        self.fp_get_or_create(encoder.context(), specialization)?.encode(
            weights,
            (a, a_offset),
            output_bias,
            rht_factors,
            &mut *d,
            k,
            n,
            k,
            ab_scale,
            m,
            specialization.output_rows_per_threadgroup(),
            encoder,
        );

        Ok(())
    }

    fn encode_quant<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let mask = arguments.d_transform.mask();
        if mask.contains(GemmDTransform::SCALE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::SCALE,
                path: QUANT_PATH,
            });
        }
        if mask.contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: QUANT_PATH,
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() || arguments.b_offset != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: QUANT_PATH,
            });
        }

        let hadamard_factors = arguments.d_transform.rht_factors;
        let post_bias = arguments.d_transform.bias;

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
        let use_fast = n % 8 == 0 && k % 512 == 0;

        let (zero_points, biases) = match method {
            QuantizationMethod::ScaleZeroPoint => (Some(zp_or_bias), None),
            QuantizationMethod::ScaleBias => (None, Some(zp_or_bias)),
        };

        if use_fast {
            let key = QmvKey {
                group_size,
                bits,
                quant_method: method,
                use_hadamard: hadamard_factors.is_some(),
            };
            let context = encoder.context();
            let kernel = match self.qmv_fast.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let kernel = <<Metal as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                        context,
                        self.data_type,
                        group_size,
                        bits,
                        method,
                        hadamard_factors.is_some(),
                    )
                    .map_err(MatmulError::BackendError)?;
                    entry.insert(kernel)
                },
            };
            kernel.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                &mut *d,
                hadamard_factors,
                k,
                n,
                m,
                encoder,
            );
        } else {
            if hadamard_factors.is_some() {
                return Err(MatmulError::UnsupportedDOp {
                    bit: GemmDTransform::RHT,
                    path: QUANT_PATH,
                });
            }
            let key = QmvKey {
                group_size,
                bits,
                quant_method: method,
                use_hadamard: false,
            };
            let context = encoder.context();
            let kernel = match self.qmv.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let kernel = <<Metal as Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                        context,
                        self.data_type,
                        group_size,
                        bits,
                        method,
                    )
                    .map_err(MatmulError::BackendError)?;
                    entry.insert(kernel)
                },
            };
            kernel.encode(weights, scales, zero_points, biases, (a, a_offset), &mut *d, k, n, m, encoder);
        }

        if let Some(bias) = post_bias {
            self.bias_add.encode(None::<&Allocation<Metal>>, bias, d, n, m * n, encoder);
        }

        Ok(())
    }
}
