use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
            kernel::{
                Kernels, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
                matmul::{MatmulArguments, MatmulB, MatmulDOp, MatmulError},
            },
        },
        metal::{Metal, context::MetalContext},
    },
};

const PATH: &str = "QuantGemv";

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct QmvKey {
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct QmvFastKey {
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    use_hadamard: bool,
}

pub(crate) struct QuantGemvKernel {
    data_type: DataType,
    qmv: HashMap<QmvKey, <<Metal as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel>,
    qmv_fast: HashMap<
        QmvFastKey,
        <<Metal as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel,
    >,
}

impl QuantGemvKernel {
    pub(crate) fn new(
        _context: &MetalContext,
        data_type: DataType,
    ) -> Self {
        Self {
            data_type,
            qmv: HashMap::new(),
            qmv_fast: HashMap::new(),
        }
    }

    pub(crate) fn encode<'a>(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<'a, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        // Quant gemv: SCALE/ACCUMULATE not supported (deferred). BIAS handled
        // as outer post-pass by the dispatcher; we only need to reject those
        // bits. RHT is fused via qmv_fast when the fast path is eligible.
        let mask = MatmulDOp::mask(&arguments.d_transform);
        if mask.contains(GemmDTransform::SCALE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::SCALE,
                path: PATH,
            });
        }
        if mask.contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: PATH,
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
            return Err(MatmulError::UnsupportedLayout {
                path: PATH,
            });
        }

        let hadamard_factors = arguments.d_transform.iter().find_map(|op| op.as_rht());

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
            } => panic!("QuantGemvKernel requires quantized B"),
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
            let key = QmvFastKey {
                group_size,
                bits,
                quant_method: method,
                use_hadamard: hadamard_factors.is_some(),
            };
            let context = encoder.context();
            let kernel = match self.qmv_fast.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let kernel = <<Metal as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
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
            kernel.encode(weights, scales, zero_points, biases, (a, a_offset), d, hadamard_factors, k, n, m, encoder);
        } else {
            if hadamard_factors.is_some() {
                return Err(MatmulError::UnsupportedHadamard);
            }
            let key = QmvKey {
                group_size,
                bits,
                quant_method: method,
            };
            let context = encoder.context();
            let kernel = match self.qmv.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let kernel = <<Metal as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
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
            kernel.encode(weights, scales, zero_points, biases, (a, a_offset), d, k, n, m, encoder);
        }

        Ok(())
    }
}
