use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                Kernels, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
                matmul::{MatmulArguments, MatmulError, MatmulWeights},
            },
        },
        metal::{Metal, context::MetalContext},
    },
};

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

    pub(crate) fn encode(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;
        let MatmulWeights::Quantized {
            b: weights,
            scales,
            zero_points_or_biases,
            method,
            mode,
            group_size,
            hadamard_factors,
        } = b
        else {
            panic!("QuantGemvKernel requires Quantized weights");
        };

        let bits = match mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        let use_fast = output_dim % 8 == 0 && input_dim % 512 == 0;

        let (zero_points, biases) = match method {
            QuantizationMethod::ScaleZeroPoint => (Some(zero_points_or_biases), None),
            QuantizationMethod::ScaleBias => (None, Some(zero_points_or_biases)),
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
            kernel.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                d,
                hadamard_factors,
                input_dim,
                output_dim,
                batch_dim,
                encoder,
            );
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
            kernel.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                d,
                input_dim,
                output_dim,
                batch_dim,
                encoder,
            );
        }

        Ok(())
    }
}
