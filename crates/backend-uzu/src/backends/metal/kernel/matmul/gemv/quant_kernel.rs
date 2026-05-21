use std::collections::{HashMap, hash_map::Entry};

use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                Kernels, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
                quant_matmul::{
                    QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
                },
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
    qmv_fast: HashMap<QmvFastKey, <<Metal as crate::backends::common::Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel>,
}

impl QuantGemvKernel {
    pub(crate) fn new(
        _context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, QuantizedMatmulError<Metal>> {
        Ok(Self {
            data_type,
            qmv: HashMap::new(),
            qmv_fast: HashMap::new(),
        })
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut Encoder<Metal>,
        arguments: QuantizedMatmulArguments<Metal>,
        configuration: &QuantizedMatmulConfiguration,
    ) -> Result<(), QuantizedMatmulError<Metal>> {
        let bits = match configuration.mode {
            QuantizationMode::U4 => 4u32,
            QuantizationMode::I8 | QuantizationMode::U8 => 8u32,
        };
        let group_size = configuration.group_size as u32;
        let use_fast = configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0;

        let QuantizedMatmulArguments {
            a,
            a_offset,
            b,
            scales,
            zero_points_or_biases,
            output,
            hadamard_factors,
            batch_dim,
        } = arguments;

        let (zero_points, biases) = match configuration.quantization_method {
            QuantizationMethod::ScaleZeroPoint => (Some(zero_points_or_biases), None),
            QuantizationMethod::ScaleBias => (None, Some(zero_points_or_biases)),
        };

        if use_fast {
            let key = QmvFastKey {
                group_size,
                bits,
                quant_method: configuration.quantization_method,
                use_hadamard: configuration.use_hadamard,
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
                        configuration.quantization_method,
                        configuration.use_hadamard,
                    )
                    .map_err(QuantizedMatmulError::BackendError)?;
                    entry.insert(kernel)
                },
            };
            kernel.encode(
                b,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                output,
                hadamard_factors,
                configuration.input_dim as u32,
                configuration.output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        } else {
            if configuration.use_hadamard {
                return Err(QuantizedMatmulError::UnsupportedHadamard);
            }
            let key = QmvKey {
                group_size,
                bits,
                quant_method: configuration.quantization_method,
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
                        configuration.quantization_method,
                    )
                    .map_err(QuantizedMatmulError::BackendError)?;
                    entry.insert(kernel)
                },
            };
            kernel.encode(
                b,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                output,
                configuration.input_dim as u32,
                configuration.output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        }

        Ok(())
    }
}
