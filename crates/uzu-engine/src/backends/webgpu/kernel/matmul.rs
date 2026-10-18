use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, Kernels,
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulKernel, QuantParamsStrides, QuantizedCorrection},
            },
        },
        webgpu::{
            WebGPU, buffer::binding_parts, command_buffer::WebGPUCommandBufferEncoding, context::WebGPUContext,
            error::WebGPUError,
        },
    },
    data_type::DataType,
};

include!(concat!(env!("OUT_DIR"), "/webgpu/matmul/matmul.rs"));

pub struct WebGPUMatmul {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    output_rht: ActivationTransform<WebGPU>,
    bias_add: <<WebGPU as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
}

impl MatmulKernel for WebGPUMatmul {
    type Backend = WebGPU;
    type Error = WebGPUError;
    type WeightsLayout = ();

    fn new(
        context: &WebGPUContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error> {
        for dtype in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(dtype, DataType::F32 | DataType::F16 | DataType::BF16) {
                return Err(WebGPUError::UnsupportedKernelVariant(format!("matmul {dtype:?}")));
            }
        }
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            output_rht: ActivationTransform::output_rht(context, output_data_type, None, true)?,
            bias_add: <<WebGPU as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
                context,
                output_data_type,
                weights_data_type,
                true,
            )?,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            WebGPU,
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
            impl BufferMut<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
        >,
        command: &mut WebGPUCommandBufferEncoding,
    ) -> Result<(), Self::Error> {
        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            output,
            gather_indices,
            m,
            n,
            k,
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset,
        } = a
        else {
            return Err(WebGPUError::NotSupported);
        };
        let mut destination = output.values;
        let row_stride = output.row_stride.unwrap_or(n);
        let ops = output.ops;
        if row_stride < n || (ops.rht_factors.is_some() && row_stride != n) {
            return Err(WebGPUError::NotSupported);
        }
        let (b, scales, quant_biases, zero_points, bits, signed_codes, correction, group_size, strides, zp_strides) =
            match b {
                MatmulB::FullPrecision {
                    b,
                } => {
                    (b, None, None, None, 4, false, 0, 1, QuantParamsStrides::default(), QuantParamsStrides::default())
                },
                MatmulB::Quantized(q) => {
                    let bits = q.bits();
                    if !b_transpose || q.group_size == 0 || !k.is_multiple_of(q.group_size) || !matches!(bits, 4 | 8) {
                        return Err(WebGPUError::NotSupported);
                    }
                    let zp_strides = q.zero_point_strides();
                    let (correction, biases, points) = match q.correction {
                        QuantizedCorrection::Symmetric => (0, None, None),
                        QuantizedCorrection::Biases(biases) => (1, Some(biases), None),
                        QuantizedCorrection::ZeroPoints(points) => (2, None, Some(points)),
                    };
                    (
                        q.codes,
                        Some(q.scales),
                        biases,
                        points,
                        bits,
                        q.signed_codes,
                        correction,
                        q.group_size,
                        q.params.scale_strides(),
                        zp_strides,
                    )
                },
                MatmulB::Trellis {
                    ..
                } => return Err(WebGPUError::NotSupported),
            };
        let bias = ops.bias.filter(|_| ops.rht_factors.is_none());
        let kernel = MatmulSimpleWebGPUKernel::new(
            command.context(),
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            scales.is_some(),
            bits,
            signed_codes,
            correction,
            b_transpose,
            bias.is_some(),
            ops.soft_cap.is_some(),
            gather_indices.is_some(),
        )?;
        kernel.encode(
            a,
            b,
            scales,
            quant_biases,
            zero_points,
            bias,
            gather_indices,
            destination.reborrow(),
            m,
            n,
            k,
            u32::try_from(offset).map_err(|_| WebGPUError::NotSupported)?,
            b_leading_dimension.unwrap_or(if b_transpose {
                k
            } else {
                n
            }),
            row_stride,
            group_size,
            strides.output_stride,
            strides.group_stride,
            zp_strides.output_stride,
            zp_strides.group_stride,
            ops.ab_scale,
            ops.soft_cap.unwrap_or(1.0),
            command,
        );
        if let Some(factors) = ops.rht_factors {
            self.output_rht.encode_fp_in_place(destination.reborrow(), factors, None, m, n, command);
            if let Some(bias) = ops.bias {
                self.bias_add.encode(
                    None::<&<WebGPU as Backend>::ScratchBuffer>,
                    bias,
                    destination.reborrow(),
                    n,
                    m * n,
                    command,
                );
            }
        }
        Ok(())
    }
}
