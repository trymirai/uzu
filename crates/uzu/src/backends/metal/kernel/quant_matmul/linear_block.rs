use std::rc::Rc;

use metal::{Buffer as MTLBuffer, MTLResourceOptions};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{QuantizedMatmulArguments, QuantizedMatmulKernel};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::QuantizationConfig,
    parameters::ParameterTree,
};

pub struct QuantizedLinearKernelBlock {
    kernel: QuantizedMatmulKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    input_dim: usize,
    output_dim: usize,
    #[allow(dead_code)]
    group_size: usize,
    #[allow(dead_code)]
    kernel_data_type: DataType,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl QuantizedLinearKernelBlock {
    pub fn new(
        mtl_context: &MTLContext,
        config: &QuantizationConfig,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, MTLError> {
        let kernel_data_type: DataType = config.activation_precision.into();

        if !matches!(kernel_data_type, DataType::F16 | DataType::F32) {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for quantized kernel: {:?}",
                kernel_data_type
            )));
        }

        let mut weights = parameter_tree.leaf("weights").map_err(|e| {
            MTLError::Generic(format!("Failed to load weights: {:?}", e))
        })?;

        if weights.data_type() != DataType::U8 {
            return Err(MTLError::Generic(format!(
                "Expected packed U8 weights, got {:?}",
                weights.data_type()
            )));
        }

        let mut scales = parameter_tree.leaf("scales").map_err(|e| {
            MTLError::Generic(format!("Failed to load scales: {:?}", e))
        })?;

        let zero_points = parameter_tree.leaf("zero_points").map_err(|e| {
            MTLError::Generic(format!("Failed to load zero_points: {:?}", e))
        })?;

        // Only accept new layout for transposed kernels:
        // weights: [N, K/2], scales: [N, K_g], zero_points: [N, ceil(K_g/2)]
        let k_g = (input_dim + config.group_size - 1) / config.group_size;

        let w_shape = weights.shape();
        let s_shape = scales.shape();
        let zp_shape = zero_points.shape();

        if !(w_shape == [output_dim, input_dim / 2]
            && s_shape == [output_dim, k_g]
            && zp_shape == [output_dim, (k_g + 1) / 2])
        {
            return Err(MTLError::Generic(format!(
                "Unexpected shapes. weights={:?}, scales={:?}, zero_points={:?}; expected [N,K/2],[N,K_g],[N,(K_g+1)/2]",
                w_shape, s_shape, zp_shape,
            )));
        }

        let (scales_buffer, biases_buffer) = {
            if scales.data_type() != kernel_data_type {
                return Err(MTLError::Generic(format!(
                    "Scales dtype mismatch: got {:?}, expected {:?}",
                    scales.data_type(),
                    kernel_data_type
                )));
            }
            let scales_buffer = unsafe { scales.mtl_buffer() }.to_owned();
            let biases_buffer = Self::convert_biases(
                mtl_context,
                kernel_data_type,
                config.group_size,
                input_dim,
                output_dim,
                &scales,
                &zero_points,
            )?;
            (scales_buffer, biases_buffer)
        };

        // Weights must already be [N, K/2] and packed correctly
        let weights_buffer: MTLBuffer = unsafe { weights.mtl_buffer() }.to_owned();

        if kernel_data_type != DataType::F16 {
            return Err(MTLError::Generic(
                "Only F16 activation precision is supported for transposed quantized kernel".into(),
            ));
        }
        let g = config.group_size;
        let kernel_name = match g {
            64 => "qmm_transposed_f16_g64_b4",
            128 => "qmm_transposed_f16_g128_b4",
            other => {
                return Err(MTLError::Generic(format!(
                    "Unsupported group size {} for transposed F16 kernel",
                    other
                )));
            },
        }
        .to_string();

        let kernel = QuantizedMatmulKernel::new(
            mtl_context,
            kernel_data_type,
            &kernel_name,
        )
        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            weights_buffer,
            scales_buffer,
            biases_buffer,
            input_dim,
            output_dim,
            group_size: config.group_size,
            kernel_data_type,
            input_array_id,
            output_array_id,
        })
    }

    fn convert_biases(
        mtl_context: &MTLContext,
        target_dtype: DataType,
        group_size: usize,
        k: usize,
        n: usize,
        scales_src: &crate::backends::metal::array::MetalArray,
        zero_points_src: &crate::backends::metal::array::MetalArray,
    ) -> Result<MTLBuffer, MTLError> {
        use half::f16;
        if zero_points_src.data_type() != DataType::U8 {
            return Err(MTLError::Generic("zero_points must be U8".into()));
        }
        // New layout only: scales [N, K_g], zps [N, ceil(K_g/2)]
        let k_g = (k + group_size - 1) / group_size;
        let total = n * k_g;
        match (target_dtype, scales_src.data_type()) {
            (DataType::F16, DataType::F16) => {
                let s_len = n * k_g;
                let zp_len = n * ((k_g + 1) / 2);
                let s_src = unsafe {
                    std::slice::from_raw_parts(
                        scales_src.buffer().as_ptr() as *const f16,
                        s_len,
                    )
                };
                let zp = unsafe {
                    std::slice::from_raw_parts(
                        zero_points_src.buffer().as_ptr() as *const u8,
                        zp_len,
                    )
                };
                let mut out: Vec<f16> = Vec::with_capacity(total);
                let zp_row_stride_halves = (k_g + 1) / 2;
                for n_col in 0..n {
                    for kg in 0..k_g {
                        let s = s_src[n_col * k_g + kg].to_f32();
                        let zp_b = zp[n_col * zp_row_stride_halves + (kg / 2)];
                        let zp_n = if (kg & 1) == 0 {
                            zp_b & 0x0F
                        } else {
                            (zp_b >> 4) & 0x0F
                        } as u8;
                        out.push(f16::from_f32(-s * (zp_n as f32)));
                    }
                }
                Ok(mtl_context.device.new_buffer_with_data(
                    out.as_ptr() as *const _,
                    (out.len() * std::mem::size_of::<f16>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            },
            (DataType::F32, DataType::F32) => {
                let s_len = n * k_g;
                let zp_len = n * ((k_g + 1) / 2);
                let s_src = unsafe {
                    std::slice::from_raw_parts(
                        scales_src.buffer().as_ptr() as *const f32,
                        s_len,
                    )
                };
                let zp = unsafe {
                    std::slice::from_raw_parts(
                        zero_points_src.buffer().as_ptr() as *const u8,
                        zp_len,
                    )
                };
                let mut out: Vec<f32> = Vec::with_capacity(total);
                let zp_row_stride_halves = (k_g + 1) / 2;
                for n_col in 0..n {
                    for kg in 0..k_g {
                        let s = s_src[n_col * k_g + kg];
                        let zp_b = zp[n_col * zp_row_stride_halves + (kg / 2)];
                        let zp_n = if (kg & 1) == 0 {
                            zp_b & 0x0F
                        } else {
                            (zp_b >> 4) & 0x0F
                        } as u8;
                        out.push(-s * (zp_n as f32));
                    }
                }
                Ok(mtl_context.device.new_buffer_with_data(
                    out.as_ptr() as *const _,
                    (out.len() * std::mem::size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            },
            _ => Err(MTLError::Generic(
                "Unsupported dtype combo for biases".into(),
            )),
        }
    }
}

impl EncodableWithState for QuantizedLinearKernelBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let input_array = arrays[0].borrow();

        let batch_size = input_array.shape()[0];

        drop(input_array);

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let root_command_buffer = command_buffer.root_command_buffer();
        let encoder = root_command_buffer.new_compute_command_encoder();

        let m = batch_size;
        let n = self.output_dim;
        let k = self.input_dim;

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            b_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            biases_buffer: &self.biases_buffer,
            output_buffer: output_buffer,
            m: m as i32,
            n: n as i32,
            k: k as i32,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized matmul kernel");

        encoder.end_encoding();
    }
}
