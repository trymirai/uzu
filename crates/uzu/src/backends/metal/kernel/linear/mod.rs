use std::rc::Rc;

use metal::Buffer as MTLBuffer;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    KernelDataType, TensorAddBias,
    quant_matmul::{
        QuantizationType, QuantizedMatmulArguments, QuantizedMatmulKernel,
        quantized_kernel_names,
    },
};
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
    kernel_mm: QuantizedMatmulKernel,
    kernel_mv: QuantizedMatmulKernel,
    kernel_mv_fast: Option<QuantizedMatmulKernel>,
    bias_add_kernel: Option<TensorAddBias>,
    biases_buffer: Option<MTLBuffer>,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    zero_points_or_biases_buffer: MTLBuffer,
    quantization_type: QuantizationType,
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

        if !matches!(
            kernel_data_type,
            DataType::F16 | DataType::BF16 | DataType::F32
        ) {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for quantized kernel: {:?}",
                kernel_data_type
            )));
        }

        let mut weights = parameter_tree.leaf("weights").map_err(|e| {
            MTLError::Generic(format!("Failed to load weights: {:?}", e))
        })?;

        let packing_divisor = config.weight_quantization_mode.packing_divisor();
        let storage_type = config.weight_quantization_mode.storage_type();

        if weights.data_type() != storage_type {
            return Err(MTLError::Generic(format!(
                "Expected weights of type {:?}, got {:?}",
                storage_type,
                weights.data_type()
            )));
        }

        let mut scales = parameter_tree.leaf("scales").map_err(|e| {
            MTLError::Generic(format!("Failed to load scales: {:?}", e))
        })?;
        let k_g = (input_dim + config.group_size - 1) / config.group_size;

        let w_shape = weights.shape();
        let s_shape = scales.shape();

        // Determine quantization style: prefer MLX (deq_biases), else AWQ (zero_points)
        let (quantization_type, zero_points_or_biases_buffer, scales_buffer) = {
            if scales.data_type() != kernel_data_type {
                return Err(MTLError::Generic(format!(
                    "Scales dtype mismatch: got {:?}, expected {:?}",
                    scales.data_type(),
                    kernel_data_type
                )));
            }
            match parameter_tree.leaf("deq_biases") {
                Ok(mut deq_biases) => {
                    let db_shape = deq_biases.shape();
                    if !(w_shape == [output_dim, input_dim / packing_divisor]
                        && s_shape == [output_dim, k_g]
                        && db_shape == [output_dim, k_g])
                    {
                        return Err(MTLError::Generic(format!(
                            "Unexpected MLX shapes. weights={:?}, scales={:?}, deq_biases={:?}; expected [N,K/{}],[N,K_g],[N,K_g]",
                            w_shape, s_shape, db_shape, packing_divisor
                        )));
                    }
                    if deq_biases.data_type() != kernel_data_type {
                        return Err(MTLError::Generic(format!(
                            "deq_biases dtype mismatch: got {:?}, expected {:?}",
                            deq_biases.data_type(),
                            kernel_data_type
                        )));
                    }
                    let scales_buffer =
                        unsafe { scales.mtl_buffer() }.to_owned();
                    let biases_buf =
                        unsafe { deq_biases.mtl_buffer() }.to_owned();
                    (QuantizationType::Mlx, biases_buf, scales_buffer)
                },
                Err(_) => {
                    let mut zero_points =
                        parameter_tree.leaf("zero_points").map_err(|e| {
                            MTLError::Generic(format!(
                                "Failed to load zero_points: {:?}",
                                e
                            ))
                        })?;
                    let zp_shape = zero_points.shape();
                    let expected_zp_entries =
                        (k_g + packing_divisor - 1) / packing_divisor;
                    if !(w_shape == [output_dim, input_dim / packing_divisor]
                        && s_shape == [output_dim, k_g]
                        && zp_shape == [output_dim, expected_zp_entries])
                    {
                        return Err(MTLError::Generic(format!(
                            "Unexpected AWQ shapes. weights={:?}, scales={:?}, zero_points={:?}; expected [N,K/{}],[N,K_g],[N,(K_g+{})/{}]",
                            w_shape,
                            s_shape,
                            zp_shape,
                            packing_divisor,
                            packing_divisor - 1,
                            packing_divisor
                        )));
                    }
                    if zero_points.data_type() != storage_type {
                        return Err(MTLError::Generic(format!(
                            "Zero-points dtype mismatch: got {:?}, expected {:?}",
                            zero_points.data_type(),
                            storage_type
                        )));
                    }
                    let scales_buffer =
                        unsafe { scales.mtl_buffer() }.to_owned();
                    let zps_buf =
                        unsafe { zero_points.mtl_buffer() }.to_owned();
                    (QuantizationType::ZeroPoint, zps_buf, scales_buffer)
                },
            }
        };

        let weights_buffer: MTLBuffer =
            unsafe { weights.mtl_buffer() }.to_owned();

        // Optional trainable bias support
        let (bias_add_kernel, biases_buffer) =
            match parameter_tree.leaf("biases") {
                Ok(mut biases) => {
                    if biases.shape() != [output_dim] {
                        return Err(MTLError::Generic(format!(
                            "Bias shape mismatch: got {:?}, expected [{:?}]",
                            biases.shape(),
                            output_dim
                        )));
                    }
                    if biases.data_type() != kernel_data_type {
                        return Err(MTLError::Generic(format!(
                            "Bias dtype mismatch: got {:?}, expected {:?}",
                            biases.data_type(),
                            kernel_data_type
                        )));
                    }
                    let bias_add_kernel = Some(TensorAddBias::new(
                        mtl_context,
                        KernelDataType::from(kernel_data_type),
                    )?);
                    let biases_buffer: MTLBuffer =
                        unsafe { biases.mtl_buffer() }.to_owned();
                    (bias_add_kernel, Some(biases_buffer))
                },
                Err(_) => (None, None),
            };

        let g = config.group_size;
        let Some((kernel_name_mm, kernel_name_mv)) = quantized_kernel_names(
            kernel_data_type,
            g,
            output_dim,
            input_dim,
            config.weight_quantization_mode,
        ) else {
            return Err(MTLError::Generic(format!(
                "Unsupported group size {} or bits {:?} for transposed {:?} kernel",
                g, config.weight_quantization_mode, kernel_data_type
            )));
        };

        let kernel_mm = QuantizedMatmulKernel::new(
            mtl_context,
            kernel_data_type,
            &kernel_name_mm,
            quantization_type,
        )
        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?;
        let kernel_mv = QuantizedMatmulKernel::new(
            mtl_context,
            kernel_data_type,
            &kernel_name_mv,
            quantization_type,
        )
        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?;

        let can_use_fast = output_dim % 8 == 0 && input_dim % 512 == 0;
        let kernel_mv_fast = if can_use_fast {
            let fast_name = format!("{}_fast", kernel_name_mv);
            let result = QuantizedMatmulKernel::new(
                mtl_context,
                kernel_data_type,
                &fast_name,
                quantization_type,
            );

            result.ok()
        } else {
            None
        };

        Ok(Self {
            kernel_mm,
            kernel_mv,
            kernel_mv_fast,
            bias_add_kernel,
            biases_buffer,
            weights_buffer,
            scales_buffer,
            zero_points_or_biases_buffer,
            quantization_type,
            input_dim,
            output_dim,
            group_size: config.group_size,
            kernel_data_type,
            input_array_id,
            output_array_id,
        })
    }
}

impl EncodableWithState for QuantizedLinearKernelBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
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
            zero_points_or_biases_buffer: &self.zero_points_or_biases_buffer,
            output_buffer: output_buffer,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            quantization_type: self.quantization_type,
        };

        let use_gemv = batch_size == 1;
        let kernel = if use_gemv {
            if let Some(ref fast_kernel) = self.kernel_mv_fast {
                fast_kernel
            } else {
                &self.kernel_mv
            }
        } else {
            &self.kernel_mm
        };

        kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized matmul kernel");

        encoder.end_encoding();

        if let (Some(bias_add), Some(bias_buf)) =
            (&self.bias_add_kernel, &self.biases_buffer)
        {
            let total_len = batch_size * self.output_dim;
            let retained_cb = root_command_buffer.to_owned();
            bias_add.encode_into_command_buffer(
                &output_buffer,
                bias_buf,
                &output_buffer,
                self.output_dim,
                total_len,
                &retained_cb,
            );
        }

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
