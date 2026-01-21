use std::rc::Rc;

use crate::backends::metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, MTLCommandBuffer,
    MTLCommandEncoder,
};

use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        encodable_block::{EncodableBlock, EncodingParameters},
        forward_pass::{ArrayId, ForwardPassState},
        kernel::{
            KernelDataType, TensorAddBias,
            quant_matmul::{
                QuantizationType, QuantizedMatmulArguments,
                QuantizedMatmulKernel,
            },
        },
    },
    config::QuantizationConfig,
    device::array::Array,
    parameters::ParameterTree,
};

pub struct QuantizedLinear {
    kernel: QuantizedMatmulKernel,
    bias_add_kernel: Option<TensorAddBias>,
    biases_buffer: Option<Buffer>,
    weights_buffer: Buffer,
    scales_buffer: Buffer,
    zero_points_or_biases_buffer: Buffer,
    quantization_type: QuantizationType,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl QuantizedLinear {
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

        // Determine if weights are transposed by checking shape
        let weights_transposed = w_shape[0] == output_dim;

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
                    let scales_buffer: Buffer =
                        unsafe { objc2::rc::Retained::retain(scales.mtl_buffer() as *const _ as *mut _).unwrap() };
                    let biases_buf: Buffer =
                        unsafe { objc2::rc::Retained::retain(deq_biases.mtl_buffer() as *const _ as *mut _).unwrap() };
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
                    let scales_buffer: Buffer =
                        unsafe { objc2::rc::Retained::retain(scales.mtl_buffer() as *const _ as *mut _).unwrap() };
                    let zps_buf: Buffer =
                        unsafe { objc2::rc::Retained::retain(std::ptr::from_ref(&*zero_points.mtl_buffer()) as *mut _).unwrap() };
                    (QuantizationType::ZeroPoint, zps_buf, scales_buffer)
                },
            }
        };

        let weights_buffer: Buffer =
            unsafe { weights.mtl_buffer() }.to_owned().into();

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
                    let biases_buffer: Buffer =
                        unsafe { biases.mtl_buffer() }.to_owned().into();
                    (bias_add_kernel, Some(biases_buffer))
                },
                Err(_) => (None, None),
            };

        let g = config.group_size;
        let mode = config.weight_quantization_mode;

        let kernel = QuantizedMatmulKernel::new(
            mtl_context,
            kernel_data_type,
            g,
            input_dim,
            output_dim,
            mode,
            quantization_type,
            weights_transposed,
        )
        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            bias_add_kernel,
            biases_buffer,
            weights_buffer,
            scales_buffer: scales_buffer.into(),
            zero_points_or_biases_buffer: zero_points_or_biases_buffer.into(),
            quantization_type,
            input_dim,
            output_dim,
            input_array_id,
            output_array_id,
        })
    }
}

impl EncodableBlock for QuantizedLinear {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: CommandBufferRef<'_>,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            a_offset: 0,
            b_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            zero_points_or_biases_buffer: &self.zero_points_or_biases_buffer,
            output_buffer: output_buffer,
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            quantization_type: self.quantization_type,
        };

        self.kernel
            .encode(&encoder, args)
            .expect("Failed to encode quantized matmul kernel");

        encoder.end_encoding();

        if let (Some(bias_add), Some(bias_buf)) =
            (&self.bias_add_kernel, &self.biases_buffer)
        {
            let total_len = batch_size * self.output_dim;
            bias_add.encode_into_command_buffer(
                output_buffer,
                &bias_buf,
                output_buffer,
                self.output_dim,
                total_len,
                command_buffer,
                parameters.predicate.map(|v| &**v),
            );
        }

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: ComputeCommandEncoderRef<'_>,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            a_offset: 0,
            b_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            zero_points_or_biases_buffer: &self.zero_points_or_biases_buffer,
            output_buffer: output_buffer,
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            quantization_type: self.quantization_type,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized matmul kernel");

        if let (Some(bias_add), Some(bias_buf)) =
            (&self.bias_add_kernel, &self.biases_buffer)
        {
            let total_len = batch_size * self.output_dim;
            bias_add.encode_with_encoder(
                output_buffer,
                &bias_buf,
                output_buffer,
                self.output_dim,
                total_len,
                encoder,
                parameters.predicate.map(|v| &**v),
            );
        }
    }
}
