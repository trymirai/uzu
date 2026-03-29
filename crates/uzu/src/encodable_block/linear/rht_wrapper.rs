use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use half::bf16;
use thiserror::Error;

use super::{Linear, LinearBlockError, QuantizedLinear};
use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{
        Backend, Encoder,
        kernel::{
            HadamardTransformMulKernel, Kernels, QuantizedMatmulQmmTransposedOutputHadamardKernel,
            QuantizedMatmulQmvFastOutputHadamardKernel, quant_matmul::QuantizedMatmulType,
        },
    },
    config::{LinearConfig, QuantizationConfig},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[source] Box<LinearBlockError<B>>),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Input dimension {input_dimension} is not divisible by block size {block_size}")]
    InputDimensionNotDivisibleByBlockSize {
        input_dimension: usize,
        block_size: usize,
    },
    #[error("Output dimension {output_dimension} is not divisible by block size {block_size}")]
    OutputDimensionNotDivisibleByBlockSize {
        output_dimension: usize,
        block_size: usize,
    },
    #[error("Input factors shape mismatch: expected [{expected_dimension}], got {actual_shape:?}")]
    InputFactorsShapeMismatch {
        expected_dimension: usize,
        actual_shape: Box<[usize]>,
    },
    #[error("Output factors shape mismatch: expected [{expected_dimension}], got {actual_shape:?}")]
    OutputFactorsShapeMismatch {
        expected_dimension: usize,
        actual_shape: Box<[usize]>,
    },
}

struct OutputFusedDecodeKernel<B: Backend> {
    kernel: <B::Kernels as Kernels>::QuantizedMatmulQmvFastOutputHadamardKernel,
    weights_buffer: Rc<RefCell<B::Buffer>>,
    scales_buffer: Rc<RefCell<B::Buffer>>,
    zero_points_or_biases_buffer: Rc<RefCell<B::Buffer>>,
    is_mlx_quant: bool,
}

struct PrefillFusedKernel<B: Backend> {
    kernel: <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedOutputHadamardKernel,
    weights_buffer: Rc<RefCell<B::Buffer>>,
    scales_buffer: Rc<RefCell<B::Buffer>>,
    zero_points_or_biases_buffer: Rc<RefCell<B::Buffer>>,
    is_mlx_quant: bool,
}

pub struct RHTLinearWrapper<B: Backend> {
    inner_linear: Box<dyn Linear<B>>,
    input_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformMulKernel, Rc<RefCell<B::Buffer>>)>,
    output_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformMulKernel,
    output_factors_buffer: Rc<RefCell<B::Buffer>>,
    input_dimension: usize,
    output_dimension: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    output_fused: Option<OutputFusedDecodeKernel<B>>,
    prefill_fused: Option<PrefillFusedKernel<B>>,
}

fn quant_bits(config: &QuantizationConfig) -> i32 {
    match config.weight_quantization_mode {
        crate::backends::common::gpu_types::QuantizationMode::UINT4 => 4,
        crate::backends::common::gpu_types::QuantizationMode::INT8
        | crate::backends::common::gpu_types::QuantizationMode::UINT8 => 8,
    }
}

fn try_create_output_fused_from_quantized<B: Backend>(
    context: &B::Context,
    inner_config: &LinearConfig,
    quantized_linear: &QuantizedLinear<B>,
    input_dimension: usize,
    output_dimension: usize,
) -> Option<OutputFusedDecodeKernel<B>> {
    let quant_config = match inner_config {
        LinearConfig::Quantized(q) | LinearConfig::MLXQuantized(q) => q,
        _ => return None,
    };

    if output_dimension % 32 != 0 || input_dimension % 512 != 0 {
        return None;
    }

    let kernel_data_type: DataType = quant_config.activation_precision.into();
    let bits = quant_bits(quant_config);
    let group_size = quant_config.group_size as i32;
    let is_mlx_quant = matches!(quantized_linear.quantization_type(), QuantizedMatmulType::Mlx);

    let kernel = <B::Kernels as Kernels>::QuantizedMatmulQmvFastOutputHadamardKernel::new(
        context,
        kernel_data_type,
        group_size,
        bits,
        !is_mlx_quant,
        is_mlx_quant,
    )
    .ok()?;

    Some(OutputFusedDecodeKernel {
        kernel,
        weights_buffer: Rc::clone(quantized_linear.weights_buffer()),
        scales_buffer: Rc::clone(quantized_linear.scales_buffer()),
        zero_points_or_biases_buffer: Rc::clone(quantized_linear.zero_points_or_biases_buffer()),
        is_mlx_quant,
    })
}

fn try_create_prefill_fused_from_quantized<B: Backend>(
    context: &B::Context,
    inner_config: &LinearConfig,
    quantized_linear: &QuantizedLinear<B>,
    output_dimension: usize,
) -> Option<PrefillFusedKernel<B>> {
    let quant_config = match inner_config {
        LinearConfig::Quantized(q) | LinearConfig::MLXQuantized(q) => q,
        _ => return None,
    };

    if output_dimension % 32 != 0 {
        return None;
    }

    let kernel_data_type: DataType = quant_config.activation_precision.into();
    let bits = quant_bits(quant_config);
    let group_size = quant_config.group_size as i32;
    let is_mlx_quant = matches!(quantized_linear.quantization_type(), QuantizedMatmulType::Mlx);

    let kernel = <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedOutputHadamardKernel::new(
        context,
        kernel_data_type,
        group_size,
        bits,
        !is_mlx_quant,
        is_mlx_quant,
    )
    .ok()?;

    Some(PrefillFusedKernel {
        kernel,
        weights_buffer: Rc::clone(quantized_linear.weights_buffer()),
        scales_buffer: Rc::clone(quantized_linear.scales_buffer()),
        zero_points_or_biases_buffer: Rc::clone(quantized_linear.zero_points_or_biases_buffer()),
        is_mlx_quant,
    })
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        block_size: usize,
        inner_config: &LinearConfig,
        input_dimension: usize,
        output_dimension: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        if input_dimension % block_size != 0 {
            return Err(RHTLinearWrapperError::InputDimensionNotDivisibleByBlockSize {
                input_dimension,
                block_size,
            });
        }
        if output_dimension % block_size != 0 {
            return Err(RHTLinearWrapperError::OutputDimensionNotDivisibleByBlockSize {
                output_dimension,
                block_size,
            });
        }

        let kernel_data_type: DataType = inner_config.activation_precision().into();

        let input_factors_raw =
            parameter_tree.leaf_array("input_factors").map_err(RHTLinearWrapperError::ParameterError)?;

        if input_factors_raw.shape() != [input_dimension] {
            return Err(RHTLinearWrapperError::InputFactorsShapeMismatch {
                expected_dimension: input_dimension,
                actual_shape: input_factors_raw.shape().into(),
            });
        }

        let output_factors_raw =
            parameter_tree.leaf_array("output_factors").map_err(RHTLinearWrapperError::ParameterError)?;

        if output_factors_raw.shape() != [output_dimension] {
            return Err(RHTLinearWrapperError::OutputFactorsShapeMismatch {
                expected_dimension: output_dimension,
                actual_shape: output_factors_raw.shape().into(),
            });
        }

        let input_factors = convert_int32_factors_to_kernel_type(context, &input_factors_raw, kernel_data_type);
        let output_factors = convert_int32_factors_to_kernel_type(context, &output_factors_raw, kernel_data_type);

        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformMulKernel::new(context, kernel_data_type)
            .map_err(RHTLinearWrapperError::BackendError)?;

        let output_hadamard_kernel =
            <B::Kernels as Kernels>::HadamardTransformMulKernel::new(context, kernel_data_type)
                .map_err(RHTLinearWrapperError::BackendError)?;

        let inner_linear_tree =
            parameter_tree.subtree("inner_linear").map_err(RHTLinearWrapperError::ParameterError)?;

        let input_factors_buffer_rc = input_factors.buffer();

        let (inner_linear, output_fused, prefill_fused) = match inner_config {
            LinearConfig::Quantized(q) | LinearConfig::MLXQuantized(q) => {
                let ql = QuantizedLinear::new(
                    context,
                    q,
                    input_dimension,
                    output_dimension,
                    &inner_linear_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(|e| {
                    RHTLinearWrapperError::InnerLinearError(Box::new(super::LinearBlockError::QuantizedLinearError(e)))
                })?;

                let out_fused = try_create_output_fused_from_quantized(
                    context,
                    inner_config,
                    &ql,
                    input_dimension,
                    output_dimension,
                );

                let pf_fused = try_create_prefill_fused_from_quantized(context, inner_config, &ql, output_dimension);

                (Box::new(ql) as Box<dyn Linear<B>>, out_fused, pf_fused)
            },
            _ => {
                let inner = <dyn Linear<B>>::new(
                    inner_config,
                    false,
                    input_dimension,
                    [output_dimension],
                    context,
                    &inner_linear_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(|error| RHTLinearWrapperError::InnerLinearError(Box::new(error)))?;
                (inner, None, None)
            },
        };

        Ok(Self {
            inner_linear,
            input_hadamard: Some((input_hadamard_kernel, input_factors_buffer_rc)),
            output_hadamard_kernel,
            output_factors_buffer: output_factors.buffer(),
            input_dimension,
            output_dimension,
            input_array_id,
            output_array_id,
            output_fused,
            prefill_fused,
        })
    }

    pub fn take_input_hadamard_factors(&mut self) -> Option<Rc<RefCell<B::Buffer>>> {
        self.input_hadamard.take().map(|(_, factors)| factors)
    }
}

pub fn convert_int32_factors_to_kernel_type<B: Backend>(
    context: &B::Context,
    source_array: &Array<B>,
    target_data_type: DataType,
) -> Array<B> {
    let int32_values = source_array.as_slice::<i32>();

    match target_data_type {
        DataType::BF16 => {
            let converted: Vec<bf16> = int32_values.iter().map(|&value| bf16::from_f32(value as f32)).collect();
            context.create_array_from(source_array.shape(), &converted, "rht_factors")
        },
        DataType::F16 => {
            let converted: Vec<half::f16> =
                int32_values.iter().map(|&value| half::f16::from_f32(value as f32)).collect();
            context.create_array_from(source_array.shape(), &converted, "rht_factors")
        },
        DataType::F32 => {
            let converted: Vec<f32> = int32_values.iter().map(|&value| value as f32).collect();
            context.create_array_from(source_array.shape(), &converted, "rht_factors")
        },
        other => panic!("Unsupported kernel data type for RHT factors: {other:?}"),
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let batch_size = state.active_suffix_length();

        // Input Hadamard (standalone dispatch, or skipped if fused into preceding norm)
        if let Some((ref kernel, ref factors_buffer)) = self.input_hadamard {
            let input_total_blocks = (batch_size * self.input_dimension / 32) as u32;
            let input_array = state.array(self.input_array_id);
            kernel.encode(
                input_array.buffer().borrow_mut().deref_mut(),
                factors_buffer.borrow().deref(),
                input_total_blocks,
                self.input_dimension as u32,
                encoder,
            );
        }

        // QMV + Output Hadamard: fused for decode, separate for prefill
        if batch_size < 32 {
            if let Some(ref out_fused) = self.output_fused {
                let input_array = state.array(self.input_array_id);
                let output_array = state.array(self.output_array_id);

                let w_borrow = out_fused.weights_buffer.borrow();
                let s_borrow = out_fused.scales_buffer.borrow();
                let zp_borrow = out_fused.zero_points_or_biases_buffer.borrow();
                let in_buf_rc = input_array.buffer();
                let in_borrow = in_buf_rc.borrow();
                let out_buf_rc = output_array.buffer();
                let mut out_borrow = out_buf_rc.borrow_mut();
                let of_borrow = self.output_factors_buffer.borrow();

                let zp_opt: Option<&B::Buffer> = if !out_fused.is_mlx_quant {
                    Some(zp_borrow.deref())
                } else {
                    None
                };
                let bias_opt: Option<&B::Buffer> = if out_fused.is_mlx_quant {
                    Some(zp_borrow.deref())
                } else {
                    None
                };

                out_fused.kernel.encode(
                    w_borrow.deref(),
                    s_borrow.deref(),
                    zp_opt,
                    bias_opt,
                    in_borrow.deref(),
                    out_borrow.deref_mut(),
                    of_borrow.deref(),
                    self.input_dimension as i32,
                    self.output_dimension as i32,
                    batch_size as i32,
                    encoder,
                );

                return Ok(());
            }
        }

        // Prefill fused path: QMM + Output Hadamard in one kernel
        if let Some(ref pf_fused) = self.prefill_fused {
            let input_array = state.array(self.input_array_id);
            let output_array = state.array(self.output_array_id);

            let w_borrow = pf_fused.weights_buffer.borrow();
            let s_borrow = pf_fused.scales_buffer.borrow();
            let zp_borrow = pf_fused.zero_points_or_biases_buffer.borrow();
            let in_buf_rc = input_array.buffer();
            let in_borrow = in_buf_rc.borrow();
            let out_buf_rc = output_array.buffer();
            let mut out_borrow = out_buf_rc.borrow_mut();
            let of_borrow = self.output_factors_buffer.borrow();

            let zp_opt: Option<&B::Buffer> = if !pf_fused.is_mlx_quant {
                Some(zp_borrow.deref())
            } else {
                None
            };
            let bias_opt: Option<&B::Buffer> = if pf_fused.is_mlx_quant {
                Some(zp_borrow.deref())
            } else {
                None
            };

            pf_fused.kernel.encode(
                w_borrow.deref(),
                s_borrow.deref(),
                zp_opt,
                bias_opt,
                in_borrow.deref(),
                out_borrow.deref_mut(),
                of_borrow.deref(),
                self.input_dimension as i32,
                self.output_dimension as i32,
                batch_size as i32,
                encoder,
            );

            return Ok(());
        }

        // Fallback: separate inner_linear + output Hadamard
        self.inner_linear.encode(state, encoder)?;

        {
            let output_total_blocks = (batch_size * self.output_dimension / 32) as u32;
            let output_array = state.array(self.output_array_id);
            self.output_hadamard_kernel.encode(
                output_array.buffer().borrow_mut().deref_mut(),
                self.output_factors_buffer.borrow().deref(),
                output_total_blocks,
                self.output_dimension as u32,
                encoder,
            );
        }

        Ok(())
    }
}
