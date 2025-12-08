use std::{mem::size_of, rc::Rc};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::super::{MTLContext, kernel::matmul::MatmulArguments};
use crate::{
    DataType,
    backends::metal::{
        ForwardPassState, MTLError, encodable_block::QuantizedEmbeddingError,
        forward_pass::ArrayId,
    },
    config::QuantizationMode,
    device::array::Array,
    parameters::ParameterTree,
};

pub struct QuantizedEmbeddingLookupKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct QuantizedEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a MTLBuffer, // [batch_size] as U64
    pub weights_buffer: &'a MTLBuffer, // [vocab_size, model_dim/packing_divisor] as U8/I8
    pub scales_buffer: &'a MTLBuffer,  // [vocab_size, num_groups]
    pub biases_buffer: &'a MTLBuffer,  // [vocab_size, num_groups]
    pub output_buffer: &'a MTLBuffer,  // [batch_size, model_dim]
    pub batch_size: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub group_size: u32,
    pub input_scale: f32,
}

impl QuantizedEmbeddingLookupKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        mode: QuantizationMode,
    ) -> Result<Self, QuantizedEmbeddingError> {
        let dtype_suffix = match data_type {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            other => {
                return Err(QuantizedEmbeddingError::UnsupportedDataType(
                    other,
                ));
            },
        };
        let mode_suffix = match mode {
            QuantizationMode::UInt4 => "uint4",
            QuantizationMode::Int8 => "int8",
            QuantizationMode::UInt8 => "uint8",
        };
        let kernel_name = format!(
            "quantized_embedding_lookup_{}_{}",
            dtype_suffix, mode_suffix
        );

        let (pipeline, _) = mtl_context
            .compute_pipeline_state_with_reflection(&kernel_name, None)
            .map_err(QuantizedEmbeddingError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: QuantizedEmbeddingLookupArguments,
    ) -> Result<(), QuantizedEmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.token_ids_buffer), 0);
        encoder.set_buffer(1, Some(args.weights_buffer), 0);
        encoder.set_buffer(2, Some(args.scales_buffer), 0);
        encoder.set_buffer(3, Some(args.biases_buffer), 0);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        // Set constants
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &args.batch_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &args.vocab_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &args.model_dim as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &args.group_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.input_scale as *const f32 as *const _,
        );

        // Dispatch one thread per output element
        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );

        Ok(())
    }
}

pub struct FullPrecisionReadoutKernelBlock {
    kernel: std::cell::RefCell<super::super::kernel::matmul::MatmulKernel>,
    weights_buffer: MTLBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl FullPrecisionReadoutKernelBlock {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for full precision readout: {:?}",
                data_type
            )));
        }

        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("output_weights").map_err(|e| {
                MTLError::Generic(format!("Failed to load weights: {:?}", e))
            })?,
        };

        if weights.shape() != [vocab_size, model_dim] {
            return Err(MTLError::Generic(format!(
                "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                weights.shape(),
                vocab_size,
                model_dim
            )));
        }

        if weights.data_type() != data_type {
            return Err(MTLError::Generic(format!(
                "Weights dtype mismatch: got {:?}, expected {:?}",
                weights.data_type(),
                data_type
            )));
        }

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };

        let kernel = super::super::kernel::matmul::MatmulKernel::new(
            mtl_context,
            data_type,
            false,
            true,
        )?;

        Ok(Self {
            kernel: std::cell::RefCell::new(kernel),
            weights_buffer,
            vocab_size,
            model_dim,
        })
    }
}

impl FullPrecisionReadoutKernelBlock {
    fn encode_impl(
        &self,
        state: &mut ForwardPassState,
        encoder: &metal::ComputeCommandEncoderRef,
    ) {
        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);
        let batch_size = state.active_suffix_length();
        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let args = MatmulArguments {
            a: input_buffer,
            b: &self.weights_buffer,
            d: output_buffer,
            batch: batch_size as i32,
            input_dim: self.model_dim as i32,
            output_dim: self.vocab_size as i32,
            lda: self.model_dim as i32,
            ldb: self.model_dim as i32, // B is [vocab_size, model_dim], stride is model_dim
            ldd: self.vocab_size as i32,
            batch_count: 1,
        };

        self.kernel
            .borrow_mut()
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode full precision readout kernel");
    }
}
