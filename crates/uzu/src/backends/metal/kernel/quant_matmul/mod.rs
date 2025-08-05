use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{super::MTLContext, KernelDataType};
use crate::{
    DataType,
    backends::metal::{
        MTLError,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Invalid dimensions: M={m}, N={n}, K={k}")]
    InvalidDimensions {
        m: usize,
        n: usize,
        k: usize,
    },
}

pub struct QuantizedMatmulKernel {
    pipeline: MTLComputePipelineState,
    name: String,
}

#[derive(Debug)]
pub struct QuantizedMatmulArguments<'a> {
    pub a_buffer: &'a MTLBuffer, // Input A (float)
    pub b_buffer: &'a MTLBuffer, // Input B (quantized)
    pub scales_buffer: &'a MTLBuffer,
    pub biases_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
}

impl QuantizedMatmulKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        kernel_name: &str,
    ) -> Result<Self, QuantizedMatmulError> {
        // Validate inputs
        if !matches!(data_type, DataType::F16 | DataType::F32) {
            return Err(QuantizedMatmulError::UnsupportedDataType(data_type));
        }

        let pipeline = mtl_context
            .compute_pipeline_state(kernel_name, None)
            .map_err(QuantizedMatmulError::MetalError)?;

        Ok(Self {
            pipeline,
            name: kernel_name.to_string(),
        })
    }

    fn kernel_name_for_config(data_type: DataType) -> String {
        let type_suffix = match data_type {
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            _ => unreachable!(),
        };

        format!("qmm_{}_g64_b4", type_suffix)
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: QuantizedMatmulArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.b_buffer), 0);
        encoder.set_buffer(1, Some(args.scales_buffer), 0);
        encoder.set_buffer(2, Some(args.biases_buffer), 0);
        encoder.set_buffer(3, Some(args.a_buffer), 0);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        // Set constants
        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.k as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.n as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.m as *const i32 as *const _,
        );

        if self.name.starts_with("qmv") {
            let bk = 32;
            let bn = 8;
            let n_tgp_y = (args.n + bn - 1) / bn; // ceiling division
            let threadgroups = MTLSize::new(args.m as u64, n_tgp_y as u64, 1);
            let threads_per_threadgroup = MTLSize::new(bk as u64, 2, 1);

            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        } else if self.name.starts_with("qvm") {
            let bk = 32;
            let bn = 64;
            let n_tgp_y = (args.n + bn - 1) / bn; // ceiling division for columns
            let threadgroups = MTLSize::new(args.m as u64, n_tgp_y as u64, 1);
            let threads_per_threadgroup = MTLSize::new(bk as u64, 2, 1);

            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        } else {
            let bm = 32;
            let bn = 32;
            let wm = 2;
            let wn = 2;

            let threads_per_threadgroup = MTLSize::new(32, wn, wm);
            let threadgroups = MTLSize::new(
                (args.n as u64 + bn - 1) / bn,
                (args.m as u64 + bm - 1) / bm,
                1,
            );

            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }

        Ok(())
    }
}

pub struct QuantizedMatmulKernelEncodable {
    kernel: QuantizedMatmulKernel,
    kernel_data_type: KernelDataType,
    group_size: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl QuantizedMatmulKernelEncodable {
    pub fn new(
        mtl_context: &MTLContext,
        kernel_data_type: KernelDataType,
        group_size: usize,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, QuantizedMatmulError> {
        let data_type = match kernel_data_type {
            KernelDataType::Float16 => DataType::F16,
            KernelDataType::Float32 => DataType::F32,
            _ => {
                return Err(QuantizedMatmulError::UnsupportedDataType(
                    kernel_data_type.into(),
                ));
            },
        };

        let kernel = QuantizedMatmulKernel::new(
            mtl_context,
            data_type,
            "qmm_f32_g64_b4",
        )?;

        Ok(Self {
            kernel,
            kernel_data_type,
            group_size,
            input_array_id,
            output_array_id,
        })
    }
}

impl EncodableWithState for QuantizedMatmulKernelEncodable {
    fn encode(
        &self,
        _state: &mut ForwardPassState,
        _command_buffer: &MPSCommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        // For now, just a placeholder - full implementation will come with weight conversion
        eprintln!(
            "QuantizedMatmulKernelEncodable::encode - not yet implemented"
        );

        // TODO:
        // 1. Get input array from state
        // 2. Convert weights from uzu format
        // 3. Convert zero_points to biases
        // 4. Call kernel.encode with converted buffers
        // 5. Handle batch dimensions properly
    }
}
