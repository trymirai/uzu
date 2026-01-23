use std::{ffi::c_void, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::{
    DataType,
    backends::metal::{
        FunctionConstantValuesLegacy, MTLBuffer, MTLFunctionConstantValues,
        MTLComputePipelineState, MTLContext, MTLDataType, MTLError, MTLSize,
        ProtocolObject, Retained,
    },
    config::Activation,
};

// MLP Fused Epilogue Function Constant Indices
// These must match the values in kernel/common/mlp_epilogue.h
pub const MLP_FUSED_FC_INDEX: u64 = 50;
pub const MLP_HIDDEN_DIM_FC_INDEX: u64 = 51;
pub const MLP_ACTIVATION_FC_INDEX: u64 = 52;

/// MLP activation type for fused kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MlpActivationType {
    SiLU = 0,
    Gelu = 1,
}

impl From<&Activation> for MlpActivationType {
    fn from(act: &Activation) -> Self {
        match act {
            Activation::SiLU {
                ..
            } => MlpActivationType::SiLU,
            Activation::Gelu => MlpActivationType::Gelu,
            Activation::Identity => {
                panic!("Identity activation not supported for MLP fusion")
            },
        }
    }
}

/// Configuration for MLP fused matmul epilogue
#[derive(Debug, Clone, Copy)]
pub struct MlpFusedConfig {
    pub hidden_dim: u32,
    pub activation: MlpActivationType,
}

impl MlpFusedConfig {
    pub fn new(
        hidden_dim: usize,
        activation: &Activation,
    ) -> Self {
        Self {
            hidden_dim: hidden_dim as u32,
            activation: MlpActivationType::from(activation),
        }
    }

    /// Create function constants for MLP fused matmul
    pub fn make_function_constants(&self) -> Retained<MTLFunctionConstantValues> {
        let fcv = MTLFunctionConstantValues::new();
        let fused = true;
        fcv.set_constant_value_at_index(
            &fused as *const bool as *const std::ffi::c_void,
            MTLDataType::Bool,
            MLP_FUSED_FC_INDEX,
        );
        fcv.set_constant_value_at_index(
            &self.hidden_dim as *const u32 as *const std::ffi::c_void,
            MTLDataType::UInt,
            MLP_HIDDEN_DIM_FC_INDEX,
        );
        let act_val = self.activation as u32;
        fcv.set_constant_value_at_index(
            &act_val as *const u32 as *const std::ffi::c_void,
            MTLDataType::UInt,
            MLP_ACTIVATION_FC_INDEX,
        );
        fcv
    }
}

/// Create function constants for non-fused (standard) matmul
pub fn make_non_fused_function_constants() -> Retained<MTLFunctionConstantValues> {
    let fcv = MTLFunctionConstantValues::new();
    let fused = false;
    fcv.set_constant_value_at_index(
        &fused as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        MLP_FUSED_FC_INDEX,
    );
    fcv
}

pub struct MlpGateActMulKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct MlpGateActMulEncodable {
    kernel: MlpGateActMulKernel,
    activation: Activation,
    hidden_dim: usize,
}

impl MlpGateActMulEncodable {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        activation: Activation,
        hidden_dim: usize,
    ) -> Result<Self, MTLError> {
        let kernel = MlpGateActMulKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        fused_up: &ProtocolObject<dyn MTLBuffer>,
        hidden: &ProtocolObject<dyn MTLBuffer>,
        m: i32,
    ) -> Result<(), MTLError> {
        self.kernel.encode(
            encoder,
            &self.activation,
            fused_up,
            hidden,
            m,
            self.hidden_dim as i32,
        )
    }
}

impl MlpGateActMulKernel {
    fn kernel_name_for_type(
        data_type: DataType
    ) -> Result<&'static str, MTLError> {
        match data_type {
            DataType::F16 => Ok("mlp_activation_mul_f16"),
            DataType::F32 => Ok("mlp_activation_mul_f32"),
            DataType::BF16 => Ok("mlp_activation_mul_bf16"),
            other => Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP activation: {:?}",
                other
            ))),
        }
    }

    pub fn new(
        context: &MTLContext,
        data_type: DataType,
    ) -> Result<Self, MTLError> {
        let fn_name = Self::kernel_name_for_type(data_type)?;
        let pipeline = context.compute_pipeline_state(fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    fn act_code(act: &Activation) -> u16 {
        match act {
            Activation::SiLU {
                ..
            } => 0,
            Activation::Gelu => 1,
            Activation::Identity => {
                panic!("Identity activation is not supported for MLP kernels")
            },
        }
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        activation: &Activation,
        fused_up_buffer: &ProtocolObject<dyn MTLBuffer>,
        hidden_buffer: &ProtocolObject<dyn MTLBuffer>,
        m: i32,
        h: i32,
    ) -> Result<(), MTLError> {
        let act_code = Self::act_code(activation);
        let threads_per_tg = MTLSize::new(64, 1, 1);
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(Some(fused_up_buffer), 0, 0);
        encoder.set_buffer(Some(hidden_buffer), 0, 1);
        unsafe {
            encoder.set_bytes(
                NonNull::new(&h as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                2,
            );
            encoder.set_bytes(
                NonNull::new(&m as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                3,
            );
            encoder.set_bytes(
                NonNull::new(&act_code as *const u16 as *mut c_void).unwrap(),
                std::mem::size_of::<u16>(),
                4,
            );
        }

        let grid = MTLSize::new(h as usize, m as usize, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }
}
