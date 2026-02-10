use crate::{
    DataType,
    backends::{
        common::kernel::MlpGateActMulKernel,
        metal::{
            FunctionConstantValuesSetValue, MTLBuffer, MTLComputeCommandEncoder,
            MTLContext, MTLError, MTLFunctionConstantValues, ProtocolObject,
            Retained, kernel::dsl::MlpGateActMulMetalKernel,
        },
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
        fcv.set_value(&true, MLP_FUSED_FC_INDEX as usize);
        fcv.set_value(&self.hidden_dim, MLP_HIDDEN_DIM_FC_INDEX as usize);
        fcv.set_value(
            &(self.activation as u32),
            MLP_ACTIVATION_FC_INDEX as usize,
        );
        fcv
    }
}

/// Create function constants for non-fused (standard) matmul
pub fn make_non_fused_function_constants() -> Retained<MTLFunctionConstantValues> {
    let fcv = MTLFunctionConstantValues::new();
    fcv.set_value(&false, MLP_FUSED_FC_INDEX as usize);
    fcv
}

pub struct MlpGateActMulEncodable {
    kernel: MlpGateActMulMetalKernel,
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
        let kernel = MlpGateActMulMetalKernel::new(context, data_type.into())?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        fused_up: &Retained<ProtocolObject<dyn MTLBuffer>>,
        hidden: &Retained<ProtocolObject<dyn MTLBuffer>>,
        m: i32,
    ) -> Result<(), MTLError> {
        let act_type = match self.activation {
            Activation::SiLU {
                ..
            } => 0,
            Activation::Gelu => 1,
            Activation::Identity => {
                panic!("Identity activation is not supported for kernel")
            },
        };
        self.kernel.encode(fused_up, hidden, self.hidden_dim as i32, m, act_type, encoder);
        Ok(())
    }
}
