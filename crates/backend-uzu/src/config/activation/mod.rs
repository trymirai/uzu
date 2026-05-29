use proc_macros::uzu_config_abstract;

use crate::backends::common::gpu_types::ActivationType;

pub mod gelu;
pub mod identity;
pub mod silu;

#[uzu_config_abstract(silu::SiLU, gelu::GELU, identity::Identity)]
pub struct Activation;

impl AnyActivation {
    pub fn act_type(&self) -> ActivationType {
        match self {
            AnyActivation::SiLU(_) => ActivationType::SILU,
            AnyActivation::GELU(gelu::GELU {
                approximate: true,
                ..
            }) => ActivationType::GELUApprox,
            AnyActivation::GELU(gelu::GELU {
                approximate: false,
                ..
            }) => ActivationType::GELUExact,
            AnyActivation::Identity(_) => ActivationType::IDENTITY,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            AnyActivation::SiLU(config) => config.alpha,
            AnyActivation::GELU(_) | AnyActivation::Identity(_) => 1.0,
        }
    }
}
