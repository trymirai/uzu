use uzu_engine_macros::uzu_config_abstract;

use crate::backends::common::gpu_types::activation_type::ActivationType;

pub mod gelu;
pub mod identity;
pub mod sigmoid;
pub mod silu;
pub mod soft_capped_silu;

#[uzu_config_abstract(silu::SiLU, gelu::GELU, identity::Identity, sigmoid::Sigmoid, soft_capped_silu::SoftCappedSiLU)]
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
            AnyActivation::Sigmoid(_) => ActivationType::SIGMOID,
            AnyActivation::SoftCappedSiLU(_) => {
                panic!("SoftCappedSiLU has no element-wise kernel; only the mixture-of-experts kernels apply it")
            },
        }
    }

    /// Whether the element-wise activation kernels implement this activation.
    pub fn has_act_type(&self) -> bool {
        !matches!(self, AnyActivation::SoftCappedSiLU(_))
    }

    pub fn alpha(&self) -> f32 {
        match self {
            AnyActivation::SiLU(config) => config.alpha,
            AnyActivation::GELU(_)
            | AnyActivation::Identity(_)
            | AnyActivation::Sigmoid(_)
            | AnyActivation::SoftCappedSiLU(_) => 1.0,
        }
    }

    /// Non-default activation slope encoded by kernels that specialize it.
    pub fn custom_alpha(&self) -> Option<f32> {
        match self {
            AnyActivation::SiLU(config) if config.alpha != 1.0 => Some(config.alpha),
            _ => None,
        }
    }
}
