//! Mixture-of-experts routing and expert execution.

mod experts;
mod router;

use experts::MoeExperts;
use router::MoeRouter;
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{
            ActivationType,
            moe::{ROUTER_MAX_EXPERTS, ROUTER_MAX_MODEL_DIM, ROUTER_MAX_SELECTED_EXPERTS},
        },
    },
    config::{
        mlp::mixture_of_experts::MixtureOfExpertsConfig,
        weight_matrix::{AnyWeightMatrixSpec, Layout, full_precision_spec::FullPrecisionSpec},
    },
    data_type::DataType,
    encodable_block::{
        mlp::Mlp,
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

fn valid_model_dim(model_dim: u32) -> bool {
    model_dim > 0 && model_dim <= ROUTER_MAX_MODEL_DIM && model_dim.is_multiple_of(4)
}

fn valid_routed_expert_count(expert_count: u32) -> bool {
    (1..=ROUTER_MAX_EXPERTS).contains(&expert_count)
}

fn valid_active_expert_count(
    active_expert_count: u32,
    routed_expert_count: u32,
) -> bool {
    (1..=ROUTER_MAX_SELECTED_EXPERTS.min(routed_expert_count)).contains(&active_expert_count)
}

pub struct MoeBlock<B: Backend> {
    router: MoeRouter<B>,
    experts: MoeExperts<B>,
}

#[derive(Debug, Error)]
pub enum MoeBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Expert weight loading error: {0}")]
    WeightMatrixError(#[from] WeightMatrixError<B>),
    #[error("MoE model_dim must be nonzero, divisible by 4, and within the fused router capacity")]
    InvalidModelDim,
    #[error("MoE num_routed_experts must be nonzero and within the fused router capacity")]
    InvalidRoutedExpertCount,
    #[error("MoE num_active_routed_experts must be nonzero, within TopK capacity, and <= num_routed_experts")]
    InvalidActiveExpertCount,
    #[error("MoE expert_hidden_dim must be > 0 and 2 * expert_hidden_dim must fit in u32")]
    InvalidExpertHiddenDim,
    #[error("MoE shared experts are not supported")]
    UnsupportedSharedExperts,
    #[error("MoE expert gate is not supported")]
    UnsupportedExpertGate,
    #[error("MoE without router, up, and down biases is not supported")]
    UnsupportedNoBiases,
    #[error("Unsupported MoE router configuration: {0}")]
    UnsupportedRouterConfiguration(String),
    #[error("Unsupported MoE {projection} weight configuration: {spec}")]
    UnsupportedExpertWeightConfiguration {
        projection: &'static str,
        spec: String,
    },
    #[error("Unsupported MoE expert activation: {0:?}")]
    UnsupportedExpertActivation(ActivationType),
}

impl<B: Backend> MoeBlock<B> {
    fn expert_spec(weights_tree: &ParameterTree<B>) -> Result<AnyWeightMatrixSpec, ParameterLoaderError<B>> {
        match weights_tree.metadata::<AnyWeightMatrixSpec>("spec") {
            Ok(spec) => Ok(spec),
            Err(ParameterLoaderError::KeyNotFound(_)) => {
                Ok(AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec::output_input()))
            },
            Err(error) => Err(error),
        }
    }

    pub fn new(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, MoeBlockError<B>> {
        if !valid_model_dim(model_dim) {
            return Err(MoeBlockError::InvalidModelDim);
        }
        if !valid_routed_expert_count(moe_config.num_routed_experts) {
            return Err(MoeBlockError::InvalidRoutedExpertCount);
        }
        if !valid_active_expert_count(moe_config.num_active_routed_experts, moe_config.num_routed_experts) {
            return Err(MoeBlockError::InvalidActiveExpertCount);
        }
        let fused_hidden_dim = moe_config
            .expert_hidden_dim
            .checked_mul(2)
            .filter(|_| moe_config.expert_hidden_dim > 0)
            .ok_or(MoeBlockError::InvalidExpertHiddenDim)?;
        if moe_config.num_shared_experts != 0 {
            return Err(MoeBlockError::UnsupportedSharedExperts);
        }
        if moe_config.gate_config.is_some() {
            return Err(MoeBlockError::UnsupportedExpertGate);
        }
        if !moe_config.router_has_biases
            || !moe_config.expert_config.has_up_biases
            || !moe_config.expert_config.has_down_biases
        {
            return Err(MoeBlockError::UnsupportedNoBiases);
        }
        match moe_config.expert_config.activation.act_type() {
            ActivationType::GELUApprox | ActivationType::SILU => {},
            activation => return Err(MoeBlockError::UnsupportedExpertActivation(activation)),
        }

        let router_tree = parameter_tree.subtree("router");
        let router_weights_tree = router_tree.subtree("weights");
        let router_spec = router_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
            layout: Layout::OutputInput,
            ..
        }) = &router_spec
        else {
            return Err(MoeBlockError::UnsupportedRouterConfiguration(format!("{router_spec:?}")));
        };
        let router_weights = router_weights_tree
            .leaf("weights")?
            .validate(&[moe_config.num_routed_experts, model_dim], data_type)?
            .read_allocation()?;
        let router_biases =
            router_tree.leaf("biases")?.validate(&[moe_config.num_routed_experts], data_type)?.read_allocation()?;

        let experts_tree = parameter_tree.subtree("experts");
        let up_tree = experts_tree.subtree("up_projection");
        let down_tree = experts_tree.subtree("down_projection");
        let up_weights_tree = up_tree.subtree("weights");
        let up_spec = Self::expert_spec(&up_weights_tree)?;
        if !matches!(&up_spec, AnyWeightMatrixSpec::FullPrecisionSpec(_) | AnyWeightMatrixSpec::MicrofloatSpec(_)) {
            return Err(MoeBlockError::UnsupportedExpertWeightConfiguration {
                projection: "up-projection",
                spec: format!("{up_spec:?}"),
            });
        }
        let up_projection = WeightMatrix::load_bank(
            &up_weights_tree,
            up_spec,
            Layout::OutputInput,
            moe_config.num_routed_experts,
            fused_hidden_dim,
            model_dim,
            data_type,
        )?;
        let down_weights_tree = down_tree.subtree("weights");
        let down_spec = Self::expert_spec(&down_weights_tree)?;
        if !matches!(&down_spec, AnyWeightMatrixSpec::FullPrecisionSpec(_) | AnyWeightMatrixSpec::MicrofloatSpec(_)) {
            return Err(MoeBlockError::UnsupportedExpertWeightConfiguration {
                projection: "down-projection",
                spec: format!("{down_spec:?}"),
            });
        }
        let down_projection = WeightMatrix::load_bank(
            &down_weights_tree,
            down_spec,
            Layout::OutputInput,
            moe_config.num_routed_experts,
            model_dim,
            moe_config.expert_hidden_dim,
            data_type,
        )?;
        let up_biases = up_tree
            .leaf("biases")?
            .validate(&[moe_config.num_routed_experts, fused_hidden_dim], data_type)?
            .read_allocation()?;
        let down_biases = down_tree
            .leaf("biases")?
            .validate(&[moe_config.num_routed_experts, model_dim], data_type)?
            .read_allocation()?;

        let router = MoeRouter::new(
            context,
            router_weights,
            router_biases,
            model_dim,
            moe_config.num_routed_experts,
            moe_config.num_active_routed_experts,
            matches!(
                moe_config.routing_function,
                crate::config::mlp::routing_function::AnyRoutingFunction::SoftmaxRouting(_)
            ),
            data_type,
        )
        .map_err(MoeBlockError::BackendError)?;
        let experts = MoeExperts::new(
            context,
            up_projection,
            down_projection,
            up_biases,
            down_biases,
            model_dim,
            moe_config.expert_hidden_dim,
            fused_hidden_dim,
            moe_config.num_routed_experts,
            moe_config.expert_config.activation.clone(),
            moe_config.expert_config.gate_clipping,
            moe_config.expert_config.up_clipping,
            data_type,
        )
        .map_err(MoeBlockError::BackendError)?;

        Ok(Self {
            router,
            experts,
        })
    }
}

impl<B: Backend> Mlp<B> for MoeBlock<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("mlp (moe)");
        let routes = self.router.route(&input, batch_dim, encoder)?;
        let output = self.experts.encode(&input, &routes, encoder)?;
        encoder.pop_debug_group();
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/encodable_block/moe/mod.rs"]
mod tests;
