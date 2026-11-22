use uzu_engine_macros::uzu_config_abstract;

pub mod sigmoid_routing;
pub mod softmax_routing;

#[uzu_config_abstract(softmax_routing::SoftmaxRouting, sigmoid_routing::SigmoidRouting)]
pub struct RoutingFunction;
