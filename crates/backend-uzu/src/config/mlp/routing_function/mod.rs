use proc_macros::uzu_config_abstract;

pub mod softmax_routing;

#[uzu_config_abstract(softmax_routing::SoftmaxRouting)]
pub struct RoutingFunction;
