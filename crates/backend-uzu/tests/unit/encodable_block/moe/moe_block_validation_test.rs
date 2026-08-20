use std::io::Write;

use proc_macros::uzu_test;
use serde_json::{Value, json};
use tempfile::NamedTempFile;

use super::super::{MoeBlock, MoeBlockError, valid_active_expert_count, valid_model_dim, valid_routed_expert_count};
use crate::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    config::mlp::mixture_of_experts::MixtureOfExpertsConfig,
    data_type::DataType,
    parameters::ParameterLoader,
};

fn config(
    routed_experts: u32,
    active_experts: u32,
) -> MixtureOfExpertsConfig {
    serde_json::from_value(json!({
        "type": "MixtureOfExpertsConfig",
        "expert_config": {
            "type": "DenseMLPConfig",
            "linear_config": {},
            "activation": { "type": "SiLU", "alpha": 1.0 },
            "has_up_biases": true,
            "has_down_biases": true,
            "gate_clipping": null,
            "up_clipping": null
        },
        "router_config": {},
        "routing_function": { "type": "SoftmaxRouting" },
        "num_routed_experts": routed_experts,
        "num_active_routed_experts": active_experts,
        "router_has_biases": true,
        "num_shared_experts": 0,
        "expert_hidden_dim": 32,
        "gate_config": null
    }))
    .expect("valid test configuration")
}

fn empty_parameter_file() -> NamedTempFile {
    let mut header = serde_json::to_vec(&Value::Object(Default::default())).expect("serialize parameter header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));

    let mut file = NamedTempFile::new().expect("create parameter file");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
    file.write_all(&header).expect("write header");
    file
}

fn constructor_error(
    routed_experts: u32,
    active_experts: u32,
) -> MoeBlockError<Cpu> {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = empty_parameter_file();
    let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load empty parameter file");

    match MoeBlock::<Cpu>::new(
        context.as_ref(),
        &config(routed_experts, active_experts),
        16,
        DataType::BF16,
        &loader.tree(),
    ) {
        Ok(_) => panic!("invalid expert counts were accepted"),
        Err(error) => error,
    }
}

fn model_dim_error(model_dim: u32) -> MoeBlockError<Cpu> {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = empty_parameter_file();
    let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load empty parameter file");

    match MoeBlock::<Cpu>::new(context.as_ref(), &config(1, 1), model_dim, DataType::BF16, &loader.tree()) {
        Ok(_) => panic!("invalid model dimension was accepted"),
        Err(error) => error,
    }
}

fn hidden_dim_error(hidden_dim: u32) -> MoeBlockError<Cpu> {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = empty_parameter_file();
    let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load empty parameter file");
    let mut config = config(1, 1);
    config.expert_hidden_dim = hidden_dim;

    match MoeBlock::<Cpu>::new(context.as_ref(), &config, 16, DataType::BF16, &loader.tree()) {
        Ok(_) => panic!("invalid hidden dimension was accepted"),
        Err(error) => error,
    }
}

#[uzu_test]
fn rejects_invalid_routed_expert_counts() {
    assert!(valid_routed_expert_count(512));
    for routed_experts in [0, 513] {
        assert!(matches!(constructor_error(routed_experts, 1), MoeBlockError::InvalidRoutedExpertCount));
    }
}

#[uzu_test]
fn rejects_invalid_active_expert_counts() {
    assert!(valid_active_expert_count(128, 512));
    for (routed_experts, active_experts) in [(1, 0), (1, 2), (512, 129)] {
        assert!(matches!(constructor_error(routed_experts, active_experts), MoeBlockError::InvalidActiveExpertCount));
    }
}

#[uzu_test]
fn rejects_router_dimensions_beyond_its_threadgroup_cache() {
    assert!(valid_model_dim(4096));
    for model_dim in [0, 2, 4100] {
        assert!(matches!(model_dim_error(model_dim), MoeBlockError::InvalidModelDim));
    }
}

#[uzu_test]
fn rejects_invalid_expert_hidden_dimensions() {
    for hidden_dim in [0, u32::MAX] {
        assert!(matches!(hidden_dim_error(hidden_dim), MoeBlockError::InvalidExpertHiddenDim));
    }
}
