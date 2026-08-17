use std::io::Write;

use proc_macros::uzu_test;
use serde_json::{Value, json};
use tempfile::NamedTempFile;

use super::super::{MoeBlock, MoeBlockError};
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
    let loader =
        ParameterLoader::<Cpu>::new_random(file.as_file(), context.as_ref(), 0).expect("load empty parameter file");

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

#[uzu_test]
fn rejects_invalid_routed_expert_counts() {
    for routed_experts in [0, 513] {
        assert!(matches!(constructor_error(routed_experts, 1), MoeBlockError::InvalidRoutedExpertCount));
    }
}

#[uzu_test]
fn rejects_invalid_active_expert_counts() {
    for (routed_experts, active_experts) in [(1, 0), (1, 2), (512, 129)] {
        assert!(matches!(constructor_error(routed_experts, active_experts), MoeBlockError::InvalidActiveExpertCount));
    }
}
