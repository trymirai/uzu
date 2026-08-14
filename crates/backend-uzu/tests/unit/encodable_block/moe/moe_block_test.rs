use std::io::Write;

use proc_macros::uzu_test;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;

use super::super::{MoeBlock, MoeBlockError};
use crate::{
    array::size_for_shape,
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    config::mlp::mixture_of_experts::MixtureOfExpertsConfig,
    data_type::DataType,
    parameters::ParameterLoader,
};

const MODEL_DIM: u32 = 4;
const HIDDEN_DIM: u32 = 4;

fn moe_config(
    num_routed_experts: u32,
    num_active_routed_experts: u32,
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
        "num_routed_experts": num_routed_experts,
        "num_active_routed_experts": num_active_routed_experts,
        "router_has_biases": true,
        "num_shared_experts": 0,
        "expert_hidden_dim": HIDDEN_DIM,
        "gate_config": null
    }))
    .expect("valid MoE test configuration")
}

fn parameter_header(num_routed_experts: Option<u32>) -> NamedTempFile {
    let mut header = Map::new();

    if let Some(num_routed_experts) = num_routed_experts {
        header.insert(
            "__metadata__".to_string(),
            json!({
                "router.weights.spec": json!({
                    "type": "FullPrecisionSpec",
                    "layout": "output_input"
                })
                .to_string()
            }),
        );
        let tensors = [
            ("router.weights.weights", vec![num_routed_experts, MODEL_DIM]),
            ("router.biases", vec![num_routed_experts]),
            ("experts.up_projection.weights.weights", vec![num_routed_experts, HIDDEN_DIM * 2, MODEL_DIM]),
            ("experts.down_projection.weights.weights", vec![num_routed_experts, MODEL_DIM, HIDDEN_DIM]),
            ("experts.up_projection.biases", vec![num_routed_experts, HIDDEN_DIM * 2]),
            ("experts.down_projection.biases", vec![num_routed_experts, MODEL_DIM]),
        ];
        let mut offset = 0usize;

        for (name, shape) in tensors {
            let end = offset + size_for_shape(&shape, DataType::BF16);
            header.insert(
                name.to_string(),
                json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
    }

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize parameter header");
    let padding = (8 - header.len() % 8) % 8;
    header.extend(std::iter::repeat_n(b' ', padding));

    // Random loading materializes tensors from metadata, so the virtual data ranges need no payload.
    let mut file = NamedTempFile::new().expect("create parameter header");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write parameter header length");
    file.write_all(&header).expect("write parameter header");
    file
}

fn construct_moe_block(
    num_routed_experts: u32,
    num_active_routed_experts: u32,
    parameter_experts: Option<u32>,
) -> Result<(), MoeBlockError<Cpu>> {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let header = parameter_header(parameter_experts);
    let loader =
        ParameterLoader::<Cpu>::new_random(header.as_file(), context.as_ref(), 0).expect("load parameter header");
    let tree = loader.tree();
    let config = moe_config(num_routed_experts, num_active_routed_experts);

    let _block = MoeBlock::<Cpu>::new(context.as_ref(), &config, MODEL_DIM, DataType::BF16, &tree)?;
    tree.assert_all_tensors_validated()?;
    Ok(())
}

#[uzu_test]
fn test_moe_block_rejects_invalid_expert_counts() {
    for (num_routed_experts, num_active_routed_experts) in [(0, 1), (513, 1)] {
        let error = construct_moe_block(num_routed_experts, num_active_routed_experts, None)
            .expect_err("invalid routed expert count was accepted");
        assert!(matches!(error, MoeBlockError::InvalidRoutedExpertCount), "unexpected error: {error}");
    }
    for (num_routed_experts, num_active_routed_experts) in [(1, 0), (1, 2), (512, 129)] {
        let error = construct_moe_block(num_routed_experts, num_active_routed_experts, None)
            .expect_err("invalid active expert count was accepted");
        assert!(matches!(error, MoeBlockError::InvalidActiveExpertCount), "unexpected error: {error}");
    }
}

#[uzu_test]
fn test_moe_block_accepts_expert_count_boundaries() {
    for (num_routed_experts, num_active_routed_experts) in [(1, 1), (128, 128), (512, 128)] {
        construct_moe_block(num_routed_experts, num_active_routed_experts, Some(num_routed_experts))
            .unwrap_or_else(|error| panic!("valid expert counts were rejected: {error}"));
    }
}
