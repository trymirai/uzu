use std::io::Write;

use half::bf16;
use proc_macros::uzu_test;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;

use super::super::MoeBlock;
use crate::{
    array::size_for_shape,
    backends::common::{Backend, Context, Encoder},
    config::mlp::mixture_of_experts::MixtureOfExpertsConfig,
    data_type::DataType,
    encodable_block::mlp::Mlp,
    parameters::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const MODEL_DIM: u32 = 16;
const HIDDEN_DIM: u32 = 32;
const EXPERTS: u32 = 4;
const ROUTES_PER_TOKEN: u32 = 2;

fn config() -> MixtureOfExpertsConfig {
    serde_json::from_value(json!({
        "type": "MixtureOfExpertsConfig",
        "expert_config": {
            "type": "DenseMLPConfig",
            "linear_config": {},
            "activation": { "type": "SiLU", "alpha": 0.75 },
            "has_up_biases": true,
            "has_down_biases": true,
            "gate_clipping": [-1.5, 2.0],
            "up_clipping": [-2.0, 2.5]
        },
        "router_config": {},
        "routing_function": { "type": "SoftmaxRouting" },
        "num_routed_experts": EXPERTS,
        "num_active_routed_experts": ROUTES_PER_TOKEN,
        "router_has_biases": true,
        "num_shared_experts": 0,
        "expert_hidden_dim": HIDDEN_DIM,
        "gate_config": null
    }))
    .expect("valid MoE configuration")
}

fn parameter_file() -> NamedTempFile {
    let mut header = Map::new();
    header.insert(
        "__metadata__".into(),
        json!({
            "router.weights.spec": json!({
                "type": "FullPrecisionSpec",
                "layout": "output_input"
            }).to_string()
        }),
    );
    let tensors = [
        ("router.weights.weights", vec![EXPERTS, MODEL_DIM]),
        ("router.biases", vec![EXPERTS]),
        ("experts.up_projection.weights.weights", vec![EXPERTS, 2 * HIDDEN_DIM, MODEL_DIM]),
        ("experts.down_projection.weights.weights", vec![EXPERTS, MODEL_DIM, HIDDEN_DIM]),
        ("experts.up_projection.biases", vec![EXPERTS, 2 * HIDDEN_DIM]),
        ("experts.down_projection.biases", vec![EXPERTS, MODEL_DIM]),
    ];
    let mut offset = 0usize;
    for (name, shape) in tensors {
        let end = offset + size_for_shape(&shape, DataType::BF16);
        header.insert(
            name.into(),
            json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = NamedTempFile::new().expect("create parameter file");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
    file.write_all(&header).expect("write header");
    file
}

fn run<B: Backend>(token_count: u32) -> Vec<bf16> {
    let context = B::Context::new().expect("create context");
    let file = parameter_file();
    let loader = ParameterLoader::<B>::new_random(file.as_file(), context.as_ref(), 41).expect("load parameters");
    let tree = loader.tree();
    let block =
        MoeBlock::<B>::new(context.as_ref(), &config(), MODEL_DIM, DataType::BF16, &tree).expect("construct MoeBlock");
    tree.assert_all_tensors_validated().expect("validate all parameters");

    let input: Vec<bf16> =
        (0..token_count * MODEL_DIM).map(|index| bf16::from_f32((index % 17) as f32 * 0.03 - 0.2)).collect();
    let input = alloc_allocation_with_data::<B, bf16>(context.as_ref(), &input);
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    let output = block.encode(input, token_count, &mut encoder).expect("encode MoeBlock");
    let completed = encoder.end_encoding().submit().wait_until_completed().expect("execute MoeBlock");
    let values = allocation_to_vec::<B, bf16>(&output);
    drop(output);
    drop(completed);
    values
}

#[uzu_test]
fn direct_routes_cover_decode_and_prefill() {
    for token_count in [1, 33, 257] {
        let expected = run::<crate::backends::cpu::Cpu>(token_count);
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(token_count);
            assert_eq_float(&expected, &actual, 0.15, "direct MoeBlock routes");
        });
    }
}
