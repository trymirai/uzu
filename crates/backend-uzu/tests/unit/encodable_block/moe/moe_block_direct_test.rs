use std::io::Write;

use half::{bf16, f16};
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

fn parameter_file(data_type: DataType) -> NamedTempFile {
    let dtype = match data_type {
        DataType::F16 => "F16",
        DataType::BF16 => "BF16",
        _ => panic!("unsupported full-precision test data type"),
    };
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
        let end = offset + size_for_shape(&shape, data_type);
        header.insert(
            name.into(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }

    write_parameter_file(header, &[])
}

fn write_parameter_file(
    header: Map<String, Value>,
    payload: &[u8],
) -> NamedTempFile {
    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = NamedTempFile::new().expect("create parameter file");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
    file.write_all(&header).expect("write header");
    file.write_all(payload).expect("write tensors");
    file
}

fn add_tensor(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: &str,
    shape: Vec<u32>,
    data_type: DataType,
    data: Vec<u8>,
) {
    assert_eq!(data.len(), size_for_shape(&shape, data_type));
    let begin = payload.len();
    payload.extend_from_slice(&data);
    let dtype = match data_type {
        DataType::BF16 => "BF16",
        DataType::U8 => "U8",
        _ => panic!("unsupported test tensor data type"),
    };
    header.insert(
        name.into(),
        json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, payload.len()]
        }),
    );
}

fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(|value| bf16::from_f32(value).to_le_bytes()).collect()
}

fn microfloat_parameter_file() -> NamedTempFile {
    let mut header = Map::new();
    header.insert(
        "__metadata__".into(),
        json!({
            "router.weights.spec": json!({
                "type": "FullPrecisionSpec",
                "layout": "output_input"
            }).to_string(),
            "experts.up_projection.weights.spec": json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": 16,
                "scale_mode": "mxfp4",
                "layout": "output_input"
            }).to_string(),
            "experts.down_projection.weights.spec": json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": 32,
                "scale_mode": "mxfp4",
                "layout": "output_input"
            }).to_string()
        }),
    );
    let mut payload = Vec::new();
    add_tensor(
        &mut header,
        &mut payload,
        "router.weights.weights",
        vec![EXPERTS, MODEL_DIM],
        DataType::BF16,
        bf16_bytes(std::iter::repeat_n(0.0, (EXPERTS * MODEL_DIM) as usize)),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "router.biases",
        vec![EXPERTS],
        DataType::BF16,
        bf16_bytes([0.4, 0.2, 0.0, -0.2]),
    );

    let up_code_count = (EXPERTS * 2 * HIDDEN_DIM * MODEL_DIM / 2) as usize;
    add_tensor(
        &mut header,
        &mut payload,
        "experts.up_projection.weights.weights",
        vec![EXPERTS, 2 * HIDDEN_DIM, MODEL_DIM / 2],
        DataType::U8,
        (0..up_code_count).map(|index| ((index % 7 + 1) | (((index * 3 + 1) % 7 + 1) << 4)) as u8).collect(),
    );
    let up_scale_count = (EXPERTS * 2 * HIDDEN_DIM * MODEL_DIM / 16) as usize;
    add_tensor(
        &mut header,
        &mut payload,
        "experts.up_projection.weights.scales",
        vec![EXPERTS, 2 * HIDDEN_DIM, MODEL_DIM / 16],
        DataType::U8,
        (0..up_scale_count).map(|index| 123 + (index % 3) as u8).collect(),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "experts.up_projection.weights.global_scale",
        vec![EXPERTS],
        DataType::BF16,
        bf16_bytes([0.5, 0.75, 1.0, 1.25]),
    );

    let down_code_count = (EXPERTS * MODEL_DIM * HIDDEN_DIM / 2) as usize;
    add_tensor(
        &mut header,
        &mut payload,
        "experts.down_projection.weights.weights",
        vec![EXPERTS, MODEL_DIM, HIDDEN_DIM / 2],
        DataType::U8,
        (0..down_code_count)
            .map(|index| (((index * 5 + 2) % 7 + 1) | (((index * 7 + 3) % 7 + 1) << 4)) as u8)
            .collect(),
    );
    let down_scale_count = (EXPERTS * MODEL_DIM * HIDDEN_DIM / 32) as usize;
    add_tensor(
        &mut header,
        &mut payload,
        "experts.down_projection.weights.scales",
        vec![EXPERTS, MODEL_DIM, HIDDEN_DIM / 32],
        DataType::U8,
        (0..down_scale_count).map(|index| 122 + (index % 4) as u8).collect(),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "experts.down_projection.weights.global_scale",
        vec![EXPERTS],
        DataType::BF16,
        bf16_bytes([1.0, 0.75, 0.5, 0.25]),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "experts.up_projection.biases",
        vec![EXPERTS, 2 * HIDDEN_DIM],
        DataType::BF16,
        bf16_bytes((0..EXPERTS * 2 * HIDDEN_DIM).map(|index| ((index % 9) as f32 - 4.0) * 0.005)),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "experts.down_projection.biases",
        vec![EXPERTS, MODEL_DIM],
        DataType::BF16,
        bf16_bytes((0..EXPERTS * MODEL_DIM).map(|index| ((index % 7) as f32 - 3.0) * 0.004)),
    );
    write_parameter_file(header, &payload)
}

fn run<B: Backend>(
    token_count: u32,
    microfloat: bool,
) -> Vec<bf16> {
    let context = B::Context::new().expect("create context");
    let file = if microfloat {
        microfloat_parameter_file()
    } else {
        parameter_file(DataType::BF16)
    };
    let loader = if microfloat {
        ParameterLoader::<B>::new(file.as_file(), context.as_ref())
    } else {
        ParameterLoader::<B>::new_random(file.as_file(), context.as_ref(), 41)
    }
    .expect("load parameters");
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

fn run_f16<B: Backend>(token_count: u32) -> Vec<f16> {
    let context = B::Context::new().expect("create context");
    let file = parameter_file(DataType::F16);
    let loader = ParameterLoader::<B>::new_random(file.as_file(), context.as_ref(), 41).expect("load F16 parameters");
    let tree = loader.tree();
    let block = MoeBlock::<B>::new(context.as_ref(), &config(), MODEL_DIM, DataType::F16, &tree)
        .expect("construct F16 MoeBlock");
    tree.assert_all_tensors_validated().expect("validate all F16 parameters");

    let input: Vec<f16> =
        (0..token_count * MODEL_DIM).map(|index| f16::from_f32((index % 17) as f32 * 0.03 - 0.2)).collect();
    let input = alloc_allocation_with_data::<B, f16>(context.as_ref(), &input);
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    let output = block.encode(input, token_count, &mut encoder).expect("encode F16 MoeBlock");
    let completed = encoder.end_encoding().submit().wait_until_completed().expect("execute F16 MoeBlock");
    let values = allocation_to_vec::<B, f16>(&output);
    drop(output);
    drop(completed);
    values
}

#[uzu_test]
fn direct_routes_cover_decode_and_prefill() {
    for token_count in [1, 33, 257] {
        let expected = run::<crate::backends::cpu::Cpu>(token_count, false);
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(token_count, false);
            assert_eq_float(&expected, &actual, 0.15, "direct MoeBlock routes");
        });
    }
}

#[uzu_test]
fn microfloat_experts_cover_decode_and_prefill() {
    for token_count in [1, 33, 257] {
        let expected = run::<crate::backends::cpu::Cpu>(token_count, true);
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(token_count, true);
            assert_eq_float(&expected, &actual, 0.15, "microfloat MoeBlock routes");
        });
    }
}

#[uzu_test]
fn full_precision_f16_experts_remain_supported() {
    for token_count in [1, 33] {
        let expected = run_f16::<crate::backends::cpu::Cpu>(token_count);
        for_each_non_cpu_backend!(|B| {
            let actual = run_f16::<B>(token_count);
            assert_eq_float(&expected, &actual, 0.15, "F16 MoeBlock routes");
        });
    }
}
