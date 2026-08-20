use std::io::Write;

use half::{bf16, f16};
use proc_macros::uzu_test;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;

use super::super::MoeBlock;
use crate::{
    array::size_for_shape,
    backends::common::{Backend, Context, Encoder},
    config::{
        mlp::mixture_of_experts::MixtureOfExpertsConfig,
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::DataType,
    encodable_block::{mlp::Mlp, weight_matrix::WeightMatrix},
    parameters::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const MODEL_DIM: u32 = 32;
const HIDDEN_DIM: u32 = 64;
const EXPERTS: u32 = 4;
const ROUTES_PER_TOKEN: u32 = 2;
const ROUTER_SLOPES: [f32; EXPERTS as usize] = [-4.0, -3.0, 4.0, 3.0];

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

fn fixture_e2m1(code: u8) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code)]
}

fn fixture_microfloat_weight(
    up_projection: bool,
    expert: usize,
    row: usize,
    column: usize,
) -> f32 {
    let (rows, columns, group_size) = if up_projection {
        ((2 * HIDDEN_DIM) as usize, MODEL_DIM as usize, 16usize)
    } else {
        (MODEL_DIM as usize, HIDDEN_DIM as usize, 32usize)
    };
    let element = (expert * rows + row) * columns + column;
    let byte = element / 2;
    let (low, high, exponent, outer_scale) = if up_projection {
        (
            (byte % 7 + 1) as u8,
            ((byte * 3 + 1) % 7 + 1) as u8,
            123 + ((expert * rows + row) * (columns / group_size) + column / group_size) % 3,
            [0.5, 0.75, 1.0, 1.25][expert],
        )
    } else {
        (
            ((byte * 5 + 2) % 7 + 1) as u8,
            ((byte * 7 + 3) % 7 + 1) as u8,
            122 + ((expert * rows + row) * (columns / group_size) + column / group_size) % 4,
            [1.0, 0.75, 0.5, 0.25][expert],
        )
    };
    let code = if column.is_multiple_of(2) {
        low
    } else {
        high
    };
    fixture_e2m1(code) * 2.0f32.powi(exponent as i32 - 127) * outer_scale
}

fn independent_microfloat_output(token_count: usize) -> Vec<bf16> {
    let input: Vec<f32> = (0..token_count * MODEL_DIM as usize)
        .map(|index| bf16::from_f32((index % 17) as f32 * 0.03 - 0.2).to_f32())
        .collect();
    let mut output = vec![0.0f32; token_count * MODEL_DIM as usize];

    for token in 0..token_count {
        let input_first = input[token * MODEL_DIM as usize];
        let experts = if input_first < 0.0 {
            [0usize, 1]
        } else {
            [2usize, 3]
        };
        let logits = experts.map(|expert| input_first * ROUTER_SLOPES[expert]);
        let denominator = logits[0].exp() + logits[1].exp();
        let route_weights = logits.map(|logit| bf16::from_f32(logit.exp() / denominator).to_f32());

        for (expert, route_weight) in experts.into_iter().zip(route_weights) {
            let mut fused_up = vec![0.0f32; (2 * HIDDEN_DIM) as usize];
            for row in 0..(2 * HIDDEN_DIM) as usize {
                let bias =
                    bf16::from_f32(((expert * (2 * HIDDEN_DIM) as usize + row) % 9) as f32 * 0.005 - 0.02).to_f32();
                fused_up[row] = (0..MODEL_DIM as usize).fold(bias, |sum, column| {
                    sum + input[token * MODEL_DIM as usize + column]
                        * fixture_microfloat_weight(true, expert, row, column)
                });
            }
            let hidden: Vec<f32> = (0..HIDDEN_DIM as usize)
                .map(|column| {
                    let value = fused_up[column].clamp(-2.0, 2.5);
                    let gate = fused_up[HIDDEN_DIM as usize + column].clamp(-1.5, 2.0);
                    value * gate / (1.0 + (-0.75 * gate).exp())
                })
                .collect();
            for row in 0..MODEL_DIM as usize {
                let bias = bf16::from_f32(((expert * MODEL_DIM as usize + row) % 7) as f32 * 0.004 - 0.012).to_f32();
                let down = (0..HIDDEN_DIM as usize).fold(bias, |sum, column| {
                    sum + hidden[column] * fixture_microfloat_weight(false, expert, row, column)
                });
                output[token * MODEL_DIM as usize + row] += route_weight * bf16::from_f32(down).to_f32();
            }
        }
    }
    output.into_iter().map(bf16::from_f32).collect()
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
        bf16_bytes((0..EXPERTS as usize * MODEL_DIM as usize).map(|index| {
            if index.is_multiple_of(MODEL_DIM as usize) {
                ROUTER_SLOPES[index / MODEL_DIM as usize]
            } else {
                0.0
            }
        })),
    );
    add_tensor(
        &mut header,
        &mut payload,
        "router.biases",
        vec![EXPERTS],
        DataType::BF16,
        bf16_bytes([0.0; EXPERTS as usize]),
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
fn microfloat_experts_match_an_independent_scalar_oracle() {
    let expected = independent_microfloat_output(2);
    let actual = run::<crate::backends::cpu::Cpu>(2, true);
    assert_eq_float(&expected, &actual, 0.15, "scalar MXFP4 MoeBlock oracle");
    for_each_non_cpu_backend!(|B| {
        let actual = run::<B>(2, true);
        assert_eq_float(&expected, &actual, 0.15, "scalar MXFP4 MoeBlock oracle");
    });
}

#[uzu_test]
fn banked_microfloat_tensors_require_the_bank_loader() {
    let context = <crate::backends::cpu::Cpu as Backend>::Context::new().expect("create context");
    let file = microfloat_parameter_file();
    let loader = ParameterLoader::<crate::backends::cpu::Cpu>::new(file.as_file(), context.as_ref())
        .expect("load microfloat parameters");
    let tree = loader.tree().subtree("experts").subtree("up_projection").subtree("weights");
    let spec = tree.metadata::<AnyWeightMatrixSpec>("spec").expect("microfloat spec");
    let result = WeightMatrix::load(&tree, spec, Layout::OutputInput, 2 * HIDDEN_DIM, MODEL_DIM, DataType::BF16);
    assert!(result.is_err(), "bank-shaped tensor was accepted as a single dense matrix");
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
