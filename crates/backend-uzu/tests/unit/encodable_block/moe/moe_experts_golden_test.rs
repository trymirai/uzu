use std::{io::Write, num::NonZeroU32};

use proc_macros::uzu_test;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;

use super::super::{experts::MoeExperts, router::MoeRoutes};
use crate::{
    backends::common::{Backend, Context, Encoder},
    config::{
        activation::AnyActivation,
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::DataType,
    encodable_block::weight_matrix::WeightMatrix,
    parameters::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const TOKENS: usize = 2;
const ROUTES_PER_TOKEN: usize = 2;
const EXPERTS: usize = 3;
const MODEL_DIM: usize = 4;
const HIDDEN_DIM: usize = 2;
const FUSED_HIDDEN_DIM: usize = 2 * HIDDEN_DIM;

const INPUT: [f32; TOKENS * MODEL_DIM] = [1.0, 2.0, -1.0, 0.5, -2.0, 1.0, 0.5, 3.0];
const EXPERT_IDS: [i32; TOKENS * ROUTES_PER_TOKEN] = [2, 0, 1, 2];
const ROUTE_WEIGHTS: [f32; TOKENS * ROUTES_PER_TOKEN] = [0.75, 0.25, 0.40, 0.60];

#[rustfmt::skip]
const UP_WEIGHTS: [f32; EXPERTS * FUSED_HIDDEN_DIM * MODEL_DIM] = [
    1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0,

    1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 1.0,
    1.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 1.0,

    2.0, 0.0, -1.0, 0.0,  0.0, -1.0, 0.0, 2.0,
    0.5, 0.0, 0.0, 1.0,  0.0, 1.0, -0.5, 0.0,
];

const UP_BIASES: [f32; EXPERTS * FUSED_HIDDEN_DIM] = [0.1, -0.2, 0.3, -0.4, -0.3, 0.2, 0.4, -0.1, 0.5, -0.6, 0.7, -0.8];

#[rustfmt::skip]
const DOWN_WEIGHTS: [f32; EXPERTS * MODEL_DIM * HIDDEN_DIM] = [
    1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  -1.0, 0.5,
    0.5, 1.0,  1.0, -0.5,  -1.0, 1.0,  2.0, 0.25,
    1.5, -1.0,  0.25, 2.0,  1.0, 0.5,  -0.75, 1.25,
];

const DOWN_BIASES: [f32; EXPERTS * MODEL_DIM] = [0.01, 0.02, 0.03, 0.04, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8];

fn tensor(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: &str,
    shape: &[usize],
    values: &[f32],
) {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let begin = payload.len();
    payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
    header.insert(
        name.into(),
        json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [begin, payload.len()]
        }),
    );
}

fn parameter_file() -> NamedTempFile {
    let mut header = Map::new();
    let mut payload = Vec::new();
    tensor(&mut header, &mut payload, "up.weights", &[EXPERTS, FUSED_HIDDEN_DIM, MODEL_DIM], &UP_WEIGHTS);
    tensor(&mut header, &mut payload, "down.weights", &[EXPERTS, MODEL_DIM, HIDDEN_DIM], &DOWN_WEIGHTS);
    tensor(&mut header, &mut payload, "up_biases", &[EXPERTS, FUSED_HIDDEN_DIM], &UP_BIASES);
    tensor(&mut header, &mut payload, "down_biases", &[EXPERTS, MODEL_DIM], &DOWN_BIASES);

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = NamedTempFile::new().expect("create parameter file");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
    file.write_all(&header).expect("write header");
    file.write_all(&payload).expect("write payload");
    file
}

fn scalar_expected() -> Vec<f32> {
    let mut output = vec![0.0; TOKENS * MODEL_DIM];
    for route in 0..EXPERT_IDS.len() {
        let token = route / ROUTES_PER_TOKEN;
        let expert = EXPERT_IDS[route] as usize;
        let mut fused = [0.0; FUSED_HIDDEN_DIM];
        for row in 0..FUSED_HIDDEN_DIM {
            fused[row] = UP_BIASES[expert * FUSED_HIDDEN_DIM + row];
            for column in 0..MODEL_DIM {
                fused[row] += INPUT[token * MODEL_DIM + column]
                    * UP_WEIGHTS[(expert * FUSED_HIDDEN_DIM + row) * MODEL_DIM + column];
            }
        }

        let mut hidden = [0.0; HIDDEN_DIM];
        for index in 0..HIDDEN_DIM {
            let value = fused[index].clamp(-2.0, 2.5);
            let gate = fused[HIDDEN_DIM + index].clamp(-1.5, 2.0);
            hidden[index] = value * gate / (1.0 + (-gate).exp());
        }

        for row in 0..MODEL_DIM {
            let mut down = DOWN_BIASES[expert * MODEL_DIM + row];
            for column in 0..HIDDEN_DIM {
                down += hidden[column] * DOWN_WEIGHTS[(expert * MODEL_DIM + row) * HIDDEN_DIM + column];
            }
            output[token * MODEL_DIM + row] += ROUTE_WEIGHTS[route] * down;
        }
    }
    output
}

fn run<B: Backend>() -> Vec<f32> {
    let context = B::Context::new().expect("create context");
    let file = parameter_file();
    let loader = ParameterLoader::<B>::new(file.as_file(), context.as_ref()).expect("load golden expert parameters");
    let tree = loader.tree();
    let spec: AnyWeightMatrixSpec =
        serde_json::from_value(json!({"type": "FullPrecisionSpec", "layout": "output_input"}))
            .expect("full-precision spec");
    let up = WeightMatrix::load_bank(
        &tree.subtree("up"),
        spec.clone(),
        Layout::OutputInput,
        EXPERTS as u32,
        FUSED_HIDDEN_DIM as u32,
        MODEL_DIM as u32,
        DataType::F32,
    )
    .expect("load up projection");
    let down = WeightMatrix::load_bank(
        &tree.subtree("down"),
        spec,
        Layout::OutputInput,
        EXPERTS as u32,
        MODEL_DIM as u32,
        HIDDEN_DIM as u32,
        DataType::F32,
    )
    .expect("load down projection");
    let up_biases = tree
        .leaf("up_biases")
        .unwrap()
        .validate(&[EXPERTS as u32, FUSED_HIDDEN_DIM as u32], DataType::F32)
        .unwrap()
        .read_allocation()
        .unwrap();
    let down_biases = tree
        .leaf("down_biases")
        .unwrap()
        .validate(&[EXPERTS as u32, MODEL_DIM as u32], DataType::F32)
        .unwrap()
        .read_allocation()
        .unwrap();
    let activation: AnyActivation =
        serde_json::from_value(json!({"type": "SiLU", "alpha": 1.0})).expect("SiLU activation");
    let experts = MoeExperts::new(
        context.as_ref(),
        up,
        down,
        up_biases,
        down_biases,
        MODEL_DIM as u32,
        HIDDEN_DIM as u32,
        FUSED_HIDDEN_DIM as u32,
        EXPERTS as u32,
        activation,
        Some((Some(-1.5), Some(2.0))),
        Some((Some(-2.0), Some(2.5))),
        DataType::F32,
    )
    .expect("construct experts");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &INPUT);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &EXPERT_IDS);
    let route_weights = alloc_allocation_with_data::<B, f32>(context.as_ref(), &ROUTE_WEIGHTS);
    let routes = MoeRoutes::from_parts(
        expert_ids,
        route_weights,
        TOKENS as u32,
        NonZeroU32::new(ROUTES_PER_TOKEN as u32).unwrap(),
    )
    .expect("valid golden routes");
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    let output = experts.encode(&input, &routes, &mut encoder).expect("encode experts");
    let completed = encoder.end_encoding().submit().wait_until_completed().expect("execute experts");
    let values = allocation_to_vec::<B, f32>(&output);
    drop(output);
    drop(completed);
    values
}

#[uzu_test]
fn scalar_golden_covers_the_complete_expert_equation() {
    let expected = scalar_expected();
    let actual = run::<crate::backends::cpu::Cpu>();
    assert_eq_float(&expected, &actual, 1e-5, "CPU scalar expert oracle");
    for_each_non_cpu_backend!(|B| {
        let actual = run::<B>();
        assert_eq_float(&expected, &actual, 1e-4, "Metal scalar expert oracle");
    });
}
