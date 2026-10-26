use std::{fs::OpenOptions, io::Write, time::SystemTime};

use serde_json::json;
use uzu_engine_macros::uzu_test;

use super::ConfidenceHeadTree;
use crate::{
    backends::{
        common::{
            CommandBufferEncoding, Context,
            gpu_types::weaver::{FrontierIdx, TreeIdx},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::conf_head::config::ConfidenceHeadConfig,
    safetensors::ParameterLoader,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

#[uzu_test]
fn zero_layer_tree_head_normalizes_without_residual() {
    let config: ConfidenceHeadConfig = serde_json::from_value(json!({
        "type": "ConfidenceHeadConfig",
        "format": 2,
        "trained_with": {
            "speculator_sha256": "0".repeat(64),
            "tree": {"children": 1, "pool": 1, "max_depth": 1}
        },
        "target_width": 4,
        "local": {"hidden": 4, "pool_hidden": 4, "edge_hidden": 4, "logit_cap": 1.0},
        "attention": {"layers": 0, "heads": 1, "width": 4, "mlp_hidden": 4, "relations": 6},
        "layer_norm_eps": 1e-5
    }))
    .unwrap();

    // Identity projections expose the final normalization's output directly.
    let mut header = json!({"__metadata__": {}});
    let mut data = Vec::new();
    let mut tensors = Vec::new();
    for (name, input_width) in [("node_in", 4), ("child_in", 8), ("couple_node", 4), ("couple_child", 4)] {
        header["__metadata__"][format!("tree.{name}.weights.spec")] =
            json!({"type": "FullPrecisionSpec", "layout": "output_input"}).to_string().into();
        let mut weights = vec![0.0f32; 4 * input_width];
        for unit in 0..4 {
            weights[unit * input_width + unit] = 1.0;
        }
        tensors.push((format!("tree.{name}.weights.weights"), vec![4, input_width], weights));
        tensors.push((format!("tree.{name}.biases"), vec![4], vec![0.0; 4]));
    }
    tensors.extend([
        ("tree.final_norm.scales".into(), vec![4], vec![1.0; 4]),
        ("tree.final_norm.biases".into(), vec![4], vec![0.0; 4]),
        ("tree.depth_embedding".into(), vec![2, 4], vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
        ("tree.relation_weight".into(), vec![1, 6], vec![0.0; 6]),
    ]);
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header[name] = json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]});
    }
    let header = serde_json::to_vec(&header).unwrap();
    let nonce = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("uzu-zero-layer-tree-{}-{nonce}.safetensors", std::process::id()));
    let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();

    let context = create_context::<Cpu>();
    let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
    let head =
        ConfidenceHeadTree::<Cpu>::new(&context, &loader.tree(), &config, config.tree.as_ref().unwrap()).unwrap();
    loader.assert_all_tensors_read().unwrap();

    let mut tree = vec![0u32; TreeIdx::COUNT];
    tree[TreeIdx::Valid as usize] = 1;
    let mut frontier = vec![0u32; FrontierIdx::COUNT];
    frontier[FrontierIdx::TokenId as usize] = 7;
    frontier[FrontierIdx::Depth as usize] = 1;
    let tree = create_buffer_with_data::<Cpu, _>(&context, &tree);
    let frontier = create_buffer_with_data::<Cpu, _>(&context, &frontier);
    let real = create_buffer_with_data::<Cpu, _>(&context, &[1u32]);
    let node_data = create_buffer_with_data::<Cpu, _>(&context, &[1.0f32, 2.0, 3.0, 4.0]);
    let child_data = create_buffer_with_data::<Cpu, _>(&context, &[4.0f32, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let mut nodes = commands.allocate_scratch_for_shape(&[1, 4], DataType::F32).unwrap();
    let mut children = commands.allocate_scratch_for_shape(&[1, 8], DataType::F32).unwrap();
    let mut child_real = commands.allocate_scratch_for_shape(&[1], DataType::U32).unwrap();
    commands.encode_copy(&node_data, &mut nodes);
    commands.encode_copy(&child_data, &mut children);
    commands.encode_copy(&real, &mut child_real);
    let output = head.encode(&nodes, &children, &child_real, &tree, &frontier, 1, 1, &mut commands).unwrap();
    submit_command_buffer(commands);

    let scale = (2.0 + config.layer_norm_eps).sqrt().recip();
    let expected = [-2.0 * scale, 0.0, 0.0, 2.0 * scale, 2.0 * scale, 0.0, 0.0, -2.0 * scale];
    for shifts in [&output.node_shifts, &output.child_shifts] {
        let actual = buffer_to_vec::<Cpu, f32>(shifts);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
    assert_eq!(buffer_to_vec::<Cpu, u32>(&output.slot_tokens), vec![0]);
    assert_eq!(buffer_to_vec::<Cpu, u32>(&output.token_slots), vec![0, u32::MAX]);
}
