use std::{
    fs::{File, OpenOptions, remove_file},
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use half::bf16;
use proc_macros::uzu_test;
use serde_json::{Map, Value, json};

use super::super::{MoeBlock, MoeBlockError};
use crate::{
    backends::common::Encoder,
    config::mlp::mixture_of_experts::MixtureOfExpertsConfig,
    data_type::DataType,
    encodable_block::mlp::Mlp,
    parameters::ParameterLoader,
    tests::helpers::{alloc_allocation_with_data, allocation_prefix_to_vec, create_context, for_each_non_cpu_backend},
};

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct TempFile {
    file: File,
    path: PathBuf,
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

fn bf16_bytes(values: &[bf16]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

/// Minimal Lalamo-style safetensors file exercising the public ParameterTree contract.
fn write_mxfp4_moe_file(mixed_expert_specs: bool) -> TempFile {
    const EXPERTS: usize = 2;
    const MODEL_DIM: usize = 32;
    const HIDDEN_DIM: usize = 32;

    let mut up_biases = vec![bf16::ZERO; EXPERTS * 2 * HIDDEN_DIM];
    let (up_bias_rows, remainder) = up_biases.as_chunks_mut::<{ 2 * HIDDEN_DIM }>();
    assert!(remainder.is_empty());
    for rows in up_bias_rows {
        for up_bias in &mut rows[..HIDDEN_DIM] {
            *up_bias = bf16::ONE;
        }
    }
    let tensors = vec![
        Tensor {
            name: "router.weights.weights",
            dtype: "BF16",
            shape: vec![EXPERTS, MODEL_DIM],
            bytes: bf16_bytes(&[bf16::ZERO; EXPERTS * MODEL_DIM]),
        },
        Tensor {
            name: "router.biases",
            dtype: "BF16",
            shape: vec![EXPERTS],
            bytes: bf16_bytes(&[bf16::ZERO; EXPERTS]),
        },
        Tensor {
            name: "experts.up_projection.weights.weights",
            dtype: "U8",
            shape: vec![EXPERTS, 2 * HIDDEN_DIM, MODEL_DIM / 2],
            bytes: vec![0; EXPERTS * 2 * HIDDEN_DIM * MODEL_DIM / 2],
        },
        Tensor {
            name: "experts.up_projection.weights.scales",
            dtype: "U8",
            shape: vec![EXPERTS, 2 * HIDDEN_DIM, MODEL_DIM / 16],
            bytes: vec![127; EXPERTS * 2 * HIDDEN_DIM * MODEL_DIM / 16],
        },
        Tensor {
            name: "experts.up_projection.weights.global_scale",
            dtype: "BF16",
            shape: vec![EXPERTS],
            bytes: bf16_bytes(&[bf16::ONE; EXPERTS]),
        },
        Tensor {
            name: "experts.up_projection.biases",
            dtype: "BF16",
            shape: vec![EXPERTS, 2 * HIDDEN_DIM],
            bytes: bf16_bytes(&up_biases),
        },
        Tensor {
            name: "experts.down_projection.weights.weights",
            dtype: "U8",
            shape: vec![EXPERTS, MODEL_DIM, HIDDEN_DIM / 2],
            bytes: vec![0; EXPERTS * MODEL_DIM * HIDDEN_DIM / 2],
        },
        Tensor {
            name: "experts.down_projection.weights.scales",
            dtype: "U8",
            shape: vec![EXPERTS, MODEL_DIM, HIDDEN_DIM / 32],
            bytes: vec![127; EXPERTS * MODEL_DIM * HIDDEN_DIM / 32],
        },
        Tensor {
            name: "experts.down_projection.weights.global_scale",
            dtype: "BF16",
            shape: vec![EXPERTS],
            bytes: bf16_bytes(&[bf16::ONE; EXPERTS]),
        },
        Tensor {
            name: "experts.down_projection.biases",
            dtype: "BF16",
            shape: vec![EXPERTS, MODEL_DIM],
            bytes: bf16_bytes(&[bf16::ZERO; EXPERTS * MODEL_DIM]),
        },
    ];

    let full_precision_spec = json!({"type": "FullPrecisionSpec", "layout": "output_input"}).to_string();
    let gate_up_spec = json!({
        "type": "MicrofloatSpec",
        "bits": 4,
        "group_size": 16,
        "scale_mode": "mxfp4",
        "layout": "output_input"
    })
    .to_string();
    let down_spec = json!({
        "type": "MicrofloatSpec",
        "bits": 4,
        "group_size": 32,
        "scale_mode": "mxfp4",
        "layout": "output_input"
    })
    .to_string();
    let down_spec = if mixed_expert_specs {
        full_precision_spec.clone()
    } else {
        down_spec
    };
    let metadata = json!({
        "router.weights.spec": full_precision_spec,
        "experts.up_projection.weights.spec": gate_up_spec,
        "experts.down_projection.weights.spec": down_spec,
    });

    let mut offset = 0;
    let mut header = Map::new();
    header.insert("__metadata__".to_string(), metadata);
    for tensor in &tensors {
        let end = offset + tensor.bytes.len();
        header.insert(
            tensor.name.to_string(),
            json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
    header.resize(header.len().next_multiple_of(8), b' ');
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock").as_nanos();
    let path = std::env::temp_dir().join(format!("uzu-mxfp4-{}-{unique}.safetensors", std::process::id()));
    let mut file = OpenOptions::new().write(true).read(true).create_new(true).open(&path).expect("temporary file");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
    file.write_all(&header).expect("write header");
    for tensor in tensors {
        file.write_all(&tensor.bytes).expect("write tensor");
    }
    file.flush().expect("flush safetensors file");
    TempFile {
        file,
        path,
    }
}

fn lalamo_moe_config() -> MixtureOfExpertsConfig {
    serde_json::from_value(json!({
        "type": "MixtureOfExpertsConfig",
        "expert_config": {
            "type": "DenseMLPConfig",
            "linear_config": {},
            "activation": {"type": "SiLU", "alpha": 1.702},
            "has_up_biases": true,
            "has_down_biases": true,
            "gate_clipping": [null, 7.0],
            "up_clipping": [-6.0, 8.0]
        },
        "router_config": {},
        "routing_function": {"type": "SoftmaxRouting"},
        "num_routed_experts": 2,
        "num_active_routed_experts": 1,
        "router_has_biases": true,
        "num_shared_experts": 0,
        "expert_hidden_dim": 32,
        "gate_config": null
    }))
    .expect("Lalamo MoE config")
}

#[uzu_test]
fn test_moe_block_loads_and_executes_lalamo_mxfp4_contract() {
    for_each_non_cpu_backend!(|B| {
        const MODEL_DIM: usize = 32;
        let context = create_context::<B>();
        let file = write_mxfp4_moe_file(false);
        let loader = ParameterLoader::<B>::new(&file.file, context.as_ref()).expect("parameter loader");
        let tree = loader.tree();
        let config = lalamo_moe_config();

        let block = MoeBlock::new(context.as_ref(), &config, MODEL_DIM, DataType::BF16, &tree).expect("packed MoE");
        tree.assert_all_tensors_validated().expect("all packed tensors consumed");

        // More than one row deliberately exercises the packed path for prefill too.
        let input = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::ONE; 2 * MODEL_DIM]);
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        let output_allocation = block.encode(input, 2, &mut encoder).expect("encode packed MoE");
        let completed = encoder.end_encoding().submit().wait_until_completed().expect("run packed MoE");
        let output = allocation_prefix_to_vec::<B, bf16>(&output_allocation, 2 * MODEL_DIM);

        assert!(output.iter().all(|value| *value == bf16::ZERO));
        drop(output_allocation);
        drop(completed);
    });
}

#[uzu_test]
fn test_moe_block_rejects_more_active_than_routed_experts() {
    for_each_non_cpu_backend!(|B| {
        const MODEL_DIM: usize = 32;
        let context = create_context::<B>();
        let file = write_mxfp4_moe_file(false);
        let loader = ParameterLoader::<B>::new(&file.file, context.as_ref()).expect("parameter loader");
        let tree = loader.tree();
        let mut config = lalamo_moe_config();
        config.num_active_routed_experts = config.num_routed_experts + 1;

        let result = MoeBlock::new(context.as_ref(), &config, MODEL_DIM, DataType::BF16, &tree);

        assert!(matches!(result, Err(MoeBlockError::InvalidActiveExpertCount)));
    });
}

#[uzu_test]
fn test_moe_block_rejects_mixed_expert_specs() {
    for_each_non_cpu_backend!(|B| {
        const MODEL_DIM: usize = 32;
        let context = create_context::<B>();
        let file = write_mxfp4_moe_file(true);
        let loader = ParameterLoader::<B>::new(&file.file, context.as_ref()).expect("parameter loader");
        let tree = loader.tree();
        let config = lalamo_moe_config();

        let result = MoeBlock::new(context.as_ref(), &config, MODEL_DIM, DataType::BF16, &tree);

        assert!(matches!(result, Err(MoeBlockError::UnsupportedExpertConfiguration(_))));
    });
}
