use std::io::Write;

use half::bf16;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::{common::Context, cpu::Cpu},
    parameters::ParameterLoader,
};

fn add_tensor(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: &str,
    shape: &[u32],
    data_type: &str,
    data: &[u8],
) {
    let begin = payload.len();
    payload.extend_from_slice(data);
    header.insert(
        name.into(),
        json!({
            "dtype": data_type,
            "shape": shape,
            "data_offsets": [begin, payload.len()]
        }),
    );
}

fn dense_microfloat_parameter_file() -> NamedTempFile {
    const ROWS: u32 = 2;
    const COLUMNS: u32 = 32;
    const GROUP_SIZE: u32 = 16;

    let mut header = Map::new();
    header.insert(
        "__metadata__".into(),
        json!({
            "spec": json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": GROUP_SIZE,
                "scale_mode": "mxfp4",
                "layout": "output_input"
            }).to_string()
        }),
    );
    let mut payload = Vec::new();
    let codes: Vec<u8> = (0..ROWS * COLUMNS / 2).map(|index| 0x10 | (index % 8) as u8).collect();
    let scales: Vec<u8> = (0..ROWS * COLUMNS / GROUP_SIZE).map(|index| 126 + (index % 3) as u8).collect();
    let global_scale = bf16::from_f32(0.75).to_le_bytes();
    add_tensor(&mut header, &mut payload, "weights", &[ROWS, COLUMNS / 2], "U8", &codes);
    add_tensor(&mut header, &mut payload, "scales", &[ROWS, COLUMNS / GROUP_SIZE], "U8", &scales);
    add_tensor(&mut header, &mut payload, "global_scale", &[1], "BF16", &global_scale);

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = NamedTempFile::new().expect("create dense MXFP4 fixture");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write safetensors header length");
    file.write_all(&header).expect("write safetensors header");
    file.write_all(&payload).expect("write safetensors payload");
    file
}

#[uzu_test]
fn loads_dense_mxfp4_storage() {
    // Given a safetensors artifact whose metadata and tensors describe dense MXFP4 weights.
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = dense_microfloat_parameter_file();
    let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load dense MXFP4 fixture");
    let tree = loader.tree();
    let spec = tree.metadata::<AnyWeightMatrixSpec>("spec").expect("load dense MXFP4 spec");

    // When the artifact is loaded through ParameterLoader and WeightMatrix.
    let matrix =
        WeightMatrix::load(&tree, spec, Layout::OutputInput, 2, 32, DataType::BF16).expect("load dense MXFP4 matrix");

    // Then the matrix retains its microfloat operand shape and consumes every source tensor.
    let MatmulB::Microfloat {
        metadata,
        ..
    } = matrix.matmul_b()
    else {
        panic!("dense MXFP4 matrix did not preserve its operand format");
    };
    assert_eq!(metadata.rows, 2);
    assert_eq!(metadata.columns, 32);
    tree.assert_all_tensors_validated().expect("validate dense MXFP4 tensors");
}
