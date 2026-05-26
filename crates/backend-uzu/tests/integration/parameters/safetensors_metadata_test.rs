use std::fs::File;

use backend_uzu::{
    ArrayContextExt, DataType, ParameterLoader, SafeTensorData,
    backends::{
        common::{Backend, Context, allocation_as_bytes},
        cpu::Cpu,
    },
    read_safetensors_metadata, write_safetensors,
};

use crate::common::path::get_test_weights_path;

fn tensor_from_array<B: Backend>(
    name: &str,
    array: &backend_uzu::Array<B>,
) -> SafeTensorData {
    SafeTensorData {
        name: name.to_string(),
        shape: array.shape().into(),
        data_type: array.data_type(),
        data: allocation_as_bytes(array.allocation()).into(),
    }
}

#[test]
fn test_safetensors_metadata_writer_rejects_empty_tensors() {
    let mut bytes = Vec::new();
    let err = write_safetensors(&mut bytes, &[]).expect_err("empty safetensors should fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_safetensors_metadata_writer_rejects_shape_data_mismatch() {
    let mut bytes = Vec::new();
    let err = write_safetensors(
        &mut bytes,
        &[SafeTensorData {
            name: "bad".to_string(),
            shape: [2].into(),
            data_type: DataType::F32,
            data: vec![0; 4].into_boxed_slice(),
        }],
    )
    .expect_err("shape mismatch should fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_safetensors_metadata(&file).expect("read metadata");
    assert!(metadata.tensors.len() > 0);
}

#[test]
fn test_safetensors_metadata_writer_roundtrips_arrays() {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let floats = context.create_array_from(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let ints = context.create_array_from(&[2], &[7i32, 9]);
    let mut file = tempfile::NamedTempFile::new().expect("create safetensors file");

    write_safetensors(file.as_file_mut(), &[tensor_from_array("floats", &floats), tensor_from_array("ints", &ints)])
        .expect("write safetensors file");

    let loader_file = file.reopen().expect("open safetensors file");
    let loader = ParameterLoader::new(&loader_file, context.as_ref()).expect("create loader");
    assert_eq!(loader.tree().leaf_array("floats").unwrap().as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(loader.tree().leaf_array("ints").unwrap().as_slice::<i32>(), &[7, 9]);
}
