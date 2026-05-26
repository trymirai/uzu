use std::{borrow::Cow, collections::BTreeMap, fs::File};

use backend_uzu::{
    array::{Array, ArrayContextExt},
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
    parameters::{
        ParameterLoader, SafeTensorData, read_safetensors_metadata, write_safetensors, write_safetensors_with_metadata,
    },
};

use crate::common::path::get_test_weights_path;

fn tensor_from_array<'data, B: Backend>(
    name: &str,
    array: &'data Array<B>,
) -> SafeTensorData<'data> {
    SafeTensorData {
        name: name.to_string(),
        shape: array.shape().into(),
        data_type: array.data_type(),
        data: Cow::Borrowed(array.as_bytes()),
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
            data: Cow::Owned(vec![0; 4]),
        }],
    )
    .expect_err("shape mismatch should fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_safetensors_metadata_writer_rejects_duplicate_names() {
    let mut bytes = Vec::new();
    let tensor = SafeTensorData {
        name: "duplicate".to_string(),
        shape: [1].into(),
        data_type: DataType::I32,
        data: Cow::Owned(vec![0; 4]),
    };
    let err = write_safetensors(&mut bytes, &[tensor.clone(), tensor]).expect_err("duplicate names should fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_safetensors_metadata_writer_rejects_metadata_tensor_name() {
    let mut bytes = Vec::new();
    let err = write_safetensors(
        &mut bytes,
        &[SafeTensorData {
            name: "__metadata__".to_string(),
            shape: [1].into(),
            data_type: DataType::I32,
            data: Cow::Owned(vec![0; 4]),
        }],
    )
    .expect_err("__metadata__ tensor name should fail");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_safetensors_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}

#[test]
fn test_safetensors_metadata_writer_roundtrips_arrays() {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let floats = context.create_array_from(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let doubles = context.create_array_from(&[2], &[1.25f64, -2.5]);
    let i16s = context.create_array_from(&[2], &[7i16, -9]);
    let u16s = context.create_array_from(&[2], &[7u16, 9]);
    let ints = context.create_array_from(&[2], &[7i32, 9]);
    let mut file = tempfile::NamedTempFile::new().expect("create safetensors file");

    write_safetensors(
        file.as_file_mut(),
        &[
            tensor_from_array("floats", &floats),
            tensor_from_array("doubles", &doubles),
            tensor_from_array("i16s", &i16s),
            tensor_from_array("u16s", &u16s),
            tensor_from_array("ints", &ints),
        ],
    )
    .expect("write safetensors file");

    let loader_file = file.reopen().expect("open safetensors file");
    let loader = ParameterLoader::<Cpu>::new(&loader_file, context.as_ref()).expect("create loader");
    let floats = loader.tree().leaf("floats").unwrap().validate(&[2, 2], DataType::F32).unwrap().read_array().unwrap();
    let doubles = loader.tree().leaf("doubles").unwrap().validate(&[2], DataType::F64).unwrap().read_array().unwrap();
    let i16s = loader.tree().leaf("i16s").unwrap().validate(&[2], DataType::I16).unwrap().read_array().unwrap();
    let u16s = loader.tree().leaf("u16s").unwrap().validate(&[2], DataType::U16).unwrap().read_array().unwrap();
    let ints = loader.tree().leaf("ints").unwrap().validate(&[2], DataType::I32).unwrap().read_array().unwrap();
    assert_eq!(floats.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(doubles.as_slice::<f64>(), &[1.25, -2.5]);
    assert_eq!(i16s.as_slice::<i16>(), &[7, -9]);
    assert_eq!(u16s.as_slice::<u16>(), &[7, 9]);
    assert_eq!(ints.as_slice::<i32>(), &[7, 9]);
}

#[test]
fn test_safetensors_metadata_writer_roundtrips_metadata() {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let ints = context.create_array_from(&[2], &[7i32, 9]);
    let mut metadata = BTreeMap::new();
    metadata.insert("rendered_request".to_string(), "trace".to_string());
    let mut file = tempfile::NamedTempFile::new().expect("create safetensors file");

    write_safetensors_with_metadata(file.as_file_mut(), &[tensor_from_array("ints", &ints)], Some(&metadata))
        .expect("write safetensors file");

    let loader_file = file.reopen().expect("open safetensors file");
    let (_offset, loaded_metadata) = read_safetensors_metadata(&loader_file).expect("read metadata");
    assert_eq!(loaded_metadata.metadata.unwrap()["rendered_request"], "trace");
}
