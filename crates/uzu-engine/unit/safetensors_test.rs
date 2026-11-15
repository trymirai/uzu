use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
    safetensors::{ParameterLoader, ParameterLoaderError},
};

struct TensorData {
    dtype: DataType,
    shape: Vec<u32>,
    bytes: Vec<u8>,
}

fn tensor_data(
    dtype: DataType,
    shape: Vec<u32>,
    values: &[u8],
) -> TensorData {
    TensorData {
        dtype,
        shape,
        bytes: values.to_vec(),
    }
}

fn write_safetensors(
    path: &Path,
    metadata: Option<&[(&str, &str)]>,
    tensors: &[(&str, TensorData)],
) {
    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        let mut metadata_json = serde_json::Map::new();
        for (key, value) in metadata {
            metadata_json.insert(key.to_string(), serde_json::Value::String(value.to_string()));
        }
        header.insert("__metadata__".to_string(), metadata_json.into());
    }
    let mut offset = 0u64;
    let mut data = Vec::new();
    for (name, tensor) in tensors {
        let end = offset + tensor.bytes.len() as u64;
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
        data.extend_from_slice(&tensor.bytes);
    }
    let mut header_json = serde_json::to_vec(&header).unwrap();
    if header_json.len() % 8 != 0 {
        header_json.extend(std::iter::repeat_n(b' ', 8 - header_json.len() % 8));
    }
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header_json.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    file.write_all(&data).unwrap();
}

fn temp_model_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("uzu-sharded-safetensors-{}-{name}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

#[uzu_test]
fn sharded_loading() {
    let directory = temp_model_dir("sharded");
    let shard_a = directory.join("model-00001-of-00002.safetensors");
    let shard_b = directory.join("model-00002-of-00002.safetensors");
    let weight_a = 7.0f32.to_le_bytes();
    let weights_b = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
    write_safetensors(
        &shard_a,
        Some(&[("model.a", r#""a""#)]),
        &[("model.a", tensor_data(DataType::F32, vec![1], &weight_a))],
    );
    write_safetensors(
        &shard_b,
        Some(&[("model.b", r#""b""#)]),
        &[("model.b", tensor_data(DataType::F32, vec![2], &weights_b))],
    );
    fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": {
                "model.a": "model-00001-of-00002.safetensors",
                "model.b": "model-00002-of-00002.safetensors",
            },
        })
        .to_string(),
    )
    .unwrap();

    block_on(async {
        let context = <Cpu as Backend>::Context::new().await.expect("context");
        let loader = ParameterLoader::<Cpu>::new(directory.join("model.safetensors"), &context).await.unwrap();
        let tree = loader.tree().subtree("model");

        let a = tree.read_slice::<f32>("a", &[1]).await.unwrap();
        assert_eq!(a.as_ref(), &[7.0]);
        let b = tree.read_slice::<f32>("b", &[2]).await.unwrap();
        assert_eq!(b.as_ref(), &[1.0, 2.0]);

        let prefix: String = tree.metadata("a").unwrap();
        assert_eq!(prefix, "a");
        let prefix: String = tree.metadata("b").unwrap();
        assert_eq!(prefix, "b");
        assert!(matches!(tree.metadata::<String>("c"), Err(ParameterLoaderError::KeyNotFound(_))));

        loader.assert_all_tensors_read().unwrap();

        let buffer = tree.read_buffer("a", &[1], DataType::F32).await.unwrap();
        drop(buffer);
    });

    fs::remove_dir_all(&directory).unwrap();
}

#[uzu_test]
fn sharded_unread_tensors_error() {
    let directory = temp_model_dir("unread");
    let shard_a = directory.join("model-00001-of-00002.safetensors");
    let shard_b = directory.join("model-00002-of-00002.safetensors");
    write_safetensors(&shard_a, None, &[("model.a", tensor_data(DataType::F32, vec![1], &[0; 4]))]);
    write_safetensors(&shard_b, None, &[("model.b", tensor_data(DataType::F32, vec![1], &[0; 4]))]);
    fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 8 },
            "weight_map": {
                "model.a": "model-00001-of-00002.safetensors",
                "model.b": "model-00002-of-00002.safetensors",
            },
        })
        .to_string(),
    )
    .unwrap();

    block_on(async {
        let context = <Cpu as Backend>::Context::new().await.expect("context");
        let loader = ParameterLoader::<Cpu>::new(directory.join("model.safetensors"), &context).await.unwrap();
        let tree = loader.tree().subtree("model");
        tree.read_slice::<f32>("a", &[1]).await.unwrap();

        assert!(matches!(loader.assert_all_tensors_read(), Err(ParameterLoaderError::UnreadTensors(_))));
        tree.ignore("b", &[1], DataType::F32).unwrap();
        loader.assert_all_tensors_read().unwrap();

        assert!(matches!(tree.read_slice::<f32>("missing", &[1]).await, Err(ParameterLoaderError::KeyNotFound(_)),));
    });

    fs::remove_dir_all(&directory).unwrap();
}

#[uzu_test]
fn single_file_loading() {
    let directory = temp_model_dir("single");
    let path = directory.join("model.safetensors");
    let weight = 5.0f32.to_le_bytes();
    write_safetensors(
        &path,
        Some(&[("format", r#""pt""#)]),
        &[("model.w", tensor_data(DataType::F32, vec![1], &weight))],
    );

    block_on(async {
        let context = <Cpu as Backend>::Context::new().await.expect("context");
        let loader = ParameterLoader::<Cpu>::new(&path, &context).await.unwrap();
        let tree = loader.tree();
        let w = tree.read_slice::<f32>("model.w", &[1]).await.unwrap();
        assert_eq!(w.as_ref(), &[5.0]);
        let format: String = tree.metadata("format").unwrap();
        assert_eq!(format, "pt");
        loader.assert_all_tensors_read().unwrap();
    });

    fs::remove_dir_all(&directory).unwrap();
}
