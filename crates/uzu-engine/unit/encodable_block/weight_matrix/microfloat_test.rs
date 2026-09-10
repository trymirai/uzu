use std::{fmt::Debug, fs::File, io::Write};

use half::{bf16, f16};
use serde_json::{Map, Value, json};
use tempfile::tempfile;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Encoder},
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearMatmulError},
        weight_matrix::WeightMatrixError,
    },
    parameters::{ParameterLoader, ParameterLoaderError},
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, create_context, for_each_non_cpu_backend},
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

fn dense_microfloat_parameter_file<T: ArrayElement>(edit_header: impl FnOnce(&mut Map<String, Value>)) -> File {
    const ROWS: u32 = 2;
    const COLUMNS: u32 = 32;
    const GROUP_SIZE: u32 = 16;

    let mut header = Map::new();
    header.insert(
        "__metadata__".into(),
        json!({
            "weights.spec": json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": GROUP_SIZE,
                "scale_mode": "mxfp4",
                "layout": "output_input"
            }).to_string()
        }),
    );
    let mut payload = Vec::new();
    let codes: Vec<u8> = [0x21, 0xbc, 0x76, 0x09].into_iter().flat_map(|code| [code; 8]).collect();
    let scales = [126, 128, 127, 125];
    let global_scale = T::from(0.75).expect("representable outer scale");
    let biases = [T::from(1.5).unwrap(), T::from(-0.75).unwrap()];
    let data_type = match T::data_type() {
        DataType::F16 => "F16",
        DataType::BF16 => "BF16",
        DataType::F32 => "F32",
        _ => unreachable!(),
    };
    add_tensor(&mut header, &mut payload, "weights.weights", &[ROWS, COLUMNS / 2], "U8", &codes);
    add_tensor(&mut header, &mut payload, "weights.scales", &[ROWS, COLUMNS / GROUP_SIZE], "U8", &scales);
    add_tensor(&mut header, &mut payload, "weights.global_scale", &[1], data_type, bytemuck::bytes_of(&global_scale));
    add_tensor(&mut header, &mut payload, "biases", &[ROWS], data_type, bytemuck::cast_slice(&biases));

    edit_header(&mut header);
    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = tempfile().expect("create dense MXFP4 fixture");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write safetensors header length");
    file.write_all(&header).expect("write safetensors header");
    file.write_all(&payload).expect("write safetensors payload");
    file
}

fn execute_loaded_projection<B: Backend, T: ArrayElement + PartialEq + Debug>(batch_dim: u32) {
    let context = create_context::<B>();
    let file = dense_microfloat_parameter_file::<T>(|_| {});
    let loader = ParameterLoader::<B>::new(&file, context.as_ref()).expect("load dense MXFP4 fixture");
    let tree = loader.tree();
    let projection = <dyn Linear<B>>::new(32, [2], true, context.as_ref(), T::data_type(), &tree)
        .expect("load dense MXFP4 projection");
    tree.assert_all_tensors_validated().expect("validate dense MXFP4 tensors");

    let input: Vec<T> = (0..batch_dim)
        .flat_map(|row| {
            (0..32).map(move |column| {
                let value = match (row % 2, column) {
                    (1, 16..) => 0.5,
                    (1, column) if column % 2 == 1 => -1.0,
                    _ => 1.0,
                };
                T::from(value).expect("representable input")
            })
        })
        .collect();
    let input = alloc_allocation_with_data::<B, T>(context.as_ref(), &input);
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    let output = projection.encode(input, batch_dim, &mut encoder).expect("encode MXFP4 projection");
    let command_buffer = encoder.end_encoding();
    let command_buffer = command_buffer.submit();
    let completed = command_buffer.wait_until_completed().expect("execute MXFP4 projection");
    let values = allocation_to_vec::<B, T>(&output);
    // Releasing scratch before the completion handle returns its allocation pool.
    drop(output);
    drop(completed);
    let expected: Vec<T> = (0..batch_dim)
        .flat_map(|row| {
            if row % 2 == 0 {
                [-36.0, 58.5]
            } else {
                [-21.0, -13.125]
            }
        })
        .map(|value| T::from(value).expect("representable expected output"))
        .collect();
    assert_eq!(values, expected, "{} {:?} batch={batch_dim}", std::any::type_name::<B>(), T::data_type());
}

#[uzu_test]
fn loads_and_executes_dense_mxfp4_projection() {
    for batch_dim in [1, 9] {
        execute_loaded_projection::<Cpu, bf16>(batch_dim);
        execute_loaded_projection::<Cpu, f16>(batch_dim);
        execute_loaded_projection::<Cpu, f32>(batch_dim);
        for_each_non_cpu_backend!(|B| {
            execute_loaded_projection::<B, bf16>(batch_dim);
            execute_loaded_projection::<B, f16>(batch_dim);
        });
    }
}

#[uzu_test]
fn rejects_invalid_mxfp4_artifact_tensors() {
    let context = create_context::<Cpu>();
    for (tensor, field, value) in [
        ("weights.weights", "dtype", json!("I8")),
        ("weights.scales", "shape", json!([4])),
        ("weights.global_scale", "data_offsets", json!([36, 38])),
    ] {
        let file = dense_microfloat_parameter_file::<f32>(|header| {
            header[tensor][field] = value;
        });
        let loader = ParameterLoader::<Cpu>::new(&file, context.as_ref()).expect("read artifact header");
        let tree = loader.tree();
        let result = <dyn Linear<Cpu>>::new(32, [2], true, context.as_ref(), DataType::F32, &tree);
        let error = result.err().expect("reject invalid MXFP4 tensor before projection encoding");
        let LinearBlockError::LinearMatmulError(LinearMatmulError::WeightMatrix(WeightMatrixError::ParameterError(
            error,
        ))) = error
        else {
            panic!("unexpected projection error: {error}");
        };
        match (field, error) {
            (
                "dtype",
                ParameterLoaderError::InvalidTensor {
                    data_type: DataType::I8,
                    expected_data_type: DataType::U8,
                    ..
                },
            ) => {},
            (
                "shape",
                ParameterLoaderError::InvalidTensor {
                    shape,
                    expected_shape,
                    ..
                },
            ) => {
                assert_eq!(&*shape, &[4]);
                assert_eq!(&*expected_shape, &[2, 2]);
            },
            (
                "data_offsets",
                ParameterLoaderError::InvalidTensorSize {
                    size: 2,
                    expected_size: 4,
                    ..
                },
            ) => {},
            (_, error) => panic!("unexpected error for {tensor}: {error}"),
        }
    }
}
