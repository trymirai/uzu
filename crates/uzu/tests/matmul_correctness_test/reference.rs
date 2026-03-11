use half::{bf16, f16};
use metal::MTLBuffer;
use ndarray::Array2;
use objc2::runtime::ProtocolObject;
use uzu::DataType;

use super::common::matmul::{DtypeCombo, TestShape};

pub fn generate_typed_data(
    dtype: DataType,
    count: usize,
    modulus: usize,
    offset: i64,
) -> Vec<u8> {
    match dtype {
        DataType::I8 => {
            let data: Vec<i8> = (0..count).map(|i| ((i % modulus) as i8).wrapping_add(offset as i8)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::BF16 => {
            let data: Vec<bf16> =
                (0..count).map(|i| bf16::from_f32(((i % modulus) as f32) * 0.01 + offset as f32 * 0.01)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::F16 => {
            let data: Vec<f16> =
                (0..count).map(|i| f16::from_f32(((i % modulus) as f32) * 0.01 + offset as f32 * 0.01)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::F32 => {
            let data: Vec<f32> = (0..count).map(|i| ((i % modulus) as f32) * 0.01 + offset as f32 * 0.01).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        other => panic!("Unsupported dtype for data generation: {other:?}"),
    }
}

fn bytes_to_f64(
    dtype: DataType,
    bytes: &[u8],
) -> Vec<f64> {
    match dtype {
        DataType::I8 => bytemuck::cast_slice::<u8, i8>(bytes).iter().map(|&x| x as f64).collect(),
        DataType::BF16 => bytemuck::cast_slice::<u8, bf16>(bytes).iter().map(|x| x.to_f64()).collect(),
        DataType::F16 => bytemuck::cast_slice::<u8, f16>(bytes).iter().map(|x| x.to_f64()).collect(),
        DataType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).iter().map(|&x| x as f64).collect(),
        other => panic!("Unsupported dtype for conversion: {other:?}"),
    }
}

pub fn output_to_f64(
    output_dtype: DataType,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Vec<f64> {
    unsafe {
        let pointer = buffer.contents().as_ptr();
        match output_dtype {
            DataType::I32 => {
                let slice = std::slice::from_raw_parts(pointer as *const i32, count);
                slice.iter().map(|&x| x as f64).collect()
            },
            DataType::F32 => {
                let slice = std::slice::from_raw_parts(pointer as *const f32, count);
                slice.iter().map(|&x| x as f64).collect()
            },
            DataType::F16 => {
                let slice = std::slice::from_raw_parts(pointer as *const f16, count);
                slice.iter().map(|x| x.to_f64()).collect()
            },
            DataType::BF16 => {
                let slice = std::slice::from_raw_parts(pointer as *const bf16, count);
                slice.iter().map(|x| x.to_f64()).collect()
            },
            other => panic!("Unsupported output_dtype: {other:?}"),
        }
    }
}

pub fn ndarray_reference(
    combo: &DtypeCombo,
    a_bytes: &[u8],
    b_bytes: &[u8],
    shape: &TestShape,
) -> Vec<f64> {
    let a_f64 = bytes_to_f64(combo.a_dtype, a_bytes);
    let b_f64 = bytes_to_f64(combo.b_dtype, b_bytes);

    let a_array = Array2::from_shape_vec((shape.batch, shape.input_dim), a_f64).expect("A shape");
    let b_array = Array2::from_shape_vec((shape.output_dim, shape.input_dim), b_f64).expect("B shape");
    let result = a_array.dot(&b_array.t());

    match combo.output_dtype {
        DataType::I32 => result.iter().map(|&x| (x as i32) as f64).collect(),
        DataType::F32 => result.iter().map(|&x| (x as f32) as f64).collect(),
        DataType::F16 => result.iter().map(|&x| f16::from_f64(x).to_f64()).collect(),
        DataType::BF16 => result.iter().map(|&x| bf16::from_f64(x).to_f64()).collect(),
        _ => result.iter().copied().collect(),
    }
}

pub fn tolerance_for(
    combo: &DtypeCombo,
    shape: &TestShape,
) -> f64 {
    match (combo.a_dtype, combo.b_dtype, combo.output_dtype) {
        (DataType::I8, DataType::I8, DataType::I32) => 0.0,
        (DataType::I8, DataType::BF16, DataType::BF16) => 0.05 * (shape.input_dim as f64 / 1024.0).sqrt(),
        (DataType::BF16, DataType::BF16, DataType::BF16) => {
            let base = 0.01 * (shape.input_dim as f64 / 1024.0).sqrt();
            base * (1.0 + (shape.batch as f64).ln() / std::f64::consts::LN_2 * 0.02)
        },
        _ => 0.01,
    }
}
