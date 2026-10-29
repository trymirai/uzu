use std::{fs, path::PathBuf};

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use super::{WeightMatrix, WeightMatrixError};
use crate::{
    backends::{
        common::{Buffer, kernel::matmul::MatmulB},
        metal::Metal,
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    safetensors::ParameterLoader,
};

fn write_symrow2_file(
    name: &str,
    n: usize,
    k: usize,
    scale_dtype: &str,
) -> PathBuf {
    let (weight_bytes, scale_bytes) = (n * k / 4, n * 2);
    let header = format!(
        r#"{{"weights":{{"dtype":"U8","shape":[{n},{}],"data_offsets":[0,{weight_bytes}]}},"scales":{{"dtype":"{scale_dtype}","shape":[{n}],"data_offsets":[{weight_bytes},{}]}}}}"#,
        k / 4,
        weight_bytes + scale_bytes,
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.resize(bytes.len() + weight_bytes + scale_bytes, 0);
    let path = std::env::temp_dir().join(format!("uzu_{name}_{}.safetensors", std::process::id()));
    fs::write(&path, bytes).unwrap();
    path
}

fn load_symrow2(
    name: &str,
    n: u32,
    k: u32,
    scale_dtype: &str,
) -> Result<WeightMatrix<Metal>, WeightMatrixError<Metal>> {
    let context = crate::tests::util::shared_metal_context();
    let path = write_symrow2_file(name, n as usize, k as usize, scale_dtype);
    let result = block_on(async {
        let loader = ParameterLoader::<Metal>::new(&path, &*context).await.unwrap();
        WeightMatrix::load(
            n,
            k,
            DataType::BF16,
            serde_json::from_str::<AnyWeightMatrixSpec>(r#"{"type":"SymRow2Spec","layout":"output_input"}"#).unwrap(),
            Layout::OutputInput,
            &loader.tree(),
        )
        .await
    });
    fs::remove_file(path).unwrap();
    result
}

#[uzu_test]
fn symrow2_loads_bf16_row_scales_and_rejects_the_rest() {
    let matrix = load_symrow2("symrow2_ok", 32, 256, "BF16").unwrap();
    let MatmulB::SymRow2 {
        codes,
        row_scales,
    } = matrix.matmul_b()
    else {
        panic!("expected a SymRow2 matmul B");
    };
    assert_eq!((Buffer::size(codes), Buffer::size(row_scales)), (32 * 64, 64));
    assert!(load_symrow2("symrow2_fp16", 32, 256, "F16").is_err());
    assert!(load_symrow2("symrow2_n", 24, 256, "BF16").is_err());
    assert!(load_symrow2("symrow2_k", 32, 192, "BF16").is_err());
}
