#![allow(dead_code)]

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::kernel::matmul::MatmulArguments,
        metal::{Metal, MetalContext},
    },
};

#[derive(Clone)]
pub struct DtypeCombo {
    pub a_dtype: DataType,
    pub b_dtype: DataType,
    pub output_dtype: DataType,
}

impl std::fmt::Display for DtypeCombo {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{:?}*{:?}->{:?}", self.a_dtype, self.b_dtype, self.output_dtype)
    }
}

#[derive(Clone)]
pub struct TestShape {
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl std::fmt::Display for TestShape {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.batch, self.input_dim, self.output_dim)
    }
}

#[derive(Clone, Copy)]
pub enum MatmulVariant {
    Gemv,
    GemmMpp,
    Gemm,
}

pub fn test_combos() -> Vec<DtypeCombo> {
    vec![
        DtypeCombo {
            a_dtype: DataType::BF16,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
        DtypeCombo {
            a_dtype: DataType::F16,
            b_dtype: DataType::F16,
            output_dtype: DataType::F16,
        },
    ]
}

pub fn applicable_variants(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
) -> Vec<(&'static str, MatmulVariant)> {
    let mut variants = Vec::new();

    if context.device_capabilities().supports_mxu {
        variants.push(("GemmMpp", MatmulVariant::GemmMpp));
    }

    variants.push(("Gemm", MatmulVariant::Gemm));

    let is_same_type = combo.a_dtype == combo.b_dtype && combo.b_dtype == combo.output_dtype;
    if is_same_type && matches!(combo.output_dtype, DataType::F16 | DataType::BF16) {
        let m = shape.batch;
        let n = shape.output_dim;
        let gemv_eligible = if n == 1 { m == 1 } else { true };
        if gemv_eligible {
            variants.push(("Gemv", MatmulVariant::Gemv));
        }
    }

    variants
}

pub fn make_full_precision_arguments<'a>(
    a_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    b_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    output_buffer: &'a mut Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: &TestShape,
) -> MatmulArguments<'a, Metal> {
    MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        output: output_buffer,
        bias: None,
        batch: shape.batch,
        input_dim: shape.input_dim,
        output_dim: shape.output_dim,
    }
}

pub fn write_json_results<T: Serialize>(
    test_name: &str,
    device: &str,
    results: &[T],
) {
    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join(format!("{test_name}.json"));
        let wrapper = serde_json::json!({ "device": device, "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}
