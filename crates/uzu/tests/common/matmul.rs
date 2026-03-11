#![allow(dead_code)]

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::kernel::matmul::{
            MatmulArguments, MatmulDispatchDescriptor, gemm_mpp,
            gemv::{
                DispatchDescriptor as GemvDescriptor, OutputSource as GemvOutputSource,
                Specialization as GemvSpecialization,
            },
        },
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

pub fn test_combos() -> Vec<DtypeCombo> {
    vec![
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::I8,
            output_dtype: DataType::I32,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::F16,
            output_dtype: DataType::F16,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::F32,
            output_dtype: DataType::F32,
        },
        DtypeCombo {
            a_dtype: DataType::BF16,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
    ]
}

pub fn try_all_descriptors(
    context: &MetalContext,
    combo: &DtypeCombo,
    arguments: &MatmulArguments<Metal>,
) -> Vec<(&'static str, MatmulDispatchDescriptor)> {
    let mut descriptors = Vec::new();

    if let Ok(descriptor) = gemm_mpp::DispatchDescriptor::new(combo.output_dtype, arguments) {
        descriptors.push(("GemmMpp", MatmulDispatchDescriptor::GemmMpp(descriptor)));
    }

    let is_same_type = combo.a_dtype == combo.b_dtype && combo.b_dtype == combo.output_dtype;
    if is_same_type && matches!(combo.output_dtype, DataType::F16 | DataType::BF16) {
        if let Some(descriptor) = force_gemv_descriptor(combo.output_dtype, arguments) {
            descriptors.push(("Gemv", MatmulDispatchDescriptor::Gemv(descriptor)));
        }
    }

    descriptors
}

fn force_gemv_descriptor(
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> Option<GemvDescriptor> {
    if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
        return None;
    }
    if !arguments.transpose_b {
        return None;
    }

    let n = arguments.output_dim;
    if n == 1 && arguments.batch != 1 {
        return None;
    }

    let matrix_is_rhs = n != 1;
    let transpose_matrix = if matrix_is_rhs {
        !arguments.transpose_b
    } else {
        false
    };

    let output_source = if arguments.bias.is_some() {
        GemvOutputSource::Bias
    } else {
        GemvOutputSource::None
    };

    let (apply_output_scale_and_accumulate, alpha, _beta, bias_stride) = match output_source {
        GemvOutputSource::None => (false, 1.0f32, 0.0f32, 0),
        GemvOutputSource::Bias => (true, 1.0f32, 1.0f32, 1),
    };

    let output_dimension = if matrix_is_rhs {
        arguments.output_dim
    } else {
        arguments.batch
    };

    let specialization = GemvSpecialization::select(
        transpose_matrix,
        arguments.input_dim,
        output_dimension,
        apply_output_scale_and_accumulate,
    );

    let input_dimension = arguments.input_dim;
    let matrix_leading_dim = if matrix_is_rhs {
        arguments.leading_dim_b
    } else {
        arguments.leading_dim_a
    };

    let batch_shape = [if arguments.batch_count > 1 {
        arguments.batch_count
    } else {
        1
    }];

    let elements_per_matrix_a = (arguments.batch as i64) * (arguments.leading_dim_a as i64);
    let elements_per_matrix_b = if arguments.transpose_b {
        (arguments.output_dim as i64) * (arguments.leading_dim_b as i64)
    } else {
        (arguments.input_dim as i64) * (arguments.leading_dim_b as i64)
    };

    let vector_batch_stride = [if matrix_is_rhs {
        elements_per_matrix_a
    } else {
        elements_per_matrix_b
    }];
    let matrix_batch_stride = [if matrix_is_rhs {
        elements_per_matrix_b
    } else {
        elements_per_matrix_a
    }];
    let bias_batch_stride = [if arguments.batch_count > 1 {
        (output_dimension as i64) * (arguments.leading_dim_d as i64)
    } else {
        0
    }];

    Some(GemvDescriptor {
        specialization,
        matrix_is_rhs,
        output_source,
        input_dimension,
        output_dimension,
        matrix_leading_dim,
        alpha,
        beta: if arguments.bias.is_some() {
            1.0
        } else {
            0.0
        },
        batch_shape,
        vector_batch_stride,
        matrix_batch_stride,
        bias_batch_stride,
        bias_stride,
        batch_rows: arguments.batch,
    })
}

pub fn make_arguments<'a>(
    a_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    b_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    d_buffer: &'a mut Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: &TestShape,
) -> MatmulArguments<'a, Metal> {
    MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        d: d_buffer,
        bias: None,
        batch: shape.batch as i32,
        input_dim: shape.input_dim as i32,
        output_dim: shape.output_dim as i32,
        leading_dim_a: shape.input_dim as i32,
        leading_dim_b: shape.input_dim as i32,
        leading_dim_d: shape.output_dim as i32,
        batch_count: 1,
        transpose_b: true,
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
