use std::{error::Error, fmt::Display, num::NonZeroU32};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{
                GatheredMatmul, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulMatrixMap,
                MatmulRouting, MatmulRowMap,
            },
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

struct Case<T> {
    a: Vec<T>,
    a_offset: usize,
    b: Vec<T>,
    row_indices: Option<Vec<u32>>,
    source_divisor: Option<u32>,
    destination_divisor: Option<u32>,
    offsets: Option<Vec<u32>>,
    matrix_scales: Option<Vec<T>>,
    matrix_biases: Option<Vec<T>>,
    m: usize,
    n: usize,
    k: usize,
    matrix_count: usize,
    b_layout: BLayout,
    epilogue: Epilogue<T>,
}

#[derive(Clone, Copy)]
struct BLayout {
    transpose: bool,
    leading_dimension: Option<usize>,
}

impl Default for BLayout {
    fn default() -> Self {
        Self {
            transpose: true,
            leading_dimension: None,
        }
    }
}

struct Epilogue<T> {
    initial_d: Option<Vec<T>>,
    ab_scale: f32,
    accumulate: bool,
    shared_bias: Option<Vec<T>>,
    soft_cap: Option<f32>,
}

impl<T> Default for Epilogue<T> {
    fn default() -> Self {
        Self {
            initial_d: None,
            ab_scale: 1.0,
            accumulate: false,
            shared_bias: None,
            soft_cap: None,
        }
    }
}

fn scalar_reference<T: ArrayElement + Float>(case: &Case<T>) -> Vec<T> {
    let mut output = case.epilogue.initial_d.clone().unwrap_or_else(|| vec![T::zero(); case.m * case.n]);
    assert!(case.a_offset.is_multiple_of(std::mem::size_of::<T>()));
    let a_offset = case.a_offset / std::mem::size_of::<T>();
    let b_leading_dimension = case.b_layout.leading_dimension.unwrap_or(if case.b_layout.transpose {
        case.k
    } else {
        case.n
    });
    let b_matrix_stride = if case.b_layout.transpose {
        case.n * b_leading_dimension
    } else {
        case.k * b_leading_dimension
    };

    for assignment in 0..case.m {
        let source = match (&case.row_indices, case.source_divisor) {
            (Some(indices), Some(divisor)) => indices[assignment] as usize / divisor as usize,
            (_, None) => assignment,
            (None, Some(_)) => panic!("source row map requires indices"),
        };
        let destination = match (&case.row_indices, case.destination_divisor) {
            (Some(indices), Some(divisor)) => indices[assignment] as usize / divisor as usize,
            (_, None) => assignment,
            (None, Some(_)) => panic!("destination row map requires indices"),
        };
        let matrix = match &case.offsets {
            Some(offsets) => (0..case.matrix_count)
                .find(|&matrix| assignment < offsets[matrix + 1] as usize)
                .expect("every assignment must belong to a matrix segment"),
            None => 0,
        };

        for column in 0..case.n {
            let mut dot = 0.0f32;
            for inner in 0..case.k {
                let a = case.a[a_offset + source * case.k + inner].to_f32().unwrap();
                let b_index = matrix * b_matrix_stride
                    + if case.b_layout.transpose {
                        column * b_leading_dimension + inner
                    } else {
                        inner * b_leading_dimension + column
                    };
                let b = case.b[b_index].to_f32().unwrap();
                dot += a * b;
            }

            let scale = case.matrix_scales.as_ref().map(|scales| scales[matrix].to_f32().unwrap()).unwrap_or(1.0);
            let matrix_bias = case
                .matrix_biases
                .as_ref()
                .map(|biases| biases[matrix * case.n + column].to_f32().unwrap())
                .unwrap_or(0.0);
            let output_index = destination * case.n + column;
            let mut value = case.epilogue.ab_scale * scale * dot;
            if case.epilogue.accumulate {
                value += output[output_index].to_f32().unwrap();
            }
            if let Some(shared_bias) = &case.epilogue.shared_bias {
                value += shared_bias[column].to_f32().unwrap();
            }
            value += matrix_bias;
            if let Some(cap) = case.epilogue.soft_cap {
                value = cap * (value / cap).tanh();
            }
            output[output_index] = T::from(value).unwrap();
        }
    }

    output
}

fn try_run_cpu<T: ArrayElement + Float>(case: &Case<T>) -> Result<Vec<T>, Box<dyn Error>> {
    let context = <Cpu as Backend>::Context::new()?;
    let a = alloc_allocation_with_data::<Cpu, T>(&context, &case.a);
    let b = alloc_allocation_with_data::<Cpu, T>(&context, &case.b);
    let row_indices =
        case.row_indices.as_ref().map(|indices| alloc_allocation_with_data::<Cpu, u32>(&context, indices));
    let offsets = case.offsets.as_ref().map(|offsets| alloc_allocation_with_data::<Cpu, u32>(&context, offsets));
    let matrix_scales =
        case.matrix_scales.as_ref().map(|scales| alloc_allocation_with_data::<Cpu, T>(&context, scales));
    let matrix_biases =
        case.matrix_biases.as_ref().map(|biases| alloc_allocation_with_data::<Cpu, T>(&context, biases));
    let shared_bias =
        case.epilogue.shared_bias.as_ref().map(|bias| alloc_allocation_with_data::<Cpu, T>(&context, bias));
    let mut d = match &case.epilogue.initial_d {
        Some(initial_d) => alloc_allocation_with_data::<Cpu, T>(&context, initial_d),
        None => alloc_allocation::<Cpu, T>(&context, case.m * case.n),
    };

    let source_rows = case.source_divisor.map(|divisor| MatmulRowMap {
        indices: row_indices.as_ref().expect("source row map indices"),
        index_divisor: NonZeroU32::new(divisor).expect("nonzero source row divisor"),
    });
    let destination_rows = case.destination_divisor.map(|divisor| MatmulRowMap {
        indices: row_indices.as_ref().expect("destination row map indices"),
        index_divisor: NonZeroU32::new(divisor).expect("nonzero destination row divisor"),
    });
    let matrices = match offsets.as_ref() {
        Some(offsets) => MatmulMatrixMap::Segmented {
            offsets,
            matrix_count: NonZeroU32::new(case.matrix_count as u32).expect("nonzero matrix count"),
        },
        None => MatmulMatrixMap::Shared,
    };
    let routing = MatmulRouting::Gathered(GatheredMatmul {
        source_rows,
        matrices,
        destination_rows,
        matrix_scales: matrix_scales.as_ref(),
        matrix_biases: matrix_biases.as_ref(),
    });

    let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )?;
    let mut encoder = Encoder::<Cpu>::new(context.as_ref())?;
    kernel.encode(
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &a,
                offset: case.a_offset,
            },
            b: MatmulB::FullPrecision {
                b: &b,
            },
            b_leading_dimension: case.b_layout.leading_dimension.map(|dimension| dimension as u32),
            b_transpose: case.b_layout.transpose,
            d: &mut d,
            d_transform: MatmulDOps {
                ab_scale: case.epilogue.ab_scale,
                accumulate: case.epilogue.accumulate,
                bias: shared_bias.as_ref(),
                rht_factors: None,
                soft_cap: case.epilogue.soft_cap,
            },
            routing,
            m: case.m as u32,
            n: case.n as u32,
            k: case.k as u32,
        },
        &mut encoder,
    )?;
    encoder.end_encoding().submit().wait_until_completed()?;

    Ok(allocation_to_vec::<Cpu, T>(&d))
}

fn run_cpu<T: ArrayElement + Float>(case: &Case<T>) -> Vec<T> {
    try_run_cpu(case).expect("gathered CPU matmul")
}

fn assert_case<T: ArrayElement + Float + Display>(
    case: &Case<T>,
    tolerance: f32,
    label: &str,
) -> Vec<T> {
    let expected = scalar_reference(case);
    let actual = run_cpu(case);
    assert_eq_float(&expected, &actual, tolerance, label);
    actual
}

#[uzu_test]
fn gathered_lhs_only_reuses_source_rows() {
    let case = Case {
        a: vec![f32::NAN, 2.0, 3.0, 5.0],
        a_offset: std::mem::size_of::<f32>(),
        b: vec![7.0],
        row_indices: Some(vec![2, 0, 2, 1]),
        source_divisor: Some(1),
        destination_divisor: None,
        offsets: None,
        matrix_scales: None,
        matrix_biases: None,
        m: 4,
        n: 1,
        k: 1,
        matrix_count: 1,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "lhs-only gathered matmul");
    assert_eq!(actual, vec![35.0, 14.0, 35.0, 21.0]);
}

#[uzu_test]
fn gathered_rhs_segments_allow_empty_and_skewed_matrices() {
    let case = Case {
        a: (1..=7).map(|value| value as f32).collect(),
        a_offset: 0,
        b: vec![17.0f32, 10.0, 100.0, 19.0, 1000.0, 23.0],
        row_indices: None,
        source_divisor: None,
        destination_divisor: None,
        offsets: Some(vec![0, 0, 1, 6, 6, 7, 7]),
        matrix_scales: None,
        matrix_biases: None,
        m: 7,
        n: 1,
        k: 1,
        matrix_count: 6,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "rhs-segmented gathered matmul");
    assert_eq!(actual, vec![10.0, 200.0, 300.0, 400.0, 500.0, 600.0, 7000.0]);
}

#[uzu_test]
fn gathered_combined_reuses_route_map_for_source_and_destination() {
    // Expert-major bucket rows point back to token-major top-k route slots.
    let bucketed_routes = vec![1, 4, 7, 2, 6, 0, 3, 5];
    let case = Case {
        a: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        a_offset: 0,
        b: vec![
            1.0, 0.0, 0.0, 1.0, // expert 0: identity
            1.0, 1.0, 1.0, -1.0, // expert 1: sum and difference
            2.0, 0.0, 0.0, 3.0, // expert 2: independently scaled channels
        ],
        row_indices: Some(bucketed_routes),
        source_divisor: Some(2),
        destination_divisor: Some(1),
        offsets: Some(vec![0, 3, 5, 8]),
        matrix_scales: None,
        matrix_biases: None,
        m: 8,
        n: 2,
        k: 2,
        matrix_count: 3,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "combined gathered matmul");
    assert_eq!(actual, vec![2.0, 6.0, 1.0, 2.0, 7.0, -1.0, 6.0, 12.0, 5.0, 6.0, 10.0, 18.0, 15.0, -1.0, 7.0, 8.0]);
}

#[uzu_test]
fn gathered_matrix_scale_precedes_matrix_bias() {
    let case = Case {
        a: vec![1.0f32; 3],
        a_offset: 0,
        b: vec![2.0f32; 3],
        row_indices: None,
        source_divisor: None,
        destination_divisor: None,
        offsets: Some(vec![0, 1, 2, 3]),
        matrix_scales: Some(vec![0.0, 2.0, -1.0]),
        matrix_biases: Some(vec![7.0, 11.0, 13.0]),
        m: 3,
        n: 1,
        k: 1,
        matrix_count: 3,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "per-matrix gathered transforms");
    assert_eq!(actual, vec![7.0, 15.0, 11.0]);
}

#[uzu_test]
fn gathered_nontransposed_matrix_bank_honors_leading_dimension() {
    let case = Case {
        a: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        a_offset: 0,
        b: vec![
            1.0,
            0.0,
            f32::NAN,
            0.0,
            1.0,
            f32::NAN, // matrix 0: identity with padding
            2.0,
            3.0,
            f32::NAN,
            4.0,
            5.0,
            f32::NAN, // matrix 1: padded K-by-N
        ],
        row_indices: None,
        source_divisor: None,
        destination_divisor: None,
        offsets: Some(vec![0, 1, 3]),
        matrix_scales: None,
        matrix_biases: None,
        m: 3,
        n: 2,
        k: 2,
        matrix_count: 2,
        b_layout: BLayout {
            transpose: false,
            leading_dimension: Some(3),
        },
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "nontransposed padded gathered matmul");
    assert_eq!(actual, vec![1.0, 2.0, 22.0, 29.0, 34.0, 45.0]);
}

#[uzu_test]
fn gathered_epilogue_preserves_shared_transform_order() {
    let case = Case {
        a: vec![2.0f32, 3.0],
        a_offset: 0,
        b: vec![4.0f32, 5.0],
        row_indices: None,
        source_divisor: None,
        destination_divisor: None,
        offsets: Some(vec![0, 1, 2]),
        matrix_scales: Some(vec![2.0, -1.0]),
        matrix_biases: Some(vec![7.0, 11.0]),
        m: 2,
        n: 1,
        k: 1,
        matrix_count: 2,
        b_layout: BLayout::default(),
        epilogue: Epilogue {
            initial_d: Some(vec![17.0, 19.0]),
            ab_scale: 0.5,
            accumulate: true,
            shared_bias: Some(vec![13.0]),
            soft_cap: Some(10.0),
        },
    };

    let actual = assert_case(&case, 1e-6, "gathered matmul epilogue");
    let expected = vec![10.0 * 4.5f32.tanh(), 10.0 * 3.55f32.tanh()];
    assert_eq_float(&expected, &actual, 1e-6, "explicit gathered epilogue order");
}

#[uzu_test]
fn gathered_destination_map_does_not_remap_source_rows() {
    let case = Case {
        a: vec![2.0f32, 3.0],
        a_offset: 0,
        b: vec![10.0f32],
        row_indices: Some(vec![1, 0]),
        source_divisor: None,
        destination_divisor: Some(1),
        offsets: None,
        matrix_scales: None,
        matrix_biases: None,
        m: 2,
        n: 1,
        k: 1,
        matrix_count: 1,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    let actual = assert_case(&case, 1e-6, "destination-only gathered matmul");
    assert_eq!(actual, vec![30.0, 20.0]);
}

#[uzu_test]
fn gathered_rejects_undersized_row_map() {
    let case = Case {
        a: vec![1.0f32, 2.0],
        a_offset: 0,
        b: vec![3.0f32],
        row_indices: Some(vec![0]),
        source_divisor: Some(1),
        destination_divisor: None,
        offsets: None,
        matrix_scales: None,
        matrix_biases: None,
        m: 2,
        n: 1,
        k: 1,
        matrix_count: 1,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    assert!(try_run_cpu(&case).is_err());
}

#[uzu_test]
fn gathered_rejects_misaligned_a_offset() {
    let case = Case {
        a: vec![1.0f32, 2.0],
        a_offset: 1,
        b: vec![3.0f32; 4],
        row_indices: None,
        source_divisor: None,
        destination_divisor: None,
        offsets: None,
        matrix_scales: None,
        matrix_biases: None,
        m: 1,
        n: 4,
        k: 1,
        matrix_count: 1,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    };

    assert!(try_run_cpu(&case).is_err());
}

fn cast<T: Float>(value: f32) -> T {
    T::from(value).unwrap()
}

fn tail_case<T: ArrayElement + Float>() -> Case<T> {
    const M: usize = 39;
    const N: usize = 33;
    const K: usize = 35;
    const SOURCE_ROWS: usize = 13;

    let a = (0..SOURCE_ROWS * K).map(|index| cast::<T>(((index * 7) % 23) as f32 * 0.03125 - 0.34375)).collect();
    let mut b: Vec<T> =
        (0..4 * N * K).map(|index| cast::<T>(((index * 11) % 19) as f32 * 0.015625 - 0.140625)).collect();
    b[N * K..2 * N * K].fill(T::nan());
    let matrix_biases = (0..4 * N).map(|index| cast::<T>(((index * 5) % 13) as f32 * 0.0625 - 0.375)).collect();

    Case {
        a,
        a_offset: 0,
        b,
        row_indices: Some((0..M).map(|assignment| ((assignment * 17 + 5) % M) as u32).collect()),
        source_divisor: Some(3),
        destination_divisor: Some(1),
        offsets: Some(vec![0, 1, 1, 6, 39]),
        matrix_scales: Some(vec![cast(0.5), cast(-1.0), cast(1.25), cast(-0.25)]),
        matrix_biases: Some(matrix_biases),
        m: M,
        n: N,
        k: K,
        matrix_count: 4,
        b_layout: BLayout::default(),
        epilogue: Epilogue::default(),
    }
}

#[uzu_test]
fn gathered_unaligned_tail_f32() {
    assert_case(&tail_case::<f32>(), 1e-5, "unaligned F32 gathered matmul");
}

#[uzu_test]
fn gathered_unaligned_tail_bf16() {
    assert_case(&tail_case::<bf16>(), 0.02, "unaligned BF16 gathered matmul");
}

#[uzu_test]
fn gathered_unaligned_tail_f16() {
    assert_case(&tail_case::<f16>(), 0.02, "unaligned F16 gathered matmul");
}
