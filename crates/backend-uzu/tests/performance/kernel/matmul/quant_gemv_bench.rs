use backend_uzu::{
    array::ArrayElement,
    backends::common::{
        Allocation, Backend, Context,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{
            Kernels,
            matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::{QuantBuffers, QuantInput, Shape, bench_quant_gemv_shapes, iter_encode_loop, quant_arguments},
        type_short_name,
    },
    uzu_bench,
};

fn bench_gemv_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Gemv/{}", type_short_name::<B>(), label));

    for shape in bench_quant_gemv_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<B, T>::allocate(context, &input);
        let mut matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                let args = quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("encode failed");
            });
        });
    }
}

const LLOYD_MAX_CODEBOOK_SIZE: usize = 16;
const BIAS_CODEBOOK_SIZE: usize = 16;

struct LloydMaxQuantInput<T: ArrayElement + Float> {
    w_packed: Vec<u32>,
    scales: Vec<T>,
    codebook: [f16; LLOYD_MAX_CODEBOOK_SIZE],
    bias_indices: Vec<u8>,
    bias_codebook: [f16; BIAS_CODEBOOK_SIZE],
    x: Vec<T>,
    k: u32,
    n: u32,
    m: u32,
    group_size: u32,
    mode: QuantizationMode,
}

impl<T: ArrayElement + Float> LloydMaxQuantInput<T> {
    fn new(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
    ) -> Self {
        let num_groups_k = k / group_size as usize;
        Self {
            w_packed: deterministic_packed_u4_weights(n, k),
            scales: lloyd_max_scales(n, num_groups_k),
            codebook: lloyd_max_codebook(),
            bias_indices: lloyd_max_bias_indices(n, num_groups_k),
            bias_codebook: lloyd_max_bias_codebook(),
            x: lloyd_max_input_values(m, k),
            k: k as u32,
            n: n as u32,
            m: m as u32,
            group_size,
            mode: QuantizationMode::U4,
        }
    }
}

struct LloydMaxQuantBuffers<B: Backend, T: ArrayElement + Float> {
    w: Allocation<B>,
    scales: Allocation<B>,
    codebook: Allocation<B>,
    bias_indices: Allocation<B>,
    bias_codebook: Allocation<B>,
    x: Allocation<B>,
    y: Allocation<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> LloydMaxQuantBuffers<B, T> {
    fn allocate(
        context: &B::Context,
        input: &LloydMaxQuantInput<T>,
    ) -> Self {
        Self {
            w: alloc_allocation_with_data::<B, u32>(context, &input.w_packed),
            scales: alloc_allocation_with_data::<B, T>(context, &input.scales),
            codebook: alloc_allocation_with_data::<B, f16>(context, &input.codebook),
            bias_indices: alloc_allocation_with_data::<B, u8>(context, &input.bias_indices),
            bias_codebook: alloc_allocation_with_data::<B, f16>(context, &input.bias_codebook),
            x: alloc_allocation_with_data::<B, T>(context, &input.x),
            y: alloc_allocation::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        }
    }
}

fn lloyd_max_quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut LloydMaxQuantBuffers<B, T>,
    input: &LloydMaxQuantInput<T>,
) -> MatmulArguments<'a, B> {
    MatmulArguments {
        a: &buffers.x,
        a_offset: 0,
        b: MatmulB::LloydMaxDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            codebook: &buffers.codebook,
            bias_indices: &buffers.bias_indices,
            bias_codebook: &buffers.bias_codebook,
            mode: input.mode,
            group_size: input.group_size,
        },
        b_offset: 0,
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.y,
        d_transform: MatmulDOps::none(),
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

fn lloyd_max_codebook() -> [f16; LLOYD_MAX_CODEBOOK_SIZE] {
    [
        -1.0f32,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_04,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ]
    .map(f16::from_f32)
}

fn lloyd_max_bias_codebook() -> [f16; BIAS_CODEBOOK_SIZE] {
    [
        -0.045f32, -0.039, -0.033, -0.026, -0.020, -0.013, -0.007, 0.0, 0.007, 0.013, 0.020, 0.026, 0.033, 0.039,
        0.045, 0.052,
    ]
    .map(f16::from_f32)
}

fn deterministic_packed_u4_weights(
    output_size: usize,
    input_size: usize,
) -> Vec<u32> {
    (0..(output_size * input_size / 8)).map(|word_index| word_index.wrapping_mul(2_654_435_761) as u32).collect()
}

fn lloyd_max_scale_value(
    output_index: usize,
    group_index: usize,
) -> f32 {
    0.07 + 0.013 * ((output_index + 5 * group_index) % 13) as f32
}

fn lloyd_max_scales<T: ArrayElement + Float>(
    output_size: usize,
    group_count: usize,
) -> Vec<T> {
    (0..output_size)
        .flat_map(|output_index| {
            (0..group_count).map(move |group_index| T::from(lloyd_max_scale_value(output_index, group_index)).unwrap())
        })
        .collect()
}

fn lloyd_max_bias_indices(
    output_size: usize,
    group_count: usize,
) -> Vec<u8> {
    let bias_stride = group_count.div_ceil(2);
    (0..(output_size * bias_stride)).map(|byte_index| byte_index.wrapping_mul(2_246_822_519) as u8).collect()
}

fn lloyd_max_input_values<T: ArrayElement + Float>(
    batch_size: usize,
    input_size: usize,
) -> Vec<T> {
    (0..batch_size)
        .flat_map(|batch_index| {
            (0..input_size).map(move |input_index| {
                T::from((((batch_index * 11 + input_index * 7) % 23) as f32 - 11.0) * 0.022).unwrap()
            })
        })
        .collect()
}

fn bench_lloyd_max_shapes() -> impl Iterator<Item = Shape> {
    [Shape::new(1, 4096, 4096), Shape::new(2, 4096, 4096), Shape::new(3, 4096, 4096), Shape::new(4, 4096, 4096)]
        .into_iter()
}

fn bench_lloyd_max_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Gemv/{}", type_short_name::<B>(), label));

    for shape in bench_lloyd_max_shapes() {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = LloydMaxQuantInput::<T>::new(m, k, n, group_size);
        let mut buffers = LloydMaxQuantBuffers::<B, T>::allocate(context, &input);
        let mut matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                let args = lloyd_max_quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("encode failed");
            });
        });
    }
}

#[uzu_bench]
fn bench_gemv(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs32", 32, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs32", 32, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
        bench_lloyd_max_typed::<B, bf16>(c, &context, "LloydMax_BF16_gs64", 64);
    });
}
