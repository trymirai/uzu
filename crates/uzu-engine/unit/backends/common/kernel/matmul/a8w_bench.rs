#![cfg(backend = "metal")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, CommandBuffer, CommandBufferEncoding,
            gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationQuantization, ActivationTransform, Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, QuantParams,
                    QuantParamsLayout, QuantizedB, QuantizedCorrection,
                },
            },
        },
        metal::{GemmEngine, Metal, MetalContext},
    },
    data_type::DataType,
    tests::{
        helpers::{create_buffer, create_buffer_with_data},
        matmul::{QuantInput, iter_encode_loop_named, qwen3_layer_shapes, transpose_metadata},
        util::{shared_metal_context, type_short_name},
    },
};

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

#[derive(Clone, Copy)]
enum BenchPath {
    A8GemmMxu,
    Bf16GemmMxu,
    Bf16Routed,
}

impl BenchPath {
    fn label(self) -> &'static str {
        match self {
            BenchPath::A8GemmMxu => "a8_gemm_mxu",
            BenchPath::Bf16GemmMxu => "abf16_gemm_mxu",
            BenchPath::Bf16Routed => "abf16_routed",
        }
    }
}

struct BenchmarkData {
    unsigned_weights: <Metal as Backend>::GlobalBuffer,
    a8_weights: <Metal as Backend>::GlobalBuffer,
    weight_scales: <Metal as Backend>::GlobalBuffer,
    /// The `[G, N]` scale plane the A8 MXU route reads.
    group_major_weight_scales: <Metal as Backend>::GlobalBuffer,
    activations: <Metal as Backend>::GlobalBuffer,
    rht_factors: <Metal as Backend>::GlobalBuffer,
    a_working: <Metal as Backend>::GlobalBuffer,
    a_int8: <Metal as Backend>::GlobalBuffer,
    a_scales: <Metal as Backend>::GlobalBuffer,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    mode: QuantizationMode,
}

impl BenchmarkData {
    fn new(
        context: &MetalContext,
        m: u32,
        k: u32,
        n: u32,
        bits: u32,
        group_size: u32,
        seed: u64,
    ) -> Self {
        let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleSymmetric, seed);
        let input = input.with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);

        let unsigned_weights = create_buffer_with_data::<Metal, u32>(context, &input.w_packed);
        let a8_weights = input.weights_for_upload();
        let a8_weights = create_buffer_with_data::<Metal, u32>(context, &a8_weights);
        let weight_scales = create_buffer_with_data::<Metal, bf16>(context, &input.scales);
        let mut group_major_weight_scales = create_buffer_with_data::<Metal, bf16>(context, &input.scales);
        transpose_metadata(group_major_weight_scales.as_slice_mut(), n, k.div_ceil(group_size), 16);
        let activations = create_buffer_with_data::<Metal, bf16>(context, &input.x);
        let rht: Vec<i32> = (0..k)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let rht_factors = create_buffer_with_data::<Metal, i32>(context, &rht);

        let groups = k / group_size;
        Self {
            unsigned_weights,
            a8_weights,
            weight_scales,
            group_major_weight_scales,
            activations,
            rht_factors,
            a_working: create_buffer::<Metal, bf16>(context, (m * k) as usize),
            a_int8: create_buffer::<Metal, i8>(context, (m * k) as usize),
            a_scales: create_buffer::<Metal, f32>(context, (m * groups) as usize),
            m,
            k,
            n,
            group_size,
            mode: if bits == 4 {
                QuantizationMode::U4
            } else {
                QuantizationMode::U8
            },
        }
    }

    fn bf16_arguments<'a, D: BufferMut<Backend = Metal>>(
        &'a self,
        output: D,
    ) -> MatmulArguments<
        'a,
        Metal,
        &'a <Metal as Backend>::GlobalBuffer,
        &'a <Metal as Backend>::GlobalBuffer,
        D,
        &'a <Metal as Backend>::GlobalBuffer,
    > {
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &self.a_working,
                offset: 0,
            },
            b: MatmulB::Quantized(QuantizedB {
                codes: &self.unsigned_weights,
                scales: &self.weight_scales,
                correction: QuantizedCorrection::Symmetric,
                params: QuantParams::new(QuantParamsLayout::OutputGroup, self.n, self.k.div_ceil(self.group_size)),
                mode: self.mode,
                group_size: self.group_size,
                signed_codes: false,
            }),
            b_leading_dimension: None,
            b_transpose: true,
            d: output,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }
}

fn encode_step(
    path: BenchPath,
    data: &mut BenchmarkData,
    output: impl BufferMut<Backend = Metal>,
    prepare: &ActivationTransform<Metal>,
    hadamard: &ActivationTransform<Metal>,
    matmul: &mut MetalMatmul,
    command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
) {
    match path {
        BenchPath::A8GemmMxu => {
            prepare.encode_quantize(
                &data.activations,
                &mut data.a_int8,
                &mut data.a_scales,
                None::<&mut <Metal as Backend>::GlobalBuffer>,
                &data.rht_factors,
                data.m,
                data.k,
                command_buffer,
            );
            let args =
                MatmulArguments::<Metal, _, &<Metal as Backend>::GlobalBuffer, _, &<Metal as Backend>::GlobalBuffer> {
                    a: MatmulA::Int8Symmetric {
                        values: &data.a_int8,
                        scales: &data.a_scales,
                        group_sums: None,
                        scale_group_size: 128,
                        code_layout: Int8CodeLayout::for_right_bits(DataType::from(data.mode).size_in_bits() as u32)
                            .expect("W4/W8 benchmark"),
                    },
                    b: MatmulB::Quantized(QuantizedB {
                        codes: &data.a8_weights,
                        scales: &data.group_major_weight_scales,
                        correction: QuantizedCorrection::Symmetric,
                        params: QuantParams::new(
                            QuantParamsLayout::GroupOutput,
                            data.n,
                            data.k.div_ceil(data.group_size),
                        ),
                        mode: data.mode,
                        group_size: data.group_size,
                        signed_codes: !matches!(data.mode, QuantizationMode::U4),
                    }),
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: output,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None,
                    m: data.m,
                    n: data.n,
                    k: data.k,
                };
            matmul.gemm.encode_with_engine(args, GemmEngine::Mxu, command_buffer).expect("a8 gemm mxu encode");
        },
        BenchPath::Bf16GemmMxu => {
            command_buffer.encode_copy(&data.activations, &mut data.a_working);
            hadamard.encode_fp_in_place(&mut data.a_working, &data.rht_factors, data.m, data.k, command_buffer);
            let args = data.bf16_arguments(output);
            matmul.gemm.encode_with_engine(args, GemmEngine::Mxu, command_buffer).expect("bf16 gemm mxu encode");
        },
        BenchPath::Bf16Routed => {
            command_buffer.encode_copy(&data.activations, &mut data.a_working);
            hadamard.encode_fp_in_place(&mut data.a_working, &data.rht_factors, data.m, data.k, command_buffer);
            let args = data.bf16_arguments(output);
            matmul.encode(args, command_buffer).expect("routed bf16 matmul encode");
        },
    }
}

fn bench_bits(
    c: &mut Criterion,
    context: &MetalContext,
    prepare: &ActivationTransform<Metal>,
    hadamard: &ActivationTransform<Metal>,
    bits: u32,
) {
    let mut matmul = <MetalMatmul as MatmulKernel>::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut group = c.benchmark_group(format!("{}/Kernel/A8W/w{bits}", type_short_name::<Metal>()));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(800));

    for (layer, shape) in qwen3_layer_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let mut data = BenchmarkData::new(
            context,
            m,
            k,
            n,
            bits,
            HADAMARD_TRANSFORM_BLOCK_SIZE,
            0xA8_00 ^ u64::from(bits) ^ k as u64 ^ n as u64,
        );
        let mut output = create_buffer::<Metal, bf16>(context, (m * n) as usize);
        let shape_label = format!("{layer}_m{m}_k{k}_n{n}");

        let paths = [BenchPath::A8GemmMxu, BenchPath::Bf16GemmMxu, BenchPath::Bf16Routed];

        group.throughput(Throughput::Elements((m * k * n) as u64));
        for path in paths {
            group.bench_function(BenchmarkId::new(path.label(), &shape_label), |bench| {
                let benchmark_path =
                    format!("{}/Kernel/A8W/w{bits}/{}/{shape_label}", type_short_name::<Metal>(), path.label());
                iter_encode_loop_named::<Metal, _>(context, bench, &benchmark_path, |command_buffer| {
                    encode_step(path, &mut data, &mut output, prepare, hadamard, &mut matmul, command_buffer);
                });
            });
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_a8w(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    let hadamard = ActivationTransform::<Metal>::input_rht(&context, DataType::BF16, true).expect("hadamard kernel");

    for bits in [8u32, 4u32] {
        let prepare = ActivationTransform::<Metal>::quantize(
            &context,
            DataType::BF16,
            ActivationQuantization::new(
                128,
                128,
                false,
                Int8CodeLayout::for_right_bits(bits).expect("W4/W8 benchmark"),
            )
            .expect("supported activation quantization"),
        )
        .expect("prepare kernel");
        bench_bits(c, &context, &prepare, &hadamard, bits);
    }
}
