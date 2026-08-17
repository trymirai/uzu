#![cfg(backend = "metal")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationTransform, Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting, MatmulShape},
            },
        },
        metal::{DeviceProfile, GemmEngine, GemvDispatch, GemvSpecialization, Metal, MetalContext},
    },
    data_type::DataType,
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::{QuantInput, iter_encode_loop_named, qwen3_layer_shapes},
        util::{shared_metal_context, type_short_name},
    },
};

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

#[derive(Clone, Copy)]
enum BenchPath {
    A8GemmMxu,
    Bf16GemmMxu,
    Bf16Gemv,
}

impl BenchPath {
    fn label(self) -> &'static str {
        match self {
            BenchPath::A8GemmMxu => "a8_gemm_mxu",
            BenchPath::Bf16GemmMxu => "abf16_gemm_mxu",
            BenchPath::Bf16Gemv => "abf16_gemv",
        }
    }
}

struct BenchmarkData {
    unsigned_weights: Allocation<Metal>,
    signed_weights: Allocation<Metal>,
    weight_scales: Allocation<Metal>,
    activations: Allocation<Metal>,
    rht_factors: Allocation<Metal>,
    a_working: Allocation<Metal>,
    a_int8: Allocation<Metal>,
    a_scales: Allocation<Metal>,
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
        let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleSymmetric, seed)
            .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);

        let unsigned_weights = alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed);
        let signed_weights = alloc_allocation_with_data::<Metal, u32>(context, &input.weights_for_upload());
        let weight_scales = alloc_allocation_with_data::<Metal, bf16>(context, &input.scales);
        let activations = alloc_allocation_with_data::<Metal, bf16>(context, &input.x);
        let rht: Vec<i32> = (0..k)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let rht_factors = alloc_allocation_with_data::<Metal, i32>(context, &rht);

        let groups = k / group_size;
        let a_elements = m as usize * k as usize;
        Self {
            unsigned_weights,
            signed_weights,
            weight_scales,
            activations,
            rht_factors,
            a_working: alloc_allocation::<Metal, bf16>(context, a_elements),
            a_int8: alloc_allocation::<Metal, i8>(context, a_elements),
            a_scales: alloc_allocation::<Metal, f32>(context, m as usize * groups as usize),
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

    fn bf16_arguments<'a>(
        &'a self,
        output: &'a mut Allocation<Metal>,
    ) -> MatmulArguments<'a, 'a, 'a, Metal, &'a Allocation<Metal>> {
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &self.a_working,
                offset: 0,
            },
            b: MatmulB::ScaleSymmetricDequant {
                b: &self.unsigned_weights,
                scales: &self.weight_scales,
                mode: self.mode,
                group_size: self.group_size,
                signed_codes: false,
            },
            b_leading_dimension: None,
            b_transpose: true,
            d: output,
            d_transform: MatmulDOps::none(),
            routing: MatmulRouting::Dense,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_step(
    path: BenchPath,
    data: &mut BenchmarkData,
    output: &mut Allocation<Metal>,
    prepare: &ActivationTransform<Metal>,
    hadamard: &ActivationTransform<Metal>,
    matmul: &mut MetalMatmul,
    gemv: &mut GemvDispatch,
    device_profile: DeviceProfile,
    encoder: &mut Encoder<Metal>,
) {
    match path {
        BenchPath::A8GemmMxu => {
            prepare.encode_quantize(
                &data.activations,
                &mut data.a_int8,
                &mut data.a_scales,
                None::<&mut Allocation<Metal>>,
                &data.rht_factors,
                data.m,
                data.k,
                encoder,
            );
            let args: MatmulArguments<'_, '_, '_, Metal, &Allocation<Metal>> = MatmulArguments {
                a: MatmulA::Int8Symmetric {
                    values: &data.a_int8,
                    scales: &data.a_scales,
                    group_sums: None,
                    group_size: 128,
                },
                b: MatmulB::ScaleSymmetricDequant {
                    b: &data.signed_weights,
                    scales: &data.weight_scales,
                    mode: data.mode,
                    group_size: data.group_size,
                    signed_codes: true,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Dense,
                m: data.m,
                n: data.n,
                k: data.k,
            };
            matmul.gemm.encode_with_engine(args, GemmEngine::Mxu, encoder).expect("a8 gemm mxu encode");
        },
        BenchPath::Bf16GemmMxu => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            hadamard.encode_fp_in_place(&mut data.a_working, &data.rht_factors, data.m, data.k, encoder);
            let args = data.bf16_arguments(output);
            matmul.gemm.encode_with_engine(args, GemmEngine::Mxu, encoder).expect("bf16 gemm mxu encode");
        },
        BenchPath::Bf16Gemv => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            hadamard.encode_fp_in_place(&mut data.a_working, &data.rht_factors, data.m, data.k, encoder);
            let args = data.bf16_arguments(output);
            let spec = GemvSpecialization::select_shape(
                &MatmulShape::from_arguments(&args),
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
                device_profile,
            )
            .expect("bf16 gemv specialization");
            gemv.encode(args, spec, encoder).expect("bf16 gemv encode");
        },
    }
}

fn bench_bits(
    c: &mut Criterion,
    context: &MetalContext,
    device_profile: DeviceProfile,
    prepare: &ActivationTransform<Metal>,
    hadamard: &ActivationTransform<Metal>,
    bits: u32,
) {
    let mut matmul = <MetalMatmul as MatmulKernel>::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut gemv = GemvDispatch::new(DataType::BF16, DataType::BF16, DataType::BF16);

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
        let mut output = alloc_allocation::<Metal, bf16>(context, m as usize * n as usize);
        let shape_label = format!("{layer}_m{m}_k{k}_n{n}");

        let gemv_eligible = GemvSpecialization::select_shape(
            &MatmulShape::from_arguments(&data.bf16_arguments(&mut output)),
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            device_profile,
        )
        .is_some();

        let mut paths = vec![BenchPath::A8GemmMxu, BenchPath::Bf16GemmMxu];
        if gemv_eligible {
            paths.push(BenchPath::Bf16Gemv);
        }

        group.throughput(Throughput::Elements((m * k * n) as u64));
        for path in paths {
            group.bench_function(BenchmarkId::new(path.label(), &shape_label), |bench| {
                let benchmark_path =
                    format!("{}/Kernel/A8W/w{bits}/{}/{shape_label}", type_short_name::<Metal>(), path.label());
                iter_encode_loop_named::<Metal, _>(context, bench, &benchmark_path, |encoder| {
                    encode_step(
                        path,
                        &mut data,
                        &mut output,
                        prepare,
                        hadamard,
                        &mut matmul,
                        &mut gemv,
                        device_profile,
                        encoder,
                    );
                });
            });
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_a8w(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let device_profile = context.device_profile();

    let prepare = ActivationTransform::<Metal>::quantize(&context, DataType::BF16, 128, None).expect("prepare kernel");
    let hadamard = ActivationTransform::<Metal>::input_rht(&context, DataType::BF16, true).expect("hadamard kernel");

    for bits in [8u32, 4u32] {
        bench_bits(c, &context, device_profile, &prepare, &hadamard, bits);
    }
}
