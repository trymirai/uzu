use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{
            Backend, Buffer, Context, Encoder, Kernels,
            kernel::{QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel},
        },
        cpu::Cpu,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use itertools::iproduct;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const LORA_RANK: u32 = 16;

use super::{Input, check_tolerance, pack_weights_u32, pack_zero_points};
use crate::{
    common::{helpers::alloc_buffer_with_data, type_short_name},
    uzu_bench, uzu_test,
};

fn get_expected<T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Failed to create Context");

    let w_buf = alloc_buffer_with_data::<Cpu, u32>(&context, &input.w_packed);
    let scales_buf = alloc_buffer_with_data::<Cpu, T>(&context, &input.scales);
    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_buffer_with_data::<Cpu, u8>(&context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_buffer_with_data::<Cpu, T>(&context, b));
    let x_buf = alloc_buffer_with_data::<Cpu, T>(&context, &input.x);
    let output_size = (input.m as usize) * (input.n as usize) * T::data_type().size_in_bytes();
    let mut y_buf = context.create_buffer(output_size).expect("Failed to create buffer");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    let kernel = <<Cpu as Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        input.bits,
        input.use_zero_points,
        input.use_mlx_quant,
    )
    .expect("Failed to create QuantizedMatmulQmvKernel");
    kernel.encode(
        &w_buf,
        &scales_buf,
        zp_buf.as_ref(),
        bias_buf.as_ref(),
        &x_buf,
        &mut y_buf,
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const T;
    let y_len = (input.m as usize) * (input.n as usize);
    unsafe { std::slice::from_raw_parts(y_ptr, y_len) }.to_vec()
}

pub struct LoraInputs<'a, T: ArrayElement + Float> {
    pub h: &'a [T],
    pub a_up: &'a [T],
    pub scale: f32,
}

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &Input<T>,
    lora: Option<&LoraInputs<T>>,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let w_buf = alloc_buffer_with_data::<B, u32>(&context, &input.w_packed);
    let scales_buf = alloc_buffer_with_data::<B, T>(&context, &input.scales);
    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_buffer_with_data::<B, u8>(&context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_buffer_with_data::<B, T>(&context, b));
    let x_buf = alloc_buffer_with_data::<B, T>(&context, &input.x);
    let h_buf = lora.map(|l| alloc_buffer_with_data::<B, T>(&context, l.h));
    let a_up_buf = lora.map(|l| alloc_buffer_with_data::<B, T>(&context, l.a_up));
    let output_size = (input.m as usize) * (input.n as usize) * T::data_type().size_in_bytes();
    let mut y_buf = context.create_buffer(output_size).expect("Failed to create buffer");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        input.bits,
        LORA_RANK,
        input.use_zero_points,
        input.use_mlx_quant,
        false,
        lora.is_some(),
    )
    .expect("Failed to create QuantizedMatmulQmvFastKernel");
    kernel.encode(
        &w_buf,
        &scales_buf,
        zp_buf.as_ref(),
        bias_buf.as_ref(),
        &x_buf,
        &mut y_buf,
        None::<&B::Buffer>,
        h_buf.as_ref(),
        a_up_buf.as_ref(),
        lora.map(|l| l.scale),
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const T;
    let y_len = (input.m as usize) * (input.n as usize);
    unsafe { std::slice::from_raw_parts(y_ptr, y_len) }.to_vec()
}

fn get_test_data<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) -> (Input<T>, Vec<T>) {
    let num_groups_k = (k + group_size as usize - 1) / group_size as usize;

    let mut weights_raw: Vec<u8> = Vec::with_capacity(n * k);
    let max_val = if bits == 4 {
        15u8
    } else {
        255u8
    };
    for j in 0..n {
        for l in 0..k {
            weights_raw.push(((j * 3 + l * 7 + 1) % (max_val as usize + 1)) as u8);
        }
    }
    let w_packed = pack_weights_u32(&weights_raw, bits);

    let mut scales_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
    for j in 0..n {
        for g in 0..num_groups_k {
            scales_f32.push(0.5 + 0.1 * ((j + g) % 5) as f32);
        }
    }
    let scales: Vec<T> = scales_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    let (zero_points, biases) = if use_zero_points {
        let mut zp_raw: Vec<u8> = Vec::with_capacity(n * num_groups_k);
        for j in 0..n {
            for g in 0..num_groups_k {
                let zp_val = ((j * 2 + g * 3) % (max_val as usize + 1)) as u8;
                zp_raw.push(zp_val);
            }
        }
        let mut zp_packed: Vec<u8> = Vec::with_capacity(n * zp_stride);
        for j in 0..n {
            let row = &zp_raw[j * num_groups_k..(j + 1) * num_groups_k];
            let packed_row = pack_zero_points(row, bits);
            let mut padded = packed_row;
            padded.resize(zp_stride, 0);
            zp_packed.extend_from_slice(&padded);
        }
        (Some(zp_packed), None)
    } else if use_mlx_quant {
        let mut biases_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
        for j in 0..n {
            for g in 0..num_groups_k {
                biases_f32.push(0.01 * ((j + g * 2) % 7) as f32);
            }
        }
        let biases: Vec<T> = biases_f32.iter().map(|&v| T::from(v).unwrap()).collect();
        (None, Some(biases))
    } else {
        unreachable!("Must use either zero_points or mlx_quant");
    };

    let mut x_f32: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for l in 0..k {
            x_f32.push(0.1 * f32::sin((i * k + l) as f32 * 0.05) + 0.5);
        }
    }
    let x: Vec<T> = x_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    let input = Input {
        w_packed,
        scales,
        zero_points,
        biases,
        x,
        k: k as u32,
        n: n as u32,
        m: m as u32,
        group_size,
        bits,
        use_zero_points,
        use_mlx_quant,
    };

    let expected = get_expected::<T>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let (rel_tol, abs_tol): (f64, f64) = match T::data_type() {
        DataType::BF16 => (0.05, 0.5),
        DataType::F16 => (0.02, 0.5),
        _ => (0.01, 0.1),
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input, None);
        assert_eq!(expected.len(), output.len(), "Output length mismatch");

        let mut errors = 0;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let exp_f32 = exp.to_f32().unwrap();
            let got_f32 = got.to_f32().unwrap();
            if !check_tolerance(exp_f32, got_f32, rel_tol, abs_tol) {
                if errors < 5 {
                    eprintln!("  idx={} expected={} got={} diff={}", i, exp_f32, got_f32, (exp_f32 - got_f32).abs());
                }
                errors += 1;
            }
        }
        assert_eq!(
            errors,
            0,
            "QMV fast kernel: backend={}, m={}, k={}, n={}, gs={}, bits={}, zp={}, mlx={}: {} mismatches",
            std::any::type_name::<B>(),
            input.m,
            input.k,
            input.n,
            input.group_size,
            input.bits,
            input.use_zero_points,
            input.use_mlx_quant,
            errors,
        );
    });
}

fn get_test_dims(
    group_size: u32,
    bits: u32,
    is_half_precision: bool,
) -> Vec<(usize, usize, usize)> {
    let gs = group_size as usize;
    let block_size = if bits == 4 {
        512
    } else {
        256
    };
    let k = block_size;
    let n = gs.max(32);
    if is_half_precision && bits == 8 {
        vec![(1, k, n), (2, k, n)]
    } else {
        vec![(1, k, n), (2, k, n * 2), (1, k, n * 2)]
    }
}

fn test_basic<T: ArrayElement + Float + Debug + Display>(
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let is_half = matches!(T::data_type(), DataType::F16 | DataType::BF16);
    let dims = get_test_dims(group_size, bits, is_half);

    for (m, k, n) in dims {
        let (input, expected) = get_test_data::<T>(m, k, n, group_size, bits, use_zero_points, use_mlx_quant);
        test_internal::<T>(&input, &expected);
    }
}

macro_rules! qmv_fast_test {
    ($name:ident, gs=$gs:expr, bits=$bits:expr, zp=$zp:expr, mlx=$mlx:expr) => {
        #[uzu_test]
        fn $name() {
            for_each_float_type!(|F| {
                test_basic::<F>($gs, $bits, $zp, $mlx);
            })
        }
    };
}

qmv_fast_test!(test_gs32_4bit_zp, gs = 32, bits = 4, zp = true, mlx = false);
qmv_fast_test!(test_gs64_4bit_zp, gs = 64, bits = 4, zp = true, mlx = false);
qmv_fast_test!(test_gs128_4bit_zp, gs = 128, bits = 4, zp = true, mlx = false);
qmv_fast_test!(test_gs32_8bit_zp, gs = 32, bits = 8, zp = true, mlx = false);
qmv_fast_test!(test_gs64_8bit_zp, gs = 64, bits = 8, zp = true, mlx = false);
qmv_fast_test!(test_gs128_8bit_zp, gs = 128, bits = 8, zp = true, mlx = false);
qmv_fast_test!(test_gs32_4bit_mlx, gs = 32, bits = 4, zp = false, mlx = true);
qmv_fast_test!(test_gs64_4bit_mlx, gs = 64, bits = 4, zp = false, mlx = true);
qmv_fast_test!(test_gs128_4bit_mlx, gs = 128, bits = 4, zp = false, mlx = true);
qmv_fast_test!(test_gs32_8bit_mlx, gs = 32, bits = 8, zp = false, mlx = true);
qmv_fast_test!(test_gs64_8bit_mlx, gs = 64, bits = 8, zp = false, mlx = true);
qmv_fast_test!(test_gs128_8bit_mlx, gs = 128, bits = 8, zp = false, mlx = true);

#[uzu_test]
fn test_lora_gs128_4bit_mlx() {
    const RANK: usize = 16;
    const M: usize = 1;
    const K: usize = 512;
    const N: usize = 64;
    let (input, _) = get_test_data::<bf16>(M, K, N, 128, 4, false, true);
    let h: Vec<bf16> = (0..M * RANK).map(|i| bf16::from_f32(0.01 * f32::sin(i as f32 * 0.3))).collect();
    let a_up: Vec<bf16> = (0..N * RANK).map(|i| bf16::from_f32(0.01 * f32::sin(i as f32 * 0.17))).collect();
    let lora = LoraInputs { h: &h, a_up: &a_up, scale: 0.5 };
    let cpu_out = get_output::<Cpu, bf16>(&input, Some(&lora));
    let (rel_tol, abs_tol): (f64, f64) = (0.05, 0.5);
    for_each_non_cpu_backend!(|B| {
        let out = get_output::<B, bf16>(&input, Some(&lora));
        let mut errors = 0usize;
        for (i, (&exp, &got)) in cpu_out.iter().zip(out.iter()).enumerate() {
            let e = exp.to_f32();
            let g = got.to_f32();
            if !check_tolerance(e, g, rel_tol, abs_tol) {
                if errors < 5 { eprintln!("  idx={} exp={} got={} diff={}", i, e, g, (e - g).abs()); }
                errors += 1;
            }
        }
        assert_eq!(errors, 0, "QmvFast LoRA: backend={} mismatches={}", std::any::type_name::<B>(), errors);
    });
}

fn gen_random<T: rand::distr::uniform::SampleUniform + PartialOrd + Copy, R: rand::Rng>(
    rng: &mut R,
    range: std::ops::Range<T>,
    len: usize,
) -> Box<[T]> {
    (0..len).map(|_| rng.random_range(range.clone())).collect()
}

fn bench_qmv_fast_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmvFast/{}", type_short_name::<B>(), label));
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };

    for (m, n, k) in iproduct!([1, 2, 3, 4], [1024, 2048, 4096, 14336, 65536], [1024, 2048, 4096, 8192, 14336]) {
        if n % 8 != 0 || k % block_size != 0 {
            continue;
        }

        let num_groups = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(42);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
            context,
            T::data_type(),
            group_size,
            bits,
            LORA_RANK,
            use_zero_points,
            use_mlx_quant,
            false,
            false,
        )
        .unwrap();

        let w_buf = alloc_buffer_with_data::<B, u32>(
            context,
            &gen_random::<u32, _>(&mut rng, 0..u32::MAX, n * k * bits as usize / 32),
        );
        let scales_buf = alloc_buffer_with_data::<B, T>(
            context,
            &gen_random::<f32, _>(&mut rng, 0.01..1.0, n * num_groups)
                .iter()
                .map(|&v| T::from(v).unwrap())
                .collect::<Vec<_>>(),
        );
        let x_buf = alloc_buffer_with_data::<B, T>(
            context,
            &gen_random::<f32, _>(&mut rng, -1.0..1.0, m * k).iter().map(|&v| T::from(v).unwrap()).collect::<Vec<_>>(),
        );
        let mut y_buf = context.create_buffer(m * n * std::mem::size_of::<T>()).unwrap();

        let zp_buf = use_zero_points.then(|| {
            let zp_stride = if bits == 4 {
                (num_groups + 1) / 2
            } else {
                num_groups
            };
            alloc_buffer_with_data::<B, u8>(context, &gen_random::<u8, _>(&mut rng, 0..u8::MAX, n * zp_stride))
        });
        let bias_buf = use_mlx_quant.then(|| {
            alloc_buffer_with_data::<B, T>(
                context,
                &gen_random::<f32, _>(&mut rng, -0.5..0.5, n * num_groups)
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect::<Vec<_>>(),
            )
        });

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("M[{m}]N[{n}]K[{k}]")), |b| {
            b.iter_custom(|n_iters| {
                let mut encoder = Encoder::<B>::new(context).unwrap();
                for _ in 0..n_iters {
                    kernel.encode(
                        &w_buf,
                        &scales_buf,
                        zp_buf.as_ref(),
                        bias_buf.as_ref(),
                        &x_buf,
                        &mut y_buf,
                        None::<&B::Buffer>,
                        None::<&B::Buffer>,
                        None::<&B::Buffer>,
                        None::<f32>,
                        k as u32,
                        n as u32,
                        m as u32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            })
        });
    }
}

#[uzu_bench]
fn bench_qmv_fast(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_qmv_fast_typed::<B, bf16>(c, &context, "Mlx_BF16_gs128", 128, 4, false, true);
        bench_qmv_fast_typed::<B, f16>(c, &context, "ZP_F16_gs64", 64, 4, true, false);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, true, false);
    });
}
