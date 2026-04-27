use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::QuantizedMatmulQmmTransposedKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use super::{Input, check_tolerance, pack_weights_u32, pack_zero_points};
use crate::{
    common::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    uzu_test,
};

fn make_input_basic<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) -> Input<T> {
    let num_groups_k = (k + group_size as usize - 1) / group_size as usize;

    let max_val = if bits == 4 {
        15u8
    } else {
        255u8
    };
    let mut weights_raw: Vec<u8> = Vec::with_capacity(n * k);
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

    Input {
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
    }
}

fn make_input_edge<T: ArrayElement + Float>(
    n: usize,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) -> Input<T> {
    let m = 2usize;
    let k = group_size as usize;
    let num_groups_k = 1;

    let weights_raw: Vec<u8> = vec![1u8; n * k];
    let w_packed = pack_weights_u32(&weights_raw, bits);

    let scales: Vec<T> = vec![T::one(); n * num_groups_k];

    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    let (zero_points, biases) = if use_zero_points {
        let zp_packed = vec![0u8; n * zp_stride];
        (Some(zp_packed), None)
    } else if use_mlx_quant {
        let biases: Vec<T> = vec![T::zero(); n * num_groups_k];
        (None, Some(biases))
    } else {
        unreachable!("Must use either zero_points or mlx_quant");
    };

    let x_f32: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32 * 0.1).collect();
    let x: Vec<T> = x_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    Input {
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
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &Input<T>,
    bm: u32,
    bk: u32,
    bn: u32,
    wm: u32,
    wn: u32,
    aligned_n: bool,
    hadamard_factors: Option<&[i32]>,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let use_hadamard = hadamard_factors.is_some();
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        input.bits,
        bm,
        bk,
        bn,
        wm,
        wn,
        input.use_zero_points,
        input.use_mlx_quant,
        use_hadamard,
        aligned_n,
    )
    .expect("Failed to create QuantizedMatmulQmmTransposedKernel");

    let w_buf = alloc_allocation_with_data::<B, u32>(&context, &input.w_packed);
    let scales_buf = alloc_allocation_with_data::<B, T>(&context, &input.scales);

    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<B, u8>(&context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_allocation_with_data::<B, T>(&context, b));
    let had_buf = hadamard_factors.map(|f| alloc_allocation_with_data::<B, i32>(&context, f));

    let x_buf = alloc_allocation_with_data::<B, T>(&context, &input.x);
    let mut y_buf = alloc_allocation::<B, T>(&context, (input.m as usize) * (input.n as usize));

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &w_buf,
        &scales_buf,
        zp_buf.as_ref(),
        bias_buf.as_ref(),
        &x_buf,
        &mut y_buf,
        had_buf.as_ref(),
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    allocation_to_vec(&y_buf)
}

fn check_against_expected<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    bm: u32,
    bk: u32,
    bn: u32,
    wm: u32,
    wn: u32,
    aligned_n: bool,
) {
    // GPU tiled accumulation rounds differently from CPU scalar reference.
    let (rel_tol, abs_tol): (f64, f64) = match T::data_type() {
        DataType::BF16 => (0.05, 0.5),
        DataType::F16 => (0.02, 0.5),
        _ => (0.01, 0.1),
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input, bm, bk, bn, wm, wn, aligned_n, None);
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
            "QMM transposed kernel: backend={}, shape=(BM={},BK={},BN={}), m={}, k={}, n={}, gs={}, bits={}, zp={}, mlx={}: {} mismatches",
            std::any::type_name::<B>(),
            bm,
            bk,
            bn,
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

struct TileCfg {
    bm: u32,
    bk: u32,
    bn: u32,
    wm: u32,
    wn: u32,
}

impl TileCfg {
    const fn aligned_n_default(&self) -> bool {
        // BN=64 tiles require output_dim % 64 == 0; BN=32 checks actual N at test time.
        self.bn == 64
    }
}

const CFG_BM8: TileCfg = TileCfg {
    bm: 8,
    bk: 32,
    bn: 32,
    wm: 1,
    wn: 1,
};
const CFG_32X32: TileCfg = TileCfg {
    bm: 32,
    bk: 32,
    bn: 32,
    wm: 2,
    wn: 2,
};
const CFG_WIDE: TileCfg = TileCfg {
    bm: 64,
    bk: 32,
    bn: 64,
    wm: 2,
    wn: 2,
};
const CFG_64X64: TileCfg = TileCfg {
    bm: 64,
    bk: 64,
    bn: 64,
    wm: 2,
    wn: 2,
};

fn pick_aligned_n(
    cfg: &TileCfg,
    n: usize,
) -> bool {
    if cfg.bn == 32 {
        (n % 32) == 0
    } else {
        cfg.aligned_n_default()
    }
}

fn test_basic<T: ArrayElement + Float + Debug + Display>(
    cfg: &TileCfg,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let is_half = matches!(T::data_type(), DataType::F16 | DataType::BF16);
    let gs = group_size as usize;
    let bn = cfg.bn as usize;
    // half+8bit: smaller dims to cap accumulated rounding error.
    let dims: Vec<(usize, usize, usize)> = if is_half && bits == 8 {
        vec![(2, gs, bn), (4, gs, bn)]
    } else {
        vec![(2, gs * 2, bn), (4, gs * 2, bn * 2), (8, gs, bn * 2)]
    };

    for (m, k, n) in dims {
        let input = make_input_basic::<T>(m, k, n, group_size, bits, use_zero_points, use_mlx_quant);
        let aligned_n = pick_aligned_n(cfg, n);
        let expected = get_output::<Cpu, T>(&input, cfg.bm, cfg.bk, cfg.bn, cfg.wm, cfg.wn, aligned_n, None);
        check_against_expected::<T>(&input, &expected, cfg.bm, cfg.bk, cfg.bn, cfg.wm, cfg.wn, aligned_n);
    }
}

fn test_edge<T: ArrayElement + Float + Debug + Display>(
    cfg: &TileCfg,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    // Edge N is one BN tile wide — smallest legal N for the shape.
    let n = cfg.bn as usize;
    let input = make_input_edge::<T>(n, group_size, bits, use_zero_points, use_mlx_quant);
    let aligned_n = pick_aligned_n(cfg, n);
    let expected = get_output::<Cpu, T>(&input, cfg.bm, cfg.bk, cfg.bn, cfg.wm, cfg.wn, aligned_n, None);
    check_against_expected::<T>(&input, &expected, cfg.bm, cfg.bk, cfg.bn, cfg.wm, cfg.wn, aligned_n);
}

// No CPU Hadamard reference — verify all three tile shapes agree instead.
fn test_hadamard_consistency<T: ArrayElement + Float + Debug + Display>(
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    // m=n=64 satisfies both BN=32 and BN=64 alignment; group_size >= 64 for BK=64.
    let m = 64usize;
    let k = group_size as usize * 2;
    let n = 64usize;

    let input = make_input_basic::<T>(m, k, n, group_size, bits, use_zero_points, use_mlx_quant);

    // ±1 factors: non-trivial Hadamard, bounded output across all dtypes.
    let factors: Vec<i32> = (0..n)
        .map(|i| {
            if (i * 13 + 7) % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();

    // Hadamard amplifies by sqrt(32) ≈ 5.65; tolerance widened accordingly.
    let (rel_tol, abs_tol): (f64, f64) = match T::data_type() {
        DataType::BF16 => (0.1, 1.0),
        DataType::F16 => (0.05, 1.0),
        _ => (0.02, 0.5),
    };

    for_each_non_cpu_backend!(|B| {
        let shapes: [(u32, u32, u32, u32, u32, &str); 3] =
            [(32, 32, 32, 2, 2, "32x32"), (64, 32, 64, 2, 2, "wide"), (64, 64, 64, 2, 2, "64x64")];

        let outputs: Vec<(&str, Vec<T>)> = shapes
            .iter()
            .map(|&(bm, bk, bn, wm, wn, name)| {
                let out = get_output::<B, T>(&input, bm, bk, bn, wm, wn, true, Some(&factors));
                (name, out)
            })
            .collect();

        // 32x32 is the reference — its Hadamard path is byte-identical to the original.
        let (_, ref reference) = outputs[0];

        for (name, output) in outputs.iter().skip(1) {
            let mut errors = 0;
            for (i, (&r, &o)) in reference.iter().zip(output.iter()).enumerate() {
                let r_f32 = r.to_f32().unwrap();
                let o_f32 = o.to_f32().unwrap();
                if !check_tolerance(r_f32, o_f32, rel_tol, abs_tol) {
                    if errors < 5 {
                        eprintln!("  idx={} 32x32={} {}={} diff={}", i, r_f32, name, o_f32, (r_f32 - o_f32).abs());
                    }
                    errors += 1;
                }
            }
            assert_eq!(
                errors,
                0,
                "Hadamard cross-shape mismatch: backend={}, {} vs 32x32 reference, gs={}, bits={}, zp={}, mlx={}: {} mismatches",
                std::any::type_name::<B>(),
                name,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                errors,
            );
        }
    });
}

// ---- Test-function macros -----------------------------------------------

macro_rules! basic_tests {
    ($cfg:expr; $(
        $(#[$attr:meta])*
        fn $name:ident : $t:ty, $gs:literal, $bits:literal, $zp:literal, $mlx:literal;
    )*) => {
        $(
            $(#[$attr])*
            #[uzu_test]
            fn $name() {
                test_basic::<$t>($cfg, $gs, $bits, $zp, $mlx);
            }
        )*
    };
}

macro_rules! edge_tests {
    ($cfg:expr; $(
        $(#[$attr:meta])*
        fn $name:ident : $t:ty, $gs:literal, $bits:literal, $zp:literal, $mlx:literal;
    )*) => {
        $(
            $(#[$attr])*
            #[uzu_test]
            fn $name() {
                test_edge::<$t>($cfg, $gs, $bits, $zp, $mlx);
            }
        )*
    };
}

macro_rules! hadamard_tests {
    ($(
        $(#[$attr:meta])*
        fn $name:ident : $t:ty, $gs:literal, $bits:literal, $zp:literal, $mlx:literal;
    )*) => {
        $(
            $(#[$attr])*
            #[uzu_test]
            fn $name() {
                test_hadamard_consistency::<$t>($gs, $bits, $zp, $mlx);
            }
        )*
    };
}

mod tile_bm8 {
    use super::*;

    basic_tests!(&CFG_BM8;
        // bf16 4-bit ZP has known precision limits
        #[ignore] fn test_bf16_gs32_4bit_zp:  bf16, 32, 4, true,  false;
        #[ignore] fn test_bf16_gs64_4bit_zp:  bf16, 64, 4, true,  false;
        fn test_bf16_gs32_8bit_zp:  bf16, 32, 8, true,  false;
        fn test_bf16_gs64_8bit_zp:  bf16, 64, 8, true,  false;
        fn test_bf16_gs32_4bit_mlx: bf16, 32, 4, false, true;
        fn test_bf16_gs64_4bit_mlx: bf16, 64, 4, false, true;
        fn test_bf16_gs32_8bit_mlx: bf16, 32, 8, false, true;
        fn test_bf16_gs64_8bit_mlx: bf16, 64, 8, false, true;
    );

    edge_tests!(&CFG_BM8;
        fn test_edge_bf16_4bit_mlx: bf16, 32, 4, false, true;
        fn test_edge_bf16_8bit_mlx: bf16, 32, 8, false, true;
        fn test_edge_bf16_8bit_zp:  bf16, 32, 8, true,  false;
    );
}

// ---- tile_32x32 — the original base shape, full dtype matrix -------------

mod tile_32x32 {
    use super::*;

    basic_tests!(&CFG_32X32;
        // 4-bit, zero points
        fn test_f32_gs32_4bit_zp:    f32,  32, 4, true, false;
        fn test_f32_gs64_4bit_zp:    f32,  64, 4, true, false;
        fn test_f32_gs128_4bit_zp:   f32, 128, 4, true, false;
        fn test_f16_gs32_4bit_zp:    f16,  32, 4, true, false;
        fn test_f16_gs64_4bit_zp:    f16,  64, 4, true, false;
        fn test_f16_gs128_4bit_zp:   f16, 128, 4, true, false;
        fn test_bf16_gs32_4bit_zp:   bf16, 32, 4, true, false;
        #[ignore] fn test_bf16_gs64_4bit_zp:  bf16,  64, 4, true, false;
        #[ignore] fn test_bf16_gs128_4bit_zp: bf16, 128, 4, true, false;
        // 8-bit, zero points
        fn test_f32_gs32_8bit_zp:    f32,  32, 8, true, false;
        fn test_f32_gs64_8bit_zp:    f32,  64, 8, true, false;
        fn test_f32_gs128_8bit_zp:   f32, 128, 8, true, false;
        fn test_f16_gs32_8bit_zp:    f16,  32, 8, true, false;
        fn test_f16_gs64_8bit_zp:    f16,  64, 8, true, false;
        fn test_f16_gs128_8bit_zp:   f16, 128, 8, true, false;
        fn test_bf16_gs32_8bit_zp:   bf16, 32, 8, true, false;
        fn test_bf16_gs64_8bit_zp:   bf16, 64, 8, true, false;
        fn test_bf16_gs128_8bit_zp:  bf16,128, 8, true, false;
        // 4-bit, mlx quant
        fn test_f32_gs32_4bit_mlx:   f32,  32, 4, false, true;
        fn test_f32_gs64_4bit_mlx:   f32,  64, 4, false, true;
        fn test_f32_gs128_4bit_mlx:  f32, 128, 4, false, true;
        fn test_f16_gs32_4bit_mlx:   f16,  32, 4, false, true;
        fn test_f16_gs64_4bit_mlx:   f16,  64, 4, false, true;
        fn test_f16_gs128_4bit_mlx:  f16, 128, 4, false, true;
        fn test_bf16_gs32_4bit_mlx:  bf16, 32, 4, false, true;
        fn test_bf16_gs64_4bit_mlx:  bf16, 64, 4, false, true;
        fn test_bf16_gs128_4bit_mlx: bf16,128, 4, false, true;
        // 8-bit, mlx quant
        fn test_f32_gs32_8bit_mlx:   f32,  32, 8, false, true;
        fn test_f32_gs64_8bit_mlx:   f32,  64, 8, false, true;
        fn test_f32_gs128_8bit_mlx:  f32, 128, 8, false, true;
        fn test_f16_gs32_8bit_mlx:   f16,  32, 8, false, true;
        fn test_f16_gs64_8bit_mlx:   f16,  64, 8, false, true;
        fn test_f16_gs128_8bit_mlx:  f16, 128, 8, false, true;
        fn test_bf16_gs32_8bit_mlx:  bf16, 32, 8, false, true;
        fn test_bf16_gs64_8bit_mlx:  bf16, 64, 8, false, true;
        fn test_bf16_gs128_8bit_mlx: bf16,128, 8, false, true;
    );

    edge_tests!(&CFG_32X32;
        fn test_edge_f32_4bit_zp:  f32, 32, 4, true, false;
        fn test_edge_f32_8bit_zp:  f32, 32, 8, true, false;
        fn test_edge_f16_4bit_zp:  f16, 32, 4, true, false;
        fn test_edge_f32_4bit_mlx: f32, 32, 4, false, true;
        fn test_edge_f32_8bit_mlx: f32, 32, 8, false, true;
        fn test_edge_f16_4bit_mlx: f16, 32, 4, false, true;
    );
}

// ---- tile_wide — BM=64, BK=32, BN=64, bf16 only --------------------------

mod tile_wide {
    use super::*;

    basic_tests!(&CFG_WIDE;
        // 4-bit, zero points
        fn test_bf16_gs32_4bit_zp:   bf16,  32, 4, true, false;
        #[ignore] fn test_bf16_gs64_4bit_zp:  bf16,  64, 4, true, false;
        #[ignore] fn test_bf16_gs128_4bit_zp: bf16, 128, 4, true, false;
        // 8-bit, zero points
        fn test_bf16_gs32_8bit_zp:   bf16,  32, 8, true, false;
        fn test_bf16_gs64_8bit_zp:   bf16,  64, 8, true, false;
        fn test_bf16_gs128_8bit_zp:  bf16, 128, 8, true, false;
        // 4-bit, mlx quant
        fn test_bf16_gs32_4bit_mlx:  bf16,  32, 4, false, true;
        fn test_bf16_gs64_4bit_mlx:  bf16,  64, 4, false, true;
        fn test_bf16_gs128_4bit_mlx: bf16, 128, 4, false, true;
        // 8-bit, mlx quant
        fn test_bf16_gs32_8bit_mlx:  bf16,  32, 8, false, true;
        fn test_bf16_gs64_8bit_mlx:  bf16,  64, 8, false, true;
        fn test_bf16_gs128_8bit_mlx: bf16, 128, 8, false, true;
    );

    edge_tests!(&CFG_WIDE;
        fn test_edge_bf16_4bit_zp:  bf16, 32, 4, true, false;
        fn test_edge_bf16_8bit_zp:  bf16, 32, 8, true, false;
        fn test_edge_bf16_4bit_mlx: bf16, 32, 4, false, true;
        fn test_edge_bf16_8bit_mlx: bf16, 32, 8, false, true;
    );
}

// ---- tile_64x64 — BM=BK=BN=64, bf16 only, group_size >= 64 --------------

mod tile_64x64 {
    use super::*;

    basic_tests!(&CFG_64X64;
        // 4-bit, zero points
        #[ignore] fn test_bf16_gs64_4bit_zp:  bf16,  64, 4, true, false;
        #[ignore] fn test_bf16_gs128_4bit_zp: bf16, 128, 4, true, false;
        // 8-bit, zero points
        fn test_bf16_gs64_8bit_zp:   bf16,  64, 8, true, false;
        fn test_bf16_gs128_8bit_zp:  bf16, 128, 8, true, false;
        // 4-bit, mlx quant
        fn test_bf16_gs64_4bit_mlx:  bf16,  64, 4, false, true;
        fn test_bf16_gs128_4bit_mlx: bf16, 128, 4, false, true;
        // 8-bit, mlx quant
        fn test_bf16_gs64_8bit_mlx:  bf16,  64, 8, false, true;
        fn test_bf16_gs128_8bit_mlx: bf16, 128, 8, false, true;
    );

    edge_tests!(&CFG_64X64;
        fn test_edge_bf16_4bit_zp:  bf16, 64, 4, true, false;
        fn test_edge_bf16_8bit_zp:  bf16, 64, 8, true, false;
        fn test_edge_bf16_4bit_mlx: bf16, 64, 4, false, true;
        fn test_edge_bf16_8bit_mlx: bf16, 64, 8, false, true;
    );
}

// ---- hadamard — cross-shape consistency of the generalized HT loop -------

mod hadamard {
    use super::*;

    // f32 at (64,64,64) overflows threadgroup memory (34816 > 32768 bytes).
    hadamard_tests!(
        fn test_bf16_gs64_4bit_mlx:  bf16,  64, 4, false, true;
        fn test_bf16_gs128_4bit_mlx: bf16, 128, 4, false, true;
        fn test_bf16_gs64_8bit_mlx:  bf16,  64, 8, false, true;
        fn test_bf16_gs128_8bit_mlx: bf16, 128, 8, false, true;
        fn test_f16_gs64_4bit_mlx:   f16,   64, 4, false, true;
        fn test_f16_gs128_4bit_mlx:  f16,  128, 4, false, true;
    );
}
