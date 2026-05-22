//! Shared scaffolding for quantized matmul tests and benches.
//!
//! Provides `QuantInput<T>` (deterministic or random data), `QuantBuffers<B,T>`
//! (per-backend allocations), and `quant_arguments` (the `MatmulArguments`
//! builder). Tests use `run_quant_cpu` + `run_quant_metal` for one-shot encodes;
//! benches use `QuantBuffers::allocate` + `quant_arguments` in a custom-iter
//! loop.

use std::collections::HashSet;

#[cfg(metal_backend)]
use backend_uzu::backends::metal::{MatmulDispatchPath, Metal, MetalContext};
use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::super::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec};

pub struct QuantInput<T: ArrayElement + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub x: Vec<T>,
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub group_size: u32,
    pub quant_method: QuantizationMethod,
    pub mode: QuantizationMode,
}

fn pack_weights_u32(
    values: &[u8],
    bits: u32,
) -> Vec<u32> {
    let pack_factor = if bits == 4 { 8 } else { 4 };
    values
        .chunks(pack_factor)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, &v) in chunk.iter().enumerate() {
                let masked = if bits == 4 { (v & 0xF) as u32 } else { v as u32 };
                word |= masked << (i * (32 / pack_factor));
            }
            word
        })
        .collect()
}

fn pack_zero_points(
    values: &[u8],
    bits: u32,
) -> Vec<u8> {
    if bits == 4 {
        values
            .chunks(2)
            .map(|chunk| {
                let lo = chunk[0] & 0x0F;
                let hi = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
                lo | (hi << 4)
            })
            .collect()
    } else {
        values.to_vec()
    }
}

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::I8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
}

fn build_zp_packed(
    zp_raw: &[u8],
    n: usize,
    num_groups_k: usize,
    zp_stride: usize,
    bits: u32,
) -> Vec<u8> {
    let mut zp_packed: Vec<u8> = Vec::with_capacity(n * zp_stride);
    for j in 0..n {
        let row = &zp_raw[j * num_groups_k..(j + 1) * num_groups_k];
        let mut packed_row = pack_zero_points(row, bits);
        packed_row.resize(zp_stride, 0);
        zp_packed.extend_from_slice(&packed_row);
    }
    zp_packed
}

impl<T: ArrayElement + Float> QuantInput<T> {
    /// Deterministic, reproducible input — small integer-friendly values for
    /// parity testing.
    pub fn deterministic(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
        bits: u32,
        quant_method: QuantizationMethod,
    ) -> Self {
        let num_groups_k = k.div_ceil(group_size as usize);
        let max_val: u8 = if bits == 4 { 15 } else { 255 };

        let weights_raw: Vec<u8> =
            (0..n * k).map(|i| ((i.wrapping_mul(7).wrapping_add(1)) % (max_val as usize + 1)) as u8).collect();
        let w_packed = pack_weights_u32(&weights_raw, bits);

        let scales: Vec<T> = (0..n * num_groups_k)
            .map(|i| {
                let (j, g) = (i / num_groups_k, i % num_groups_k);
                T::from(0.5 + 0.1 * ((j + g) % 5) as f32).unwrap()
            })
            .collect();

        let zp_stride = if bits == 4 { num_groups_k.div_ceil(2) } else { num_groups_k };

        let (zero_points, biases) = match quant_method {
            QuantizationMethod::ScaleZeroPoint => {
                let zp_raw: Vec<u8> = (0..n * num_groups_k)
                    .map(|i| {
                        let (j, g) = (i / num_groups_k, i % num_groups_k);
                        ((j * 2 + g * 3) % (max_val as usize + 1)) as u8
                    })
                    .collect();
                (Some(build_zp_packed(&zp_raw, n, num_groups_k, zp_stride, bits)), None)
            },
            QuantizationMethod::ScaleBias => {
                let biases: Vec<T> = (0..n * num_groups_k)
                    .map(|i| {
                        let (j, g) = (i / num_groups_k, i % num_groups_k);
                        T::from(0.01 * ((j + g * 2) % 7) as f32).unwrap()
                    })
                    .collect();
                (None, Some(biases))
            },
        };

        let x: Vec<T> = (0..m * k).map(|i| T::from(0.1 * f32::sin(i as f32 * 0.05) + 0.5).unwrap()).collect();

        Self {
            w_packed,
            scales,
            zero_points,
            biases,
            x,
            k: k as u32,
            n: n as u32,
            m: m as u32,
            group_size,
            quant_method,
            mode: mode_for_bits(bits),
        }
    }

    /// Seeded random input — for benchmarks where data values don't affect timing.
    pub fn random(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
        bits: u32,
        quant_method: QuantizationMethod,
        seed: u64,
    ) -> Self {
        let num_groups_k = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(seed);

        let w_packed: Vec<u32> =
            (0..n * k * bits as usize / 32).map(|_| rng.random_range(0..u32::MAX)).collect();
        let scales: Vec<T> = (0..n * num_groups_k)
            .map(|_| T::from(rng.random_range(0.01f32..1.0f32)).unwrap())
            .collect();
        let x: Vec<T> =
            (0..m * k).map(|_| T::from(rng.random_range(-1.0f32..1.0f32)).unwrap()).collect();

        let zp_stride = if bits == 4 { num_groups_k.div_ceil(2) } else { num_groups_k };
        let (zero_points, biases) = match quant_method {
            QuantizationMethod::ScaleZeroPoint => (
                Some((0..n * zp_stride).map(|_| rng.random_range(0u8..u8::MAX)).collect()),
                None,
            ),
            QuantizationMethod::ScaleBias => (
                None,
                Some((0..n * num_groups_k).map(|_| T::from(rng.random_range(-0.1f32..0.1f32)).unwrap()).collect()),
            ),
        };

        Self {
            w_packed,
            scales,
            zero_points,
            biases,
            x,
            k: k as u32,
            n: n as u32,
            m: m as u32,
            group_size,
            quant_method,
            mode: mode_for_bits(bits),
        }
    }
}

pub struct QuantBuffers<B: Backend, T: ArrayElement + Float> {
    pub w: Allocation<B>,
    pub scales: Allocation<B>,
    pub zp: Option<Allocation<B>>,
    pub bias: Option<Allocation<B>>,
    pub x: Allocation<B>,
    pub y: Allocation<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> QuantBuffers<B, T> {
    pub fn allocate(
        context: &B::Context,
        input: &QuantInput<T>,
    ) -> Self {
        Self {
            w: alloc_allocation_with_data::<B, u32>(context, &input.w_packed),
            scales: alloc_allocation_with_data::<B, T>(context, &input.scales),
            zp: input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<B, u8>(context, zp)),
            bias: input.biases.as_ref().map(|b| alloc_allocation_with_data::<B, T>(context, b)),
            x: alloc_allocation_with_data::<B, T>(context, &input.x),
            y: alloc_allocation::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        }
    }
}

/// Build `MatmulArguments` referencing the given buffers. The lifetime of the
/// returned args is tied to `buffers` (immutable for read-only buffers, mutable
/// for `y`).
pub fn quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> MatmulArguments<'a, B> {
    let b_variant = match input.quant_method {
        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            zero_points: buffers.zp.as_ref().expect("zp buffer"),
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            biases: buffers.bias.as_ref().expect("bias buffer"),
            mode: input.mode,
            group_size: input.group_size,
        },
    };
    MatmulArguments {
        a: &buffers.x,
        a_offset: 0,
        b: b_variant,
        b_offset: 0,
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.y,
        d_transform: HashSet::new(),
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

/// Single-shot encode through the CPU backend; returns the result vector.
pub fn run_quant_cpu<T: ArrayElement + Float>(input: &QuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut buffers = QuantBuffers::<Cpu, T>::allocate(&context, input);
    let mut matmul = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulCpuKernel");
    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    matmul.encode(quant_arguments(&mut buffers, input), &mut encoder).expect("encode cpu quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&buffers.y)
}

/// Single-shot encode through the Metal backend with an explicit dispatch path.
#[cfg(metal_backend)]
pub fn run_quant_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    path: MatmulDispatchPath,
) -> Vec<T> {
    let mut buffers = QuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type())
        .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    matmul
        .encode_with_path(quant_arguments(&mut buffers, input), &mut encoder, path)
        .expect("encode metal quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&buffers.y)
}
