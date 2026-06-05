#[cfg(metal_backend)]
use backend_uzu::backends::metal::{GemmDispatchPath, Metal, MetalContext};
use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
};
use half::f16;
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

pub const LLOYD_MAX_CODEBOOK_SIZE: usize = 16;

pub fn lloyd_max_codebook() -> [f16; LLOYD_MAX_CODEBOOK_SIZE] {
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

pub const BIAS_CODEBOOK_SIZE: usize = 16;

pub fn lloyd_max_bias_codebook() -> [f16; BIAS_CODEBOOK_SIZE] {
    [
        -0.045f32, -0.039, -0.033, -0.026, -0.020, -0.013, -0.007, 0.0, 0.007, 0.013, 0.020, 0.026, 0.033, 0.039,
        0.045, 0.052,
    ]
    .map(f16::from_f32)
}

pub struct LloydMaxQuantInput<T: ArrayElement + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub codebook: [f16; LLOYD_MAX_CODEBOOK_SIZE],
    pub bias_indices: Vec<u8>,
    pub bias_codebook: [f16; BIAS_CODEBOOK_SIZE],
    pub x: Vec<T>,
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub group_size: u32,
    pub mode: QuantizationMode,
}

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::I8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
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

impl<T: ArrayElement + Float> QuantInput<T> {
    pub fn new(
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

        let w_packed: Vec<u32> = (0..n * k * bits as usize / 32).map(|_| rng.random_range(0..u32::MAX)).collect();
        let scales: Vec<T> =
            (0..n * num_groups_k).map(|_| T::from(rng.random_range(0.01f32..0.3f32)).unwrap()).collect();
        let x: Vec<T> = (0..m * k).map(|_| T::from(rng.random_range(-0.3f32..0.3f32)).unwrap()).collect();

        let zp_stride = if bits == 4 {
            num_groups_k.div_ceil(2)
        } else {
            num_groups_k
        };
        let (zero_points, biases) = match quant_method {
            QuantizationMethod::ScaleBias => (
                None,
                Some((0..n * num_groups_k).map(|_| T::from(rng.random_range(-0.03f32..0.03f32)).unwrap()).collect()),
            ),
            QuantizationMethod::ScaleZeroPoint => {
                (Some((0..n * zp_stride).map(|_| rng.random_range(0u8..u8::MAX)).collect()), None)
            },
            QuantizationMethod::ScaleSymmetric => (None, None),
            QuantizationMethod::LloydMax => unreachable!("use Lloyd-Max-specific test input"),
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

impl<T: ArrayElement + Float> LloydMaxQuantInput<T> {
    pub fn new(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
    ) -> Self {
        let num_groups_k = k / group_size as usize;
        let w_packed = deterministic_packed_u4_weights(n, k);
        let scales = lloyd_max_scales(n, num_groups_k);
        let bias_indices = lloyd_max_bias_indices(n, num_groups_k);
        let x = lloyd_max_input_values(m, k);
        Self {
            w_packed,
            scales,
            codebook: lloyd_max_codebook(),
            bias_indices,
            bias_codebook: lloyd_max_bias_codebook(),
            x,
            k: k as u32,
            n: n as u32,
            m: m as u32,
            group_size,
            mode: QuantizationMode::U4,
        }
    }
}

pub struct LloydMaxQuantBuffers<B: Backend, T: ArrayElement + Float> {
    pub w: Allocation<B>,
    pub scales: Allocation<B>,
    pub codebook: Allocation<B>,
    pub bias_indices: Allocation<B>,
    pub bias_codebook: Allocation<B>,
    pub x: Allocation<B>,
    pub y: Allocation<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> LloydMaxQuantBuffers<B, T> {
    pub fn allocate(
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

pub fn lloyd_max_quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
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

pub fn quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> MatmulArguments<'a, B> {
    let b_variant = match input.quant_method {
        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            biases: buffers.bias.as_ref().expect("bias buffer"),
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            zero_points: buffers.zp.as_ref().expect("zp buffer"),
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::LloydMax => unreachable!("use Lloyd-Max-specific matmul arguments"),
    };
    MatmulArguments {
        a: &buffers.x,
        a_offset: 0,
        b: b_variant,
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

pub fn run_quant_cpu<T: ArrayElement + Float>(input: &QuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut buffers = QuantBuffers::<Cpu, T>::allocate(&context, input);
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    matmul.encode(quant_arguments(&mut buffers, input), &mut encoder).expect("encode cpu quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&buffers.y)
}

pub fn run_lloyd_max_cpu<T: ArrayElement + Float>(input: &LloydMaxQuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut buffers = LloydMaxQuantBuffers::<Cpu, T>::allocate(&context, input);
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    matmul.encode(lloyd_max_quant_arguments(&mut buffers, input), &mut encoder).expect("encode CPU Lloyd-Max quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&buffers.y)
}

#[cfg(metal_backend)]
pub fn run_quant_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    path: Option<GemmDispatchPath>,
) -> Vec<T> {
    let mut buffers = QuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    let args = quant_arguments(&mut buffers, input);
    match path {
        None => matmul.encode(args, &mut encoder).expect("matmul encode failed"),
        Some(gemm_path) => {
            matmul.gemm.encode_dispatch_path(args, gemm_path, &mut encoder).expect("gemm encode_dispatch_path failed")
        },
    }
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&buffers.y)
}

#[cfg(metal_backend)]
pub fn run_lloyd_max_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &LloydMaxQuantInput<T>,
) -> Vec<T> {
    let mut buffers = LloydMaxQuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    matmul.encode(lloyd_max_quant_arguments(&mut buffers, input), &mut encoder).expect("matmul encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&buffers.y)
}
