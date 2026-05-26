#[cfg(metal_backend)]
use backend_uzu::backends::metal::{GemmDispatchPath, Metal, MetalContext};
use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
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

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::I8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
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
            QuantizationMethod::ScaleZeroPoint => {
                (Some((0..n * zp_stride).map(|_| rng.random_range(0u8..u8::MAX)).collect()), None)
            },
            QuantizationMethod::ScaleBias => (
                None,
                Some((0..n * num_groups_k).map(|_| T::from(rng.random_range(-0.03f32..0.03f32)).unwrap()).collect()),
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
        d_transform: MatmulDOps::none(),
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

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

#[cfg(metal_backend)]
pub fn run_quant_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    path: Option<GemmDispatchPath>,
) -> Vec<T> {
    let mut buffers = QuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type())
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
