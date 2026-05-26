#[cfg(metal_backend)]
use backend_uzu::backends::metal::{GemmDispatchPath, Metal, MetalContext};
use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Allocation, AllocationType, AsBufferRangeRef, Backend, Buffer, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
};
use num_traits::Float;

use super::{
    super::helpers::{alloc_allocation_with_data, allocation_to_vec},
    Shape,
};

#[cfg(metal_backend)]
pub type MetalMatmulKernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel;

#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub shape: Shape,
    pub ab_scale: f32,
    pub accumulate: bool,
    pub b_transpose: bool,
}

impl Case {
    pub const fn new(shape: Shape) -> Self {
        Self {
            shape,
            ab_scale: 1.0,
            accumulate: false,
            b_transpose: true,
        }
    }

    pub const fn with_ab_scale(
        mut self,
        ab_scale: f32,
    ) -> Self {
        self.ab_scale = ab_scale;
        self
    }

    pub const fn with_accumulate(
        mut self,
        accumulate: bool,
    ) -> Self {
        self.accumulate = accumulate;
        self
    }
}

pub struct Input<T: ArrayElement + Float> {
    pub a: Box<[T]>,
    pub b: Box<[T]>,
    pub d_prefill: Option<Box<[T]>>,
    pub case: Case,
}

pub fn deterministic_input<T: ArrayElement + Float>(case: Case) -> Input<T> {
    let Shape {
        m,
        k,
        n,
    } = case.shape;
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let d_prefill = case.accumulate.then(|| {
        (0..m * n).map(|i| T::from(((i % 7) as f32) * 0.03 - 0.09).unwrap()).collect::<Vec<_>>().into_boxed_slice()
    });
    Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        d_prefill,
        case,
    }
}

fn run<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    kernel: &mut <B::Kernels as ManualKernels>::MatmulKernel,
    input: &Input<T>,
    encode: impl FnOnce(&mut <B::Kernels as ManualKernels>::MatmulKernel, MatmulArguments<B>, &mut Encoder<B>),
) -> Vec<T> {
    let Shape {
        m,
        k,
        n,
    } = input.case.shape;
    let b_array = if input.case.b_transpose {
        context.create_array_from(&[n, k], &input.b)
    } else {
        context.create_array_from(&[k, n], &input.b)
    };
    let a_allocation = alloc_allocation_with_data::<B, T>(context, &input.a);
    let mut d_allocation = if let Some(ref prefill) = input.d_prefill {
        alloc_allocation_with_data::<B, T>(context, prefill)
    } else {
        context
            .create_allocation(m * n * std::mem::size_of::<T>(), AllocationType::Global)
            .expect("create d allocation")
    };

    let d_transform = MatmulDOps::<'_, B> {
        ab_scale: input.case.ab_scale,
        accumulate: input.case.accumulate,
        bias: None,
        rht_factors: None,
    };

    let mut encoder = Encoder::new(context).expect("encoder");
    encode(
        kernel,
        MatmulArguments::<'_, B> {
            a: &a_allocation,
            a_offset: 0,
            b: MatmulB::FullPrecision {
                b: b_array.allocation(),
            },
            b_offset: 0,
            b_leading_dimension: None,
            b_transpose: input.case.b_transpose,
            d: &mut d_allocation,
            d_transform,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, T>(&d_allocation)
}

pub fn cpu_reference<T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let mut kernel = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("CPU MatmulKernel");
    run::<Cpu, T>(&context, &mut kernel, input, |kernel, args, encoder| {
        kernel.encode(args, encoder).expect("encode failed");
    })
}

#[cfg(metal_backend)]
pub fn run_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    kernel: &mut MetalMatmulKernel,
    input: &Input<T>,
    path: Option<GemmDispatchPath>,
) -> Vec<T> {
    run::<Metal, T>(context, kernel, input, |kernel, args, encoder| match path {
        None => kernel.encode(args, encoder).expect("matmul encode failed"),
        Some(gemm_path) => {
            kernel.gemm.encode_dispatch_path(args, gemm_path, encoder).expect("gemm encode_dispatch_path failed")
        },
    })
}
