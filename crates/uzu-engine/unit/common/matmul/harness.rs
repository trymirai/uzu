use futures::executor::block_on;
use num_traits::Float;

use super::Shape;
use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            kernel::{
                Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
            },
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub shape: Shape,
    pub ab_scale: f32,
    pub b_transpose: bool,
    pub enable_rht: bool,
    pub enable_bias: bool,
}

impl Case {
    pub const fn new(shape: Shape) -> Self {
        Self {
            shape,
            ab_scale: 1.0,
            b_transpose: true,
            enable_rht: false,
            enable_bias: false,
        }
    }

    pub const fn with_ab_scale(
        mut self,
        ab_scale: f32,
    ) -> Self {
        self.ab_scale = ab_scale;
        self
    }

    pub const fn with_rht(
        mut self,
        enable_rht: bool,
    ) -> Self {
        self.enable_rht = enable_rht;
        self
    }

    pub const fn with_bias(
        mut self,
        enable_bias: bool,
    ) -> Self {
        self.enable_bias = enable_bias;
        self
    }
}

pub struct Input<T: DataTypeRepr + Float> {
    pub a: Box<[T]>,
    pub b: Box<[T]>,
    pub rht_factors: Option<Box<[i32]>>,
    pub bias: Option<Box<[T]>>,
    pub case: Case,
}

pub fn deterministic_input<T: DataTypeRepr + Float>(case: Case) -> Input<T> {
    let Shape {
        m,
        k,
        n,
    } = case.shape;
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let rht_factors = (case.enable_rht && n % 32 == 0).then(|| {
        (0..n)
            .map(|i| {
                if (i % 2) == 0 {
                    1i32
                } else {
                    -1i32
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let bias = case.enable_bias.then(|| {
        (0..n).map(|i| T::from(((i % 11) as f32) * 0.05 - 0.25).unwrap()).collect::<Vec<_>>().into_boxed_slice()
    });
    Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        rht_factors,
        bias,
        case,
    }
}

pub fn run<B: Backend, T: DataTypeRepr + Float>(
    context: &B::Context,
    kernel: &mut <B::Kernels as Kernels>::MatmulKernel,
    input: &Input<T>,
    encode: impl for<'a> FnOnce(
        &mut <B::Kernels as Kernels>::MatmulKernel,
        MatmulArguments<
            'a,
            B,
            &'a B::GlobalGpuBuffer,
            &'a B::GlobalGpuBuffer,
            &'a mut B::GlobalGpuBuffer,
            &'a B::GlobalGpuBuffer,
        >,
        &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ),
) -> Vec<T> {
    let Shape {
        m,
        k,
        n,
    } = input.case.shape;
    let b_buffer = create_buffer_with_data::<B, T>(context, &input.b);
    let a_buffer = create_buffer_with_data::<B, T>(context, &input.a);
    let mut d_buffer =
        context.create_buffer_gpu(m as usize * n as usize * std::mem::size_of::<T>()).expect("create d buffer");
    let rht_buffer = input.rht_factors.as_ref().map(|factors| create_buffer_with_data::<B, i32>(context, factors));
    let bias_buffer = input.bias.as_ref().map(|bias| create_buffer_with_data::<B, T>(context, bias));

    let d_transform = MatmulDOps::<'_, B> {
        ab_scale: input.case.ab_scale,
        bias: bias_buffer.as_ref(),
        rht_factors: rht_buffer.as_ref(),
        soft_cap: None,
    };

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    encode(
        kernel,
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &a_buffer,
                offset: 0,
            },
            b: MatmulB::FullPrecision {
                b: &b_buffer,
            },
            b_leading_dimension: None,
            b_transpose: input.case.b_transpose,
            output: MatmulOutput::new(&mut d_buffer, d_transform),
            gather_indices: None,
            m,
            n,
            k,
        },
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, T>(context, &d_buffer)
}

pub fn cpu_reference<T: DataTypeRepr + Float>(input: &Input<T>) -> Vec<T> {
    let context = block_on(<Cpu as Backend>::Context::new()).expect("CPU context");
    let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("CPU MatmulKernel");
    run::<Cpu, T>(&context, &mut kernel, input, |kernel, args, command_buffer| {
        kernel.encode(args, command_buffer).expect("encode failed");
    })
}
