use std::mem::size_of_val;

use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[cfg(backend = "metal")]
use super::harness::TestDispatch;
#[cfg(backend = "metal")]
use crate::backends::metal::{Metal, MetalContext};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationTransform, Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting},
            },
        },
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

pub struct PreparedInt8A {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_sums: Vec<i32>,
    pub activation_scale_group_size: u32,
}

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
    pub signed_codes: bool,
    pub prepared_a: Option<PreparedInt8A>,
}

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::U8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
}

impl<T: ArrayElement + Float> QuantInput<T> {
    pub fn new(
        m: u32,
        k: u32,
        n: u32,
        group_size: u32,
        bits: u32,
        quant_method: QuantizationMethod,
        seed: u64,
    ) -> Self {
        let num_groups_k = k.div_ceil(group_size);
        let mut rng = SmallRng::seed_from_u64(seed);

        let w_packed: Vec<u32> = (0..(n * k * bits / 32) as usize).map(|_| rng.random_range(0..u32::MAX)).collect();
        let scales: Vec<T> =
            (0..(n * num_groups_k) as usize).map(|_| T::from(rng.random_range(0.01f32..0.3f32)).unwrap()).collect();
        let x: Vec<T> = (0..(m * k) as usize).map(|_| T::from(rng.random_range(-0.3f32..0.3f32)).unwrap()).collect();

        let zp_stride = if bits == 4 {
            num_groups_k.div_ceil(2)
        } else {
            num_groups_k
        };
        let (zero_points, biases) = match quant_method {
            QuantizationMethod::ScaleBias => (
                None,
                Some(
                    (0..(n * num_groups_k) as usize)
                        .map(|_| T::from(rng.random_range(-0.03f32..0.03f32)).unwrap())
                        .collect(),
                ),
            ),
            QuantizationMethod::ScaleZeroPoint => {
                (Some((0..(n * zp_stride) as usize).map(|_| rng.random_range(0u8..u8::MAX)).collect()), None)
            },
            QuantizationMethod::ScaleSymmetric => (None, None),
        };

        Self {
            w_packed,
            scales,
            zero_points,
            biases,
            x,
            k,
            n,
            m,
            group_size,
            quant_method,
            mode: mode_for_bits(bits),
            signed_codes: false,
            prepared_a: None,
        }
    }

    pub fn with_signed_weight_codes(mut self) -> Self {
        self.signed_codes = true;
        self
    }

    pub fn with_prepared_a(
        mut self,
        activation_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> Self {
        self.signed_codes = true;
        let rows = self.m;
        let columns = self.k;
        assert!(columns.is_multiple_of(activation_group_size));
        if let Some(group_size) = sum_group_size {
            assert!(columns.is_multiple_of(group_size));
        }
        let context = <Cpu as Backend>::Context::new().expect("CPU context");
        let input = alloc_allocation_with_data::<Cpu, T>(&context, &self.x);
        let factors = alloc_allocation_with_data::<Cpu, i32>(&context, &vec![1; columns as usize]);
        let element_count = rows * columns;
        let mut values = alloc_allocation::<Cpu, i8>(&context, element_count as usize);
        let mut scales = alloc_allocation::<Cpu, f32>(&context, (element_count / activation_group_size) as usize);
        let mut group_sums = sum_group_size
            .map(|group_size| alloc_allocation::<Cpu, i32>(&context, (element_count / group_size) as usize));
        let transform =
            ActivationTransform::<Cpu>::quantize(&context, T::data_type(), activation_group_size, sum_group_size)
                .expect("CPU activation quantization transform");
        let mut encoder = Encoder::<Cpu>::new(&context).expect("CPU encoder");
        transform.encode_quantize(
            &input,
            &mut values,
            &mut scales,
            group_sums.as_mut(),
            &factors,
            rows,
            columns,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("CPU activation quantization");

        self.prepared_a = Some(PreparedInt8A {
            values: allocation_to_vec(&values),
            scales: allocation_to_vec(&scales),
            group_sums: group_sums.map_or_else(Vec::new, |sums| allocation_to_vec(&sums)),
            activation_scale_group_size: activation_group_size,
        });
        self
    }

    pub(crate) fn weights_for_upload(&self) -> Vec<u32> {
        let mut words = self.w_packed.clone();
        let sign_flip_mask = self.signed_codes.then(|| self.mode.weight_codes_sign_flip_mask()).flatten();
        if let Some(mask) = sign_flip_mask {
            let broadcast_mask = u32::from(mask) * 0x0101_0101;
            words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        }
        words
    }

    pub(crate) fn weight_buffer_bytes(&self) -> usize {
        size_of_val(self.w_packed.as_slice())
            + size_of_val(self.scales.as_slice())
            + self.biases.as_ref().map_or(0, |biases| size_of_val(biases.as_slice()))
            + self.zero_points.as_ref().map_or(0, |zero_points| size_of_val(zero_points.as_slice()))
    }
}

pub struct QuantBuffers<B: Backend, T: ArrayElement + Float> {
    pub w: Allocation<B>,
    pub scales: Allocation<B>,
    pub zp: Option<Allocation<B>>,
    pub bias: Option<Allocation<B>>,
    pub x: Allocation<B>,
    pub prepared_a: Option<Allocation<B>>,
    pub prepared_a_scales: Option<Allocation<B>>,
    pub prepared_a_group_sums: Option<Allocation<B>>,
    pub y: Allocation<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> QuantBuffers<B, T> {
    pub fn allocate(
        context: &B::Context,
        input: &QuantInput<T>,
    ) -> Self {
        Self {
            w: alloc_allocation_with_data::<B, u32>(context, &input.weights_for_upload()),
            scales: alloc_allocation_with_data::<B, T>(context, &input.scales),
            zp: input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<B, u8>(context, zp)),
            bias: input.biases.as_ref().map(|b| alloc_allocation_with_data::<B, T>(context, b)),
            x: alloc_allocation_with_data::<B, T>(context, &input.x),
            prepared_a: input
                .prepared_a
                .as_ref()
                .map(|prepared| alloc_allocation_with_data::<B, i8>(context, &prepared.values)),
            prepared_a_scales: input
                .prepared_a
                .as_ref()
                .map(|prepared| alloc_allocation_with_data::<B, f32>(context, &prepared.scales)),
            prepared_a_group_sums: input
                .prepared_a
                .as_ref()
                .filter(|prepared| !prepared.group_sums.is_empty())
                .map(|prepared| alloc_allocation_with_data::<B, i32>(context, &prepared.group_sums)),
            y: alloc_allocation::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        }
    }
}

pub fn quant_b_variant<'a, B: Backend, T: ArrayElement + Float>(
    w: &'a Allocation<B>,
    scales: &'a Allocation<B>,
    zero_points: Option<&'a Allocation<B>>,
    biases: Option<&'a Allocation<B>>,
    input: &QuantInput<T>,
) -> MatmulB<'a, B> {
    let signed_codes = input.signed_codes;
    match input.quant_method {
        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
            b: w,
            scales,
            biases: biases.expect("bias buffer"),
            mode: input.mode,
            group_size: input.group_size,
            signed_codes,
        },
        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
            b: w,
            scales,
            zero_points: zero_points.expect("zp buffer"),
            mode: input.mode,
            group_size: input.group_size,
            signed_codes,
        },
        QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
            b: w,
            scales,
            mode: input.mode,
            group_size: input.group_size,
            signed_codes,
        },
    }
}

pub fn quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> MatmulArguments<'a, 'a, 'a, B> {
    let QuantBuffers {
        w,
        scales,
        zp,
        bias,
        x,
        prepared_a,
        prepared_a_scales,
        prepared_a_group_sums,
        y,
        ..
    } = buffers;
    let b = quant_b_variant(w, scales, zp.as_ref(), bias.as_ref(), input);
    let a = match &input.prepared_a {
        Some(_) => MatmulA::Int8Symmetric {
            values: prepared_a.as_ref().expect("prepared activation buffer"),
            scales: prepared_a_scales.as_ref().expect("prepared activation scales"),
            // Symmetric weights carry no correction term, so the GEMM never reads these.
            group_sums: (input.quant_method != QuantizationMethod::ScaleSymmetric)
                .then(|| prepared_a_group_sums.as_ref().expect("prepared activation row sums")),
            group_size: input.prepared_a.as_ref().expect("prepared activation metadata").activation_scale_group_size,
        },
        None => MatmulA::FullPrecision {
            values: x,
            offset: 0,
        },
    };
    MatmulArguments {
        a,
        b,
        b_leading_dimension: None,
        b_transpose: true,
        d: y,
        d_transform: MatmulDOps::none(),
        routing: MatmulRouting::Dense,
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

#[cfg(backend = "metal")]
pub fn run_quant_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    dispatch: TestDispatch,
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
    if let Some(engine) = dispatch {
        matmul.gemm.encode_with_engine(args, engine, &mut encoder).expect("forced GEMM engine encode failed");
    } else {
        matmul.encode(args, &mut encoder).expect("matmul encode failed");
    }
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&buffers.y)
}
