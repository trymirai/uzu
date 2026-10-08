use std::mem::size_of_val;

use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::{PreparedInt8A, mode_for_bits};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationQuantization, ActivationTransform,
                matmul::{Int8CodeLayout, QuantParamsLayout},
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
};

#[derive(Clone)]
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
    pub params_layout: QuantParamsLayout,
    pub signed_codes: bool,
    pub prepared_a: Option<PreparedInt8A>,
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

        let w_packed: Vec<u32> = (0..(n as usize * k as usize * bits as usize).div_ceil(32))
            .map(|_| rng.random_range(0..u32::MAX))
            .collect();
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
            params_layout: QuantParamsLayout::OutputGroup,
            signed_codes: false,
            prepared_a: None,
        }
    }

    pub fn with_signed_weight_codes(mut self) -> Self {
        self.signed_codes = true;
        self
    }

    pub fn with_group_output(mut self) -> Self {
        self.params_layout = QuantParamsLayout::GroupOutput;
        self
    }

    pub fn with_prepared_a(
        self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> Self {
        let code_layout = Int8CodeLayout::for_right_bits(DataType::from(self.mode).size_in_bits() as u32)
            .expect("W4/W8 quantization");
        self.with_prepared_a_layout(activation_scale_group_size, sum_group_size, code_layout)
    }

    pub fn with_prepared_a_and_reference(
        self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> (Self, Self) {
        let code_layout = Int8CodeLayout::for_right_bits(DataType::from(self.mode).size_in_bits() as u32)
            .expect("W4/W8 quantization");
        let reference = self.clone().with_prepared_a_layout(
            activation_scale_group_size,
            sum_group_size,
            Int8CodeLayout::Sequential,
        );
        let actual = self.with_prepared_a_layout(activation_scale_group_size, sum_group_size, code_layout);
        (actual, reference)
    }

    fn with_prepared_a_layout(
        mut self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
        code_layout: Int8CodeLayout,
    ) -> Self {
        self.signed_codes = self.mode != QuantizationMode::U4;
        let rows = self.m;
        let columns = self.k;
        assert!(columns.is_multiple_of(activation_scale_group_size));
        if let Some(group_size) = sum_group_size {
            assert!(columns.is_multiple_of(group_size));
        }
        let context = <Cpu as Backend>::Context::new().expect("CPU context");
        let input = create_buffer_with_data::<Cpu, T>(&context, &self.x);
        let factors = create_buffer_with_data::<Cpu, i32>(&context, &vec![1; columns as usize]);
        let element_count = rows * columns;
        let mut values = create_buffer::<Cpu, i8>(&context, element_count as usize);
        let mut scales = create_buffer::<Cpu, f32>(&context, (element_count / activation_scale_group_size) as usize);
        let mut group_sums =
            sum_group_size.map(|group_size| create_buffer::<Cpu, i32>(&context, (element_count / group_size) as usize));
        let quantization = ActivationQuantization::new(
            activation_scale_group_size,
            sum_group_size.unwrap_or(activation_scale_group_size),
            sum_group_size.is_some(),
            code_layout,
        )
        .expect("supported activation quantization");
        let transform = ActivationTransform::<Cpu>::quantize(&context, T::data_type(), quantization)
            .expect("CPU activation quantization transform");
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        transform.encode_quantize(
            &input,
            &mut values,
            &mut scales,
            group_sums.as_mut(),
            &factors,
            rows,
            columns,
            &mut command_buffer,
        );
        command_buffer.end_encoding().submit().wait_until_completed().expect("CPU activation quantization");

        self.prepared_a = Some(PreparedInt8A {
            values: buffer_to_vec(&values),
            scales: buffer_to_vec(&scales),
            group_sums: group_sums.map_or_else(Vec::new, |sums| buffer_to_vec(&sums)),
            quantization,
        });
        self
    }

    pub fn weights_for_upload(&self) -> Vec<u32> {
        let mut words = self.w_packed.clone();
        let sign_flip_mask = self.signed_codes.then(|| self.mode.weight_codes_sign_flip_mask()).flatten();
        if let Some(mask) = sign_flip_mask {
            let broadcast_mask = u32::from(mask) * 0x0101_0101;
            words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        }
        words
    }

    pub fn weight_buffer_bytes(&self) -> usize {
        size_of_val(self.w_packed.as_slice())
            + size_of_val(self.scales.as_slice())
            + self.biases.as_ref().map_or(0, |biases| size_of_val(biases.as_slice()))
            + self.zero_points.as_ref().map_or(0, |zero_points| size_of_val(zero_points.as_slice()))
    }
}
