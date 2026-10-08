use std::mem::size_of;

use num_traits::Float;

use super::{QuantInput, pad, quant_b_variant, transpose_metadata};
use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, BufferMut,
        gpu_types::QuantizationMethod,
        kernel::matmul::{MatmulB, QuantParams, QuantParamsLayout},
    },
    data_type::DataType,
    tests::helpers::{create_buffer, create_buffer_with_data},
};

pub struct QuantBuffers<B: Backend, T: ArrayElement + Float> {
    pub w: B::GlobalBuffer,
    pub scales: B::GlobalBuffer,
    pub zp: Option<B::GlobalBuffer>,
    pub bias: Option<B::GlobalBuffer>,
    pub x: B::GlobalBuffer,
    pub prepared_a: Option<B::GlobalBuffer>,
    pub prepared_a_scales: Option<B::GlobalBuffer>,
    pub prepared_a_group_sums: Option<B::GlobalBuffer>,
    pub y: B::GlobalBuffer,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> QuantBuffers<B, T> {
    pub fn allocate(
        context: &B::Context,
        input: &QuantInput<T>,
    ) -> Self {
        let groups = input.k.div_ceil(input.group_size);
        let params_layout = input.params_layout;
        let params = QuantParams::new(params_layout, input.n, groups);
        let metadata_elements = params.scale_shape().into_iter().product::<u32>() as usize;
        let zero_point_bytes = params.zero_point_shape(input.mode).into_iter().product::<u32>() as usize;
        let mut buffers = Self {
            w: create_buffer_with_data::<B, u32>(context, &input.weights_for_upload()),
            scales: create_buffer_with_data::<B, T>(context, &pad(&input.scales, metadata_elements)),
            zp: input
                .zero_points
                .as_ref()
                .map(|zero_points| create_buffer_with_data::<B, u8>(context, &pad(zero_points, zero_point_bytes))),
            bias: input
                .biases
                .as_ref()
                .map(|biases| create_buffer_with_data::<B, T>(context, &pad(biases, metadata_elements))),
            x: create_buffer_with_data::<B, T>(context, &input.x),
            prepared_a: input
                .prepared_a
                .as_ref()
                .map(|prepared| create_buffer_with_data::<B, i8>(context, &prepared.values)),
            prepared_a_scales: input
                .prepared_a
                .as_ref()
                .map(|prepared| create_buffer_with_data::<B, f32>(context, &prepared.scales)),
            prepared_a_group_sums: input
                .prepared_a
                .as_ref()
                .filter(|prepared| !prepared.group_sums.is_empty())
                .map(|prepared| create_buffer_with_data::<B, i32>(context, &prepared.group_sums)),
            y: create_buffer::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        };
        if params_layout == QuantParamsLayout::GroupOutput {
            buffers.transpose_quant_params(input);
        }
        buffers
    }

    pub fn matmul_b<'a>(
        &'a self,
        input: &QuantInput<T>,
    ) -> MatmulB<&'a B::GlobalBuffer> {
        quant_b_variant(&self.w, &self.scales, self.zp.as_ref(), self.bias.as_ref(), input.params_layout, input)
    }

    fn transpose_quant_params(
        &mut self,
        input: &QuantInput<T>,
    ) {
        let columns = input.n;
        let groups = input.k.div_ceil(input.group_size);
        let value_bits = size_of::<T>() as u32 * u8::BITS;
        transpose_metadata(self.scales.as_slice_mut(), columns, groups, value_bits);
        match input.quant_method {
            QuantizationMethod::ScaleBias => {
                transpose_metadata(
                    self.bias.as_mut().expect("bias buffer").as_slice_mut(),
                    columns,
                    groups,
                    value_bits,
                );
            },
            QuantizationMethod::ScaleZeroPoint => {
                let correction_bits = DataType::from(input.mode).size_in_bits() as u32;
                transpose_metadata(
                    self.zp.as_mut().expect("zp buffer").as_slice_mut(),
                    columns,
                    groups,
                    correction_bits,
                );
            },
            QuantizationMethod::ScaleSymmetric => {},
        }
    }
}
