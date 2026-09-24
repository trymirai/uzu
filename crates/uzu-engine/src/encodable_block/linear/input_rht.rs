use std::mem::size_of;

use super::{LinearInput, LinearInputPreparation};
use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{ActivationTransform, matmul::ActivationFormat},
    },
    data_type::DataType,
};

pub(super) struct InputRht<B: Backend> {
    rht_signs: Allocation<B>,
    rht: ActivationTransform<B>,
    quantizer: Option<ActivationTransform<B>>,
}

impl<B: Backend> InputRht<B> {
    pub(super) fn new(
        context: &B::Context,
        data_type: DataType,
        preparation: LinearInputPreparation<B>,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        let LinearInputPreparation {
            rht_signs,
            a8_plan,
        } = preparation;
        let rht = ActivationTransform::input_rht(context, data_type, in_place)?;
        let quantizer = a8_plan
            .map(|plan| {
                ActivationTransform::quantize(context, data_type, plan.activation_group_size, plan.sum_group_size)
            })
            .transpose()?;

        Ok(Self {
            rht_signs,
            rht,
            quantizer,
        })
    }

    pub(super) fn prepare(
        &self,
        input: &Allocation<B>,
        batch_dim: u32,
        format: ActivationFormat,
        encoder: &mut Encoder<B>,
    ) -> Result<LinearInput<B>, B::Error> {
        if format == ActivationFormat::Int8
            && let Some(quantizer) = &self.quantizer
        {
            let input_dim = self.input_dim();
            let groups_per_row = input_dim.div_ceil(quantizer.activation_group_size());
            let mut values = encoder.allocate_scratch(size_for_shape(&[batch_dim, input_dim], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::F32))?;
            let mut group_sums = quantizer
                .sum_group_size()
                .map(|group_size| {
                    encoder
                        .allocate_scratch(size_for_shape(&[batch_dim, input_dim.div_ceil(group_size)], DataType::I32))
                })
                .transpose()?;
            quantizer.encode_quantize(
                input,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                &self.rht_signs,
                batch_dim,
                input_dim,
                encoder,
            );

            return Ok(LinearInput::Int8Symmetric {
                values,
                scales,
                group_sums,
                group_size: quantizer.activation_group_size(),
            });
        }

        let input_dim = self.input_dim();
        let mut transformed = encoder.allocate_scratch(input.size())?;
        self.rht.encode_fp(input, &mut transformed, &self.rht_signs, batch_dim, input_dim, encoder);
        Ok(LinearInput::FullPrecision(transformed))
    }

    pub(super) fn prepare_in_place(
        &self,
        mut input: Allocation<B>,
        batch_dim: u32,
        format: ActivationFormat,
        encoder: &mut Encoder<B>,
    ) -> Result<LinearInput<B>, B::Error> {
        if format == ActivationFormat::Int8 && self.quantizer.is_some() {
            return self.prepare(&input, batch_dim, format, encoder);
        }

        let input_dim = self.input_dim();
        self.rht.encode_fp_in_place(&mut input, &self.rht_signs, batch_dim, input_dim, encoder);
        Ok(LinearInput::FullPrecision(input))
    }

    fn input_dim(&self) -> u32 {
        (self.rht_signs.size() / size_of::<i32>()) as u32
    }
}
