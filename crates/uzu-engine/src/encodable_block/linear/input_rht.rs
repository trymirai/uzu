use std::mem::size_of;

use super::{LinearInput, LinearInputPreparation};
use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{ActivationTransform, matmul::ActivationFormat},
    },
    data_type::DataType,
};

pub(super) struct InputRht<B: Backend> {
    rht_signs: B::GlobalBuffer,
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
            activation_quantization,
        } = preparation;
        let rht = ActivationTransform::input_rht(context, data_type, in_place)?;
        let quantizer = activation_quantization
            .map(|quantization| ActivationTransform::quantize(context, data_type, quantization))
            .transpose()?;

        Ok(Self {
            rht_signs,
            rht,
            quantizer,
        })
    }

    pub(super) fn prepare(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        if format == ActivationFormat::Int8
            && let Some(quantizer) = &self.quantizer
        {
            let input_dim = self.input_dim();
            let groups_per_row = input_dim.div_ceil(quantizer.scale_group_size());
            let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, input_dim], DataType::I8)?;
            let mut scales = command_buffer.allocate_scratch_for_shape(&[batch_dim, groups_per_row], DataType::F32)?;
            let mut group_sums = quantizer
                .sum_group_size()
                .map(|group_size| {
                    command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, input_dim.div_ceil(group_size)], DataType::I32)
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
                command_buffer,
            );

            return Ok(LinearInput::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size: quantizer.scale_group_size(),
                code_layout: quantizer.code_layout(),
            });
        }

        let input_dim = self.input_dim();
        let mut transformed = command_buffer.allocate_scratch(input.size())?;
        self.rht.encode_fp(input, &mut transformed, &self.rht_signs, batch_dim, input_dim, command_buffer);
        Ok(LinearInput::FullPrecision(transformed))
    }

    pub(super) fn prepare_in_place(
        &self,
        mut input: B::ScratchBuffer,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        if format == ActivationFormat::Int8 && self.quantizer.is_some() {
            return self.prepare(&input, batch_dim, format, command_buffer);
        }

        let input_dim = self.input_dim();
        self.rht.encode_fp_in_place(&mut input, &self.rht_signs, batch_dim, input_dim, command_buffer);
        Ok(LinearInput::FullPrecision(input))
    }

    fn input_dim(&self) -> u32 {
        (self.rht_signs.size() / size_of::<i32>()) as u32
    }
}
