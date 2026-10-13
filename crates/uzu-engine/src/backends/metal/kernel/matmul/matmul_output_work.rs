use crate::{
    backends::{
        common::{Backend, BufferMut, BufferRef, kernel::ActivationTransform},
        metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulOutputWork {
    output_rht: ActivationTransform<Metal>,
    output_rht_with_bias: ActivationTransform<Metal>,
}

impl MatmulOutputWork {
    pub fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            output_rht: ActivationTransform::output_rht(context, output_data_type, None, true)?,
            output_rht_with_bias: ActivationTransform::output_rht(
                context,
                output_data_type,
                Some(weights_data_type),
                true,
            )?,
        })
    }

    pub fn apply(
        &self,
        output: impl BufferMut<Backend = Metal>,
        factors: impl BufferRef<Backend = Metal>,
        bias: Option<&<Metal as Backend>::GlobalBuffer>,
        m: u32,
        n: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) {
        let transform = if bias.is_some() {
            &self.output_rht_with_bias
        } else {
            &self.output_rht
        };
        transform.encode_fp_in_place(output, factors, bias, m, n, command_buffer);
    }
}
