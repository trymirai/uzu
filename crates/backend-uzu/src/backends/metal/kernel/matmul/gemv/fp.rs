use crate::backends::{
    common::{
        Encoder,
        kernel::matmul::{MatmulArgumentC, MatmulArguments, MatmulError},
    },
    metal::Metal,
};

use super::{kernel::GemvKernel, spec::GemvSpecialization};

pub(crate) fn encode(
    kernel: &mut GemvKernel,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<Metal>,
) -> Result<(), MatmulError<Metal>> {
    let MatmulArguments {
        a,
        a_offset,
        b,
        ab_scale,
        c,
        d,
        batch_dim,
        input_dim,
        output_dim,
    } = arguments;

    let (is_accumulate, output_bias) = match c {
        MatmulArgumentC::Accumulate => (true, None),
        MatmulArgumentC::Bias(bias) => (false, Some(bias)),
        MatmulArgumentC::None => (false, None),
    };

    let specialization = GemvSpecialization::select(input_dim, output_dim, is_accumulate, output_bias.is_some());

    kernel.get_or_create(encoder.context(), specialization)?.encode(
        b,
        (a, a_offset),
        output_bias,
        d,
        input_dim,
        output_dim,
        input_dim,
        ab_scale,
        batch_dim,
        specialization.output_rows_per_threadgroup(),
        encoder,
    );

    Ok(())
}
