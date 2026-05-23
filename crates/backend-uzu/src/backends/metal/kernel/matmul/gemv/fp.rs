use super::{kernel::GemvKernel, spec::GemvSpecialization};
use crate::backends::{
    common::{
        AsBufferRangeRef, Buffer, Encoder,
        kernel::matmul::{MatmulArgumentC, MatmulArguments, MatmulError},
    },
    metal::Metal,
};

pub(crate) fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
    kernel: &mut GemvKernel,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<Metal, TB>,
) -> Result<(), MatmulError<Metal>> {
    let MatmulArguments {
        a,
        a_offset,
        b,
        b_offset,
        b_leading_dimension,
        b_transpose,
        ab_scale,
        c,
        d,
        batch_dim,
        input_dim,
        output_dim,
    } = arguments;
    assert!(b_transpose, "encode_gemv does not support b_transpose=false");
    assert!(b_offset == 0, "encode_gemv does not support nonzero b_offset");
    assert!(
        b_leading_dimension.is_none_or(|ld| ld == input_dim),
        "encode_gemv does not support custom b_leading_dimension"
    );

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
