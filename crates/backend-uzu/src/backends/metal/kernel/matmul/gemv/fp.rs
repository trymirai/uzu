use super::{kernel::GemvKernel, spec::GemvSpecialization};
use crate::backends::{
    common::{
        AsBufferRangeRef, Buffer, Encoder,
        kernel::matmul::{MatmulArguments, MatmulB, MatmulDOp, MatmulError},
    },
    metal::Metal,
};

pub(crate) fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
    kernel: &mut GemvKernel,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<'a, Metal, TB>,
) -> Result<(), MatmulError<Metal>> {
    let ab_scale = arguments.d_transform.iter().find_map(|op| op.as_scale()).unwrap_or(1.0);
    let output_bias = arguments.d_transform.iter().find_map(|op| op.as_bias());
    let is_accumulate = arguments.d_transform.contains(&MatmulDOp::Accumulate);

    let MatmulArguments {
        a,
        a_offset,
        b,
        b_offset,
        b_leading_dimension,
        b_transpose,
        d,
        m,
        n,
        k,
        ..
    } = arguments;
    let weights = match b {
        MatmulB::FullPrecision {
            b: w,
        } => w,
        _ => panic!("FP gemv requires FullPrecision B"),
    };
    assert!(b_transpose, "encode_gemv does not support b_transpose=false");
    assert!(b_offset == 0, "encode_gemv does not support nonzero b_offset");
    assert!(b_leading_dimension.is_none_or(|ld| ld == k), "encode_gemv does not support custom b_leading_dimension");

    let specialization = GemvSpecialization::select(k, n, is_accumulate, output_bias.is_some());

    kernel.get_or_create(encoder.context(), specialization)?.encode(
        weights,
        (a, a_offset),
        output_bias,
        d,
        k,
        n,
        k,
        ab_scale,
        m,
        specialization.output_rows_per_threadgroup(),
        encoder,
    );

    Ok(())
}
