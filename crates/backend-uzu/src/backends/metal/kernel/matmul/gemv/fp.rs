use super::{kernel::GemvKernel, spec::GemvSpecialization};
use crate::backends::{
    common::{
        Encoder,
        gpu_types::gemm::GemmDTransform,
        kernel::matmul::{MatmulArguments, MatmulB, MatmulError, ResolvedDTransform},
    },
    metal::Metal,
};

/// Encodes the FP gemv. Honors SCALE and ACCUMULATE bits via the existing
/// `is_accumulate` + `ab_scale` plumbing, and honors BIAS via the existing
/// `output_bias` plumbing. The RHT bit on the D-transform is left to the
/// caller as a post-pass (this function does not run hadamard).
pub(crate) fn encode<'a>(
    kernel: &mut GemvKernel,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<'a, Metal>,
    resolved_d: ResolvedDTransform<'a, Metal>,
) -> Result<(), MatmulError<Metal>> {
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
    assert!(
        b_leading_dimension.is_none_or(|ld| ld == k),
        "encode_gemv does not support custom b_leading_dimension"
    );

    let is_accumulate = resolved_d.mask.contains(GemmDTransform::ACCUMULATE);
    let output_bias = resolved_d.bias;
    let ab_scale = resolved_d.ab_scale;

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
